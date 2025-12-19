import argparse
import json
import os
import sys
from glob import glob

import numpy as np
import torch
from scipy.ndimage import gaussian_filter
import cv2

# ensure repo root is on path
sys.path.insert(0, os.path.dirname(__file__) + "/..")

from lib.models.smpl import SMPL
# SMPL 24-joint order used by the model (after mapping to 24 GT joints)
SMPL_24_NAMES = [
    "Pelvis",          # 0
    "Left Hip",        # 1
    "Right Hip",       # 2
    "Spine1",          # 3
    "Left Knee",       # 4
    "Right Knee",      # 5
    "Spine2",          # 6
    "Left Ankle",      # 7
    "Right Ankle",     # 8
    "Spine3",          # 9
    "Left Foot",       # 10
    "Right Foot",      # 11
    "Neck",            # 12
    "Left Collar",     # 13
    "Right Collar",    # 14
    "Head",            # 15
    "Left Shoulder",   # 16
    "Right Shoulder",  # 17
    "Left Elbow",      # 18
    "Right Elbow",     # 19
    "Left Wrist",      # 20
    "Right Wrist",     # 21
    "Left Hand",       # 22
    "Right Hand",      # 23
]


def compute_world_joints(seq_folder, device="cuda"):
    """Load per-track HPS files, project SMPL joints into world space, and return skeleton."""
    hps_files = sorted(glob(os.path.join(seq_folder, "hps", "*.npy")))
    if not hps_files:
        raise RuntimeError("No hps/*.npy files found. Run estimate_humans first.")

    cam = np.load(os.path.join(seq_folder, "camera.npy"), allow_pickle=True).item()
    R_cw = torch.tensor(cam["world_cam_R"]).to(device)
    t_cw = torch.tensor(cam["world_cam_T"]).to(device)

    smpl = SMPL().to(device)

    per_frame_tracks = {}
    params_list = []
    lowest_y = []
    # skeleton from SMPL parents (one-time)
    parents = getattr(smpl, "parents", None)
    if parents is not None:
        parents = parents.tolist()
    else:
        # fallback to standard SMPL kinematic tree (24 joints)
        parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]
    bones = [[int(p), i] for i, p in enumerate(parents) if p >= 0]

    for tid, npy in enumerate(hps_files):
        data = np.load(npy, allow_pickle=True).item()
        frames = torch.tensor(data["frame"]).long()
        rotmat = data["pred_rotmat"].to(device)
        betas = data["pred_shape"].to(device)
        trans = data["pred_trans"].to(device).squeeze()

        # Use mean shape per track for stability
        betas = betas.mean(0, keepdim=True).expand_as(betas)

        with torch.no_grad():
            out = smpl(
                body_pose=rotmat[:, 1:],
                global_orient=rotmat[:, [0]],
                betas=betas,
                transl=trans,
                pose2rot=False,
                default_smpl=True,
            )
            joints_c = out.joints[:, :24]

        Rc = R_cw[frames]
        tc = t_cw[frames]
        joints_w = torch.einsum("bij,bnj->bni", Rc, joints_c) + tc[:, None]

        # Smooth root trajectory (same as traj_filter but only for joints)
        joints_w = joints_w.cpu()
        root = joints_w[:, 0]
        root_smooth = torch.from_numpy(gaussian_filter(root, sigma=3, axes=0))
        joints_w = joints_w + (root_smooth - root)[:, None]

        lowest_y.append(joints_w[:, :, 1].min())

        for j, f in enumerate(frames.tolist()):
            per_frame_tracks.setdefault(f, []).append(
                {
                    "track_id": tid,
                    "joints_world": joints_w[j].tolist(),
                    "parameters": {
                        "pred_rotmat": rotmat[j].cpu().numpy().tolist(),
                        "pred_shape": betas[j].cpu().numpy().tolist(),
                        "pred_trans": trans[j].cpu().numpy().tolist(),
                    },
                }
            )
            params_list.append(
                {
                    "pred_rotmat": rotmat[j].cpu().numpy(),
                    "pred_shape": betas[j].cpu().numpy(),
                    "pred_trans": trans[j].cpu().numpy(),
                }
            )

    # Shift so the lowest point sits on the floor (y=0)
    floor_y = float(np.min(lowest_y))
    for frame_tracks in per_frame_tracks.values():
        for tr in frame_tracks:
            for joint in tr["joints_world"]:
                joint[1] -= floor_y

    # joint names for downstream consumption (SMPL 24 order)
    joint_names = SMPL_24_NAMES

    return per_frame_tracks, params_list, bones, joint_names, floor_y


def project_points(joints_world, R, T, focal, cx, cy):
    """Project world-space joints to image pixels."""
    j_world = torch.tensor(joints_world)
    if j_world.ndim == 2:
        j_world = j_world[None]  # (1,24,3)
    j_cam = torch.einsum("ij,bnj->bni", torch.tensor(R), j_world) + torch.tensor(T)[None, None]
    x = focal * (j_cam[..., 0] / j_cam[..., 2]) + cx
    y = focal * (j_cam[..., 1] / j_cam[..., 2]) + cy
    pix = torch.stack([x, y], dim=-1)
    return pix.squeeze(0).cpu().numpy().astype(float)  # (24,2)


def load_contacts(seq_folder, num_frames):
    """Return per-frame contact labels using foot_boxes.npy if present."""
    contacts_path = os.path.join(seq_folder, "foot_boxes.npy")
    if os.path.isfile(contacts_path):
        fb = np.load(contacts_path, allow_pickle=True)
        labels = []
        for i in range(num_frames):
            has_contact = i < len(fb) and fb[i].shape[0] > 0
            labels.append("full_contact" if has_contact else "no_contact")
        return labels
    # Fallback: all no contact
    return ["no_contact"] * num_frames


def assign_boxes_to_feet(foot_boxes, foot_mids):
    """Assign each box to nearest foot midpoint; return best dist/conf per foot."""
    best = {
        "left": {"dist": None, "conf": None},
        "right": {"dist": None, "conf": None},
    }
    if foot_boxes is None or len(foot_boxes) == 0:
        return best

    centers = np.column_stack(
        [((foot_boxes[:, 0] + foot_boxes[:, 2]) / 2.0), ((foot_boxes[:, 1] + foot_boxes[:, 3]) / 2.0)]
    )
    centers = np.round(centers).astype(int)
    confs = foot_boxes[:, 4] if foot_boxes.shape[1] > 4 else np.ones(len(foot_boxes), dtype=float)

    for (cxcy, conf) in zip(centers, confs):
        d_left = np.linalg.norm(cxcy - foot_mids["left"])
        d_right = np.linalg.norm(cxcy - foot_mids["right"])
        if d_left <= d_right:
            if best["left"]["dist"] is None or d_left < best["left"]["dist"]:
                best["left"]["dist"] = float(d_left)
                best["left"]["conf"] = float(conf)
        else:
            if best["right"]["dist"] is None or d_right < best["right"]["dist"]:
                best["right"]["dist"] = float(d_right)
                best["right"]["conf"] = float(conf)
    return best


def smooth_contact_flags(flags, fill_gap=1, drop_spike=1):
    """Fill short gaps and drop short spikes in a boolean contact sequence."""
    if not flags:
        return flags

    cleaned = flags[:]
    n = len(flags)

    # Fill short gaps of no_contact between contacts.
    i = 0
    while i < n:
        if cleaned[i]:
            i += 1
            continue
        start = i
        while i < n and not cleaned[i]:
            i += 1
        end = i
        gap_len = end - start
        left = start - 1
        right = end
        if gap_len <= fill_gap and left >= 0 and right < n and cleaned[left] and cleaned[right]:
            for j in range(start, end):
                cleaned[j] = True

    # Drop short spikes of contact between no_contact.
    i = 0
    while i < n:
        if not cleaned[i]:
            i += 1
            continue
        start = i
        while i < n and cleaned[i]:
            i += 1
        end = i
        run_len = end - start
        left = start - 1
        right = end
        if run_len <= drop_spike and left >= 0 and right < n and (not cleaned[left]) and (not cleaned[right]):
            for j in range(start, end):
                cleaned[j] = False

    return cleaned


def summarize_params(params_list):
    """Compute mean/std over all frames/tracks for each parameter key."""
    if not params_list:
        return {}, {}
    keys = params_list[0].keys()
    avg, std = {}, {}
    for k in keys:
        arr = np.stack([p[k] for p in params_list], axis=0)
        avg[k] = arr.mean(axis=0).tolist()
        std[k] = arr.std(axis=0).tolist()
    return avg, std


def export_json(seq_folder, output_name="trajectories_with_contacts.json"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    max_contact_dist = None  # disabled: no distance gating
    min_box_conf = None  # disabled: no confidence gating
    fill_gap = 1  # fill gaps of <= this many frames
    drop_spike = 1  # drop isolated spikes of <= this many frames

    # Determine frame count from images or camera extrinsics
    imgfiles = sorted(glob(os.path.join(seq_folder, "images", "*.jpg")))
    num_frames = len(imgfiles)
    if num_frames == 0:
        raise RuntimeError("No images found under images/.")

    per_frame_tracks, params_list, bones, joint_names, floor_y = compute_world_joints(seq_folder, device=device)
    avg_params, std_params = summarize_params(params_list)

    # image size and camera intrinsics (for projection)
    img0 = cv2.imread(imgfiles[0])
    h, w = img0.shape[:2]
    cam = np.load(os.path.join(seq_folder, "camera.npy"), allow_pickle=True).item()
    focal = cam.get("img_focal", w * 1.0)  # fallback
    img_center = cam.get("img_center", np.array([w / 2.0, h / 2.0]))
    cx, cy = float(img_center[0]), float(img_center[1])
    world_cam_R = np.asarray(cam["world_cam_R"])
    world_cam_T = np.asarray(cam["world_cam_T"])
    # align camera translation with the floor offset applied to joints
    world_cam_T_shift = world_cam_T - np.array([0.0, floor_y, 0.0])

    # follow visualize_tram: view camera = inverse of world, with a slight retreat
    view_cam_R = np.transpose(world_cam_R, (0, 2, 1))
    view_cam_T = -np.einsum("bij,bj->bi", world_cam_R, world_cam_T_shift)
    # CAM_BACK retreat disabled for testing alignment without it
    # CAM_BACK = -2.0
    # forward_w = -view_cam_R[:, 2].copy()
    # horiz_w = forward_w.copy()
    # horiz_w[:, 1] = 0
    # norm = np.linalg.norm(horiz_w, axis=1, keepdims=True) + 1e-8
    # horiz_w = horiz_w / norm
    # view_cam_T = view_cam_T - CAM_BACK * horiz_w

    contacts_per_frame = []
    raw_flags = []
    contacts_path = os.path.join(seq_folder, "foot_boxes.npy")
    fb_array = None
    if os.path.isfile(contacts_path):
        fb_array = np.load(contacts_path, allow_pickle=True)

    for f in range(num_frames):
        # Default contacts if no tracks or no boxes
        tracks_here = per_frame_tracks.get(f, [])
        if not tracks_here:
            contacts_per_frame.append(
                {
                    "left": {"status": "no_contact", "distance_px": None},
                    "right": {"status": "no_contact", "distance_px": None},
                }
            )
            raw_flags.append({"left": False, "right": False})
        else:
            # choose the first track (lowest id) for contact association
            track = sorted(tracks_here, key=lambda t: t["track_id"])[0]
            joints_world = track["joints_world"]
            pix = project_points(joints_world, view_cam_R[f], view_cam_T[f], focal, cx, cy)

            # midpoints between ankle and foot joints
            foot_mids = {
                "left": ((pix[7] + pix[10]) / 2.0).astype(float),
                "right": ((pix[8] + pix[11]) / 2.0).astype(float),
            }

            fb = fb_array[f] if (fb_array is not None and f < len(fb_array)) else []
            best = assign_boxes_to_feet(fb, foot_mids)
            raw_left = best["left"]["dist"] is not None
            raw_right = best["right"]["dist"] is not None

            contacts_per_frame.append(
                {
                    "left": {
                        "status": "full_contact" if raw_left else "no_contact",
                        "distance_px": best["left"]["dist"],
                    },
                    "right": {
                        "status": "full_contact" if raw_right else "no_contact",
                        "distance_px": best["right"]["dist"],
                    },
                }
            )
            raw_flags.append({"left": raw_left, "right": raw_right})

    left_flags = smooth_contact_flags([rf["left"] for rf in raw_flags], fill_gap=fill_gap, drop_spike=drop_spike)
    right_flags = smooth_contact_flags([rf["right"] for rf in raw_flags], fill_gap=fill_gap, drop_spike=drop_spike)

    frames = []
    for f in range(num_frames):
        contacts_per_frame[f]["left"]["status"] = "full_contact" if left_flags[f] else "no_contact"
        contacts_per_frame[f]["right"]["status"] = "full_contact" if right_flags[f] else "no_contact"
        frames.append(
            {
                "frame": f,
                "contact": "full_contact" if (contacts_per_frame[f]["left"]["status"] == "full_contact" or contacts_per_frame[f]["right"]["status"] == "full_contact") else "no_contact",
                "contacts": contacts_per_frame[f],
                "tracks": per_frame_tracks.get(f, []),
            }
        )

    data = {
        "frames": frames,
        "average_parameters": avg_params,
        "std_parameters": std_params,
        "shapes": {
            "num_frames": num_frames,
            "joints_per_track": 24,
            "tracks_total": len(set([t["track_id"] for ft in per_frame_tracks.values() for t in ft])) if per_frame_tracks else 0,
            "parameter_frames": len(params_list),
        },
        "bones": bones,
        "joint_names": [{ "id": i, "name": n } for i, n in enumerate(joint_names)],
        "camera": {
            "img_focal": float(focal),
            "img_center": [float(cx), float(cy)],
            "world_cam_R": world_cam_R.tolist(),
            "world_cam_T": world_cam_T_shift.tolist(),
            "view_cam_R": view_cam_R.tolist(),
            "view_cam_T": view_cam_T.tolist(),
        },
    }

    out_path = os.path.join(seq_folder, output_name)
    with open(out_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Export TRAM joints + contacts to JSON.")
    ap.add_argument(
        "--seq_folder",
        required=True,
        help="Path to results/<video_basename> (must contain images/, hps/, camera.npy).",
    )
    ap.add_argument(
        "--output_name",
        default="trajectories_with_contacts.json",
        help="Output JSON filename (saved inside seq_folder).",
    )
    args = ap.parse_args()

    export_json(args.seq_folder, output_name=args.output_name)
