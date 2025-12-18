"""
Render the TRAM stickman in MeshCat and export a high-res video in one go.

Usage:
  python scripts/record_meshcat_video.py \
      --seq_folder results/<video> \
      --out meshcat_render.mp4 \
      --width 1280 --height 720 --fps 30

Requires:
  - meshcat >= 0.3
  - ffmpeg in PATH
  - pillow (pip install pillow)
"""

import argparse
import os
import shutil
import subprocess
import tempfile
import sys
from pathlib import Path

# --- ADDED: make `import lib...` work when running from anywhere ----------
# This assumes your repo layout is:
#   <repo_root>/lib/...
#   <repo_root>/scripts/record_meshcat_video.py
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parent.parent  # scripts/.. -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# ------------------------------------------------------------------------

import numpy as np
import torch
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from PIL import Image  # noqa: F401  (vis.get_image returns a PIL.Image)

from lib.models.smpl import SMPL
from lib.vis.traj import traj_filter

# SMPL-24 bones (parent -> child)
BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14), (12, 15),
    (13, 16), (14, 17), (16, 18),
    (17, 19), (18, 20), (19, 21),
    (20, 22), (21, 23),
]


def build_scene(seq_folder, radius=0.025):
    """Compute joints per frame and set up MeshCat objects."""
    img_dir = os.path.join(seq_folder, "images")
    hps_dir = os.path.join(seq_folder, "hps")

    imgfiles = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")])
    if not imgfiles:
        raise RuntimeError("No images found under images/.")
    hps_files = sorted([f for f in os.listdir(hps_dir) if f.endswith(".npy")])
    if not hps_files:
        raise RuntimeError("No hps/*.npy files found.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    smpl = SMPL().to(device)

    per_frame = [[None] * len(hps_files) for _ in range(len(imgfiles))]
    lowest_y = []

    cam = np.load(os.path.join(seq_folder, "camera.npy"), allow_pickle=True).item()
    R_cw = torch.tensor(cam["world_cam_R"]).to(device)
    t_cw = torch.tensor(cam["world_cam_T"]).to(device)

    for tid, fname in enumerate(tqdm(hps_files, desc="tracks")):
        data = np.load(os.path.join(hps_dir, fname), allow_pickle=True).item()
        frames = data["frame"]
        rotmat = data["pred_rotmat"].to(device)
        betas = data["pred_shape"].to(device)
        trans = data["pred_trans"].to(device)

        betas = betas.mean(0, keepdim=True).expand_as(betas)
        trans = trans.squeeze()

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

        Rc, tc = R_cw[frames].to(device), t_cw[frames].to(device)
        joints_w = torch.einsum("bij,bnj->bni", Rc, joints_c) + tc[:, None]

        # traj_filter expects (verts, joints) and returns (verts_f, joints_f)
        _, joints_w = traj_filter(joints_w.cpu(), joints_w.cpu())

        lowest_y.append(joints_w[:, :, 1].min())

        for j, f in enumerate(frames.tolist()):
            per_frame[f][tid] = joints_w[j].numpy()

    # shift so y==0 is the floor
    floor_y = float(np.min(lowest_y))
    for f in range(len(per_frame)):
        for tid in range(len(hps_files)):
            if per_frame[f][tid] is not None:
                per_frame[f][tid][:, 1] -= floor_y

    vis = meshcat.Visualizer().open()
    vis.delete()

    # frame conversion: (x,y,z)_SMPL -> (z,x,y)_MeshCat
    T_smpl_to_meshcat = np.eye(4)
    T_smpl_to_meshcat[:3, :3] = np.array([[0, 0, 1],
                                          [1, 0, 0],
                                          [0, 1, 0]])
    world = vis["world"]
    world.set_transform(T_smpl_to_meshcat)

    # ground plane
    world["ground"].set_object(
        g.Box([4, 0.002, 4]),
        g.MeshLambertMaterial(color=0x555555, opacity=0.3, transparent=True)
    )

    sphere_geom = g.Sphere(radius=radius)
    unit_cyl = g.Cylinder(1.0, radius * 0.35)

    pal = np.loadtxt("data/colors.txt")
    if pal.ndim == 1:
        pal = pal[None, :]
    if pal.shape[1] > 3:
        pal = pal[:, :3]
    if len(pal) < len(hps_files):
        pal = np.vstack([pal] * ((len(hps_files) + len(pal) - 1) // len(pal)))

    bone_nodes = {}
    for tid in range(len(hps_files)):
        r, g_, b = (pal[tid, :] * 255).astype(int)
        col_hex = int((b << 16) + (g_ << 8) + r)
        joint_mat = g.MeshLambertMaterial(color=col_hex)
        bone_mat = g.MeshLambertMaterial(color=col_hex)

        for jid in range(24):
            world[f"t{tid}_j{jid}"].set_object(sphere_geom, joint_mat)

        for bid, (p, c) in enumerate(BONES):
            node = world[f"t{tid}_b{bid}"]
            bone_nodes[(tid, bid)] = node
            node.set_object(unit_cyl, bone_mat)

    return vis, per_frame, bone_nodes


def bone_transform(a, b):
    """Rigid transform mapping unit cylinder (height=1 along +Y, centered) to segment a→b."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a
    height = np.linalg.norm(v)
    if height < 1e-8:
        return tf.translation_matrix([1e6, 1e6, 1e6])  # hide

    y_axis = v / height
    tmp = np.array([1.0, 0.0, 0.0]) if abs(y_axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    x_axis = np.cross(tmp, y_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    z_axis = np.cross(y_axis, x_axis)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)  # columns are basis vectors

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (a + b) / 2.0

    S = np.eye(4)
    S[1, 1] = height  # scale along +Y

    return T @ S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_folder", required=True, help="results/<video>")
    ap.add_argument("--out", default="meshcat_render.mp4", help="output video path")
    ap.add_argument("--width", type=int, default=1280, help="capture width")
    ap.add_argument("--height", type=int, default=720, help="capture height")
    ap.add_argument("--fps", type=int, default=30, help="output fps")
    args = ap.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")

    vis, per_frame, bone_nodes = build_scene(args.seq_folder)

    print("Waiting for MeshCat viewer connection...")
    vis.wait()  # open the MeshCat URL once in your browser
    print("Connected. Recording frames...")

    tmpdir = tempfile.mkdtemp(prefix="meshcat_frames_")
    print(f"Saving frames to {tmpdir}")

    hidden = tf.translation_matrix([1e6, 1e6, 1e6])

    try:
        for f, frame in enumerate(tqdm(per_frame, desc="frames")):
            for tid, joints in enumerate(frame):
                if joints is None:
                    for jid in range(24):
                        vis[f"world/t{tid}_j{jid}"].set_transform(hidden)
                    for bid in range(len(BONES)):
                        bone_nodes[(tid, bid)].set_transform(hidden)
                    continue

                for jid in range(24):
                    vis[f"world/t{tid}_j{jid}"].set_transform(tf.translation_matrix(joints[jid]))

                for bid, (p, c) in enumerate(BONES):
                    bone_nodes[(tid, bid)].set_transform(bone_transform(joints[p], joints[c]))

            frame_path = os.path.join(tmpdir, f"frame_{f:05d}.png")
            img = vis.get_image(args.width, args.height)  # returns PIL.Image
            img.save(frame_path)

        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-framerate", str(args.fps),
            "-i", os.path.join(tmpdir, "frame_%05d.png"),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            args.out,
        ]
        print("Running ffmpeg:", " ".join(ffmpeg_cmd))
        subprocess.check_call(ffmpeg_cmd)
        print(f"Wrote {args.out}")

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    main()
