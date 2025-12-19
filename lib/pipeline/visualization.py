import numpy as np
import os
import torch
import json
import cv2
from tqdm import tqdm
from glob import glob
import imageio
from collections import defaultdict

from lib.vis.traj import *
from lib.models.smpl import SMPL
from lib.vis.renderer import Renderer
from itertools import pairwise


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


def load_bboxes(coco_json_path):
    """Returns two dicts: filename→image_id, and image_id→list of [x,y,w,h]."""
    coco = json.load(open(coco_json_path, "r"))
    fname_to_id = {img["file_name"]: img["id"] for img in coco.get("images", [])}
    bboxes_by_id = defaultdict(list)
    for ann in coco.get("annotations", []):
        img_id = ann.get("image_id")
        bbox = ann.get("bbox", [])
        if img_id is not None and bbox:
            bboxes_by_id[img_id].append(bbox)
    return fname_to_id, bboxes_by_id


def load_foot_bboxes(seq_folder):
    """Returns dict: filename → array of foot bboxes (N×5), or (0×5) if none."""
    # 1) Load the object‑dtype array
    fb = np.load(os.path.join(seq_folder, 'foot_boxes.npy'), allow_pickle=True)
    # 2) Gather your image list
    img_folder = os.path.join(seq_folder, 'images')
    imgfiles   = sorted(glob(f'{img_folder}/*.jpg'))
    assert len(fb) == len(imgfiles), \
        f"foot_boxes.npy has {len(fb)} entries but found {len(imgfiles)} images"
    # 3) Map basename→foot_boxes
    return {
        os.path.basename(imgfiles[i]): fb[i]
        for i in range(len(fb))
    }

def visualize_tram(seq_folder, annotations, floor_scale=5, bin_size=-1, max_faces_per_bin=30000, draw_contacts=True, fps=30):
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))
    
    
    # fname_to_id, bboxes_by_id = load_bboxes(annotations)
    
    
        # optionally load foot‐boxes
    if draw_contacts:
        foot_by_fname = load_foot_bboxes(seq_folder)
    else:
        foot_by_fname = {}
 
    

    device = 'cuda'
    smpl = SMPL().to(device)
    colors = np.loadtxt('data/colors.txt')/255
    colors = torch.from_numpy(colors).float()

    max_track = len(hps_files)
    tstamp =  [t for t in range(len(imgfiles))]
    track_verts = {i:[] for i in tstamp}
    track_joints = {i:[] for i in tstamp}
    track_tid = {i:[] for i in tstamp}
    locations = []
    lowest = []

    ##### TRAM + VIMO #####
    pred_cam = np.load(f'{seq_folder}/camera.npy', allow_pickle=True).item()
    img_focal = pred_cam['img_focal'].item()
    img_center = pred_cam.get('img_center', None)
    world_cam_R = torch.tensor(pred_cam['world_cam_R']).to(device)
    world_cam_T = torch.tensor(pred_cam['world_cam_T']).to(device)

    for i in range(max_track):
        hps_file = hps_files[i]

        pred_smpl = np.load(hps_file, allow_pickle=True).item()
        pred_rotmat = pred_smpl['pred_rotmat'].to(device)
        pred_shape = pred_smpl['pred_shape'].to(device)
        pred_trans = pred_smpl['pred_trans'].to(device)
        frame = pred_smpl['frame']

        mean_shape = pred_shape.mean(dim=0, keepdim=True)
        pred_shape = mean_shape.repeat(len(pred_shape), 1)

        pred = smpl(body_pose=pred_rotmat[:,1:], 
                    global_orient=pred_rotmat[:,[0]], 
                    betas=pred_shape, 
                    transl=pred_trans.squeeze(),
                    pose2rot=False, 
                    default_smpl=True)
        pred_vert = pred.vertices
        pred_j3d = pred.joints[:, :24]

        cam_r = world_cam_R[frame]
        cam_t = world_cam_T[frame]

        pred_vert_w = torch.einsum('bij,bnj->bni', cam_r, pred_vert) + cam_t[:,None]
        pred_j3d_w = torch.einsum('bij,bnj->bni', cam_r, pred_j3d) + cam_t[:,None]
        pred_vert_w, pred_j3d_w = traj_filter(pred_vert_w.cpu(), 
                                            pred_j3d_w.cpu())
        locations.append(pred_j3d_w.mean(1))
        lowest.append(pred_vert_w[:, :, 1].min())

        for j, f in enumerate(frame.tolist()):
            track_tid[f].append(i)
            track_verts[f].append(pred_vert_w[j])
            track_joints[f].append(pred_j3d_w[j])


    offset = torch.min(torch.stack(lowest))
    offset = torch.tensor([0, offset, 0]).to(device)

    locations = torch.cat(locations).to(device)
    cx, cz = (locations.max(0)[0] + locations.min(0)[0])[[0, 2]] / 2.0
    sx, sz = (locations.max(0)[0] - locations.min(0)[0])[[0, 2]]
    scale = max(sx.item(), sz.item()) * floor_scale

    ##### Viewing Camera #####
    world_cam_T = world_cam_T - offset
    view_cam_R  = world_cam_R.mT.to('cuda')
    view_cam_T  = - torch.einsum('bij,bj->bi', world_cam_R, world_cam_T).to('cuda')
    
    
    # ---------------------------------- IBRICS ---------------------------------- #
    CAM_BACK = 0.0      # no retreat so render view matches input camera

    # camera forward (–Z in camera space) expressed in world coords
    forward_w = -view_cam_R[:, 2]               # (T,3)

    # project onto ground plane  ⇒  wipe out Y component
    horiz_w = forward_w.clone()
    horiz_w[:, 1] = 0                     # zero the vertical

    # normalise so we move exactly CAM_BACK metres
    horiz_w = torch.nn.functional.normalize(horiz_w, dim=1, eps=1e-8)

    # apply translation: t_wc ← t_wc  –  CAM_BACK · horiz_w
    view_cam_T = view_cam_T - CAM_BACK * horiz_w
        
        

    ##### Render video for visualization #####
    writer = imageio.get_writer(f'{seq_folder}/tram_output.mp4', fps=fps, mode='I', 
                                format='FFMPEG', macro_block_size=1)
    img = cv2.imread(imgfiles[0])
    renderer = Renderer(img.shape[1], img.shape[0], img_focal-100, 'cuda', 
                        smpl.faces, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
    renderer.set_ground(scale, cx.item(), cz.item())
    
    
    # ---------------------------------------------------------------
    disk_radius  = 10       # pixels
    max_contact_dist = None # disabled: no distance gating
    min_box_conf = None     # disabled: no confidence gating
    fill_gap = 1            # fill gaps of <= this many frames
    drop_spike = 1          # drop isolated spikes of <= this many frames

    # Precompute raw + smoothed contact flags per foot for overlay.
    raw_flags = []
    if draw_contacts:
        for i in range(len(imgfiles)):
            if len(track_joints[i]) == 0:
                raw_flags.append({"left": False, "right": False})
                continue

            j_world = track_joints[i][0].to(device) - offset  # use first track
            R = view_cam_R[i]
            T = view_cam_T[i]

            j_cam = torch.einsum('ij,kj->ki', R, j_world) + T
            if img_center is not None:
                cx, cy = float(img_center[0]), float(img_center[1])
            else:
                cx, cy = img.shape[1] / 2, img.shape[0] / 2
            x = img_focal * (j_cam[:, 0] / j_cam[:, 2]) + cx
            y = img_focal * (j_cam[:, 1] / j_cam[:, 2]) + cy
            pix = torch.stack([x, y], -1).cpu().numpy()

            foot_mids = {
                "left": ((pix[7] + pix[10]) / 2.0).astype(float),
                "right": ((pix[8] + pix[11]) / 2.0).astype(float),
            }

            fb = foot_by_fname.get(os.path.basename(imgfiles[i]), np.zeros((0, 5)))
            best = assign_boxes_to_feet(fb, foot_mids)
            raw_left = best["left"]["dist"] is not None
            raw_right = best["right"]["dist"] is not None
            raw_flags.append({"left": raw_left, "right": raw_right})

        left_flags = smooth_contact_flags([rf["left"] for rf in raw_flags], fill_gap=fill_gap, drop_spike=drop_spike)
        right_flags = smooth_contact_flags([rf["right"] for rf in raw_flags], fill_gap=fill_gap, drop_spike=drop_spike)
    else:
        left_flags = []
        right_flags = []

    

    for i in tqdm(range(len(imgfiles))):
        img = cv2.imread(imgfiles[i])[:, :, ::-1].copy()
        
        bbox_centers = []
        
        # overlay bboxes (Based on COCO, will change later)
        # fname = os.path.basename(imgfiles[i])
        # img_id = fname_to_id.get(fname)
        # bbox_centers = []
        # if img_id is not None:
        #     for x, y, w, h in bboxes_by_id.get(img_id, []):
        #         x, y, w, h = map(int, (x, y, w, h))
        #         bx, by = x + w//2, y + h//2
        #         bbox_centers.append((bx, by)) 
        #         cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        
        
        if draw_contacts:
            fb = foot_by_fname.get(os.path.basename(imgfiles[i]), np.zeros((0, 5)))
            if fb.shape[0] == 0:
                cv2.putText(img, "no contact", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                for x1, y1, x2, y2, conf in fb:
                    # draw red rectangle
                    cv2.rectangle(
                        img,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 0, 255), 2
                    )
                    cv2.putText(
                        img,
                        f"{conf:.2f}",
                        (int(x1), int(y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA
                    )
                    # record its centre for the distance step
                    bx = int((x1 + x2) / 2)
                    by = int((y1 + y2) / 2)
                    bbox_centers.append((bx, by))

            # no per-frame status text; use bottom markers instead
        
        verts_list = track_verts[i]
        if len(verts_list)>0:
            verts_list = torch.stack(track_verts[i])[:,None].to('cuda')
            verts_list -= offset
            
            tid = track_tid[i]
            verts_colors = torch.stack([colors[t] for t in tid]).to('cuda')


        # ---------------------------------------------------------------------------- #
        # 3-D world → camera coordinates
        pix = None
        pix_draw = None
        if len(track_joints[i]):
            j_world = torch.stack(track_joints[i]).to(device) - offset   # (N,24,3)

            R = view_cam_R[i]                                   # (3,3)
            T = view_cam_T[i]                                   # (3,)

            j_cam = torch.einsum('ij,nkj->nki', R, j_world) + T # (N,24,3)

            if img_center is not None:
                cx, cy = float(img_center[0]), float(img_center[1])
            else:
                cx, cy = img.shape[1] / 2, img.shape[0] / 2
            x = img_focal * (j_cam[..., 0] / j_cam[..., 2]) + cx
            y = img_focal * (j_cam[..., 1] / j_cam[..., 2]) + cy
            pix = torch.stack([x, y], -1)     # float pixels
            pix_draw = pix.round().to(torch.int32)


        # ---------------------------------------------------------------------------- #
    
    
        faces = renderer.faces.clone().squeeze(0)
        cameras, lights = renderer.create_camera_from_cv(view_cam_R[[i]], 
                                                        view_cam_T[[i]])
        rend = renderer.render_with_ground_multiple(verts_list, faces, verts_colors, 
                                                    cameras, lights)
        
        out = np.concatenate([img, rend], axis=1)
        
        
        joints_list = []
        contact_points = []  # for drawing on the rendered view
        # ---------------------------------------------------------------------------- #
        # draw the ankles and foor joints
        if pix_draw is not None:
            la = pix_draw[0, 10]   # first body, joint 10   
            cv2.circle(out,
                    (la[0].item(), la[1].item()),
                    disk_radius,
                    (0, 0, 255),   # yellow in BGR
                    thickness=-1)
            
    
        if pix_draw is not None:
            ra = pix_draw[0, 11]
            cv2.circle(out,
                    (ra[0].item(), ra[1].item()),
                    disk_radius,
                    (0, 255, 0),   # yellow in BGR
                    thickness=-1)


        if pix_draw is not None:
            lf = pix_draw[0, 7]   
            cv2.circle(out,
                    (lf[0].item(), lf[1].item()),
                    disk_radius,
                    (0, 255, 255),   # yellow in BGR
                    thickness=-1)
            

        if pix_draw is not None:
            rf = pix_draw[0, 8]                               
            cv2.circle(out,
                    (rf[0].item(), rf[1].item()),
                    disk_radius,
                    (255, 0, 0),   # yellow in BGR
                    thickness=-1)
            
        # ---------------------------------------------------------------------------- #
        # get the mid point between those joints on each leg. This seems to be more robust in case of joint misalignment 
        joints_list = []
        if pix is not None and pix.shape[0] > 0:
            # left foot box: joint 10 ↔ ankle 7
            joints_list.append((pix[0,10], pix[0,7]))
            # right foot box: joint 11 ↔ ankle 8
            joints_list.append((pix[0,11], pix[0,8]))

        # ---------------------------------------------------------------------------- #
        # now for each bbox, assign it to the closest foot midpoint and draw a line
        foot_mids = []
        for (j1, j2) in joints_list:
            fx = (float(j1[0]) + float(j2[0])) / 2.0
            fy = (float(j1[1]) + float(j2[1])) / 2.0
            foot_mids.append((fx, fy))

        # ---------------------------------------------------------------------------- #
        # draw raw vs smoothed contact markers along the bottom, aligned to x only
        if draw_contacts and foot_mids and i < len(raw_flags):
            raw_left = raw_flags[i]["left"]
            raw_right = raw_flags[i]["right"]
            sm_left = left_flags[i]
            sm_right = right_flags[i]

            # left foot = index 0, right foot = index 1
            states = [
                ("left", foot_mids[0][0], raw_left, sm_left),
                ("right", foot_mids[1][0], raw_right, sm_right),
            ]

            y_base = img.shape[0] - 15
            for _, fx, raw_on, sm_on in states:
                raw_color = (0, 165, 255) if raw_on else (0, 0, 255)
                sm_color = (0, 255, 0) if sm_on else (0, 0, 255)
                cv2.circle(out, (int(fx), y_base - 10), 6, raw_color, -1)
                cv2.circle(out, (int(fx), y_base), 6, sm_color, -1)

        for bx, by in bbox_centers:
            if not foot_mids:
                continue
            dists = [((bx - fx)**2 + (by - fy)**2)**0.5 for fx, fy in foot_mids]
            best_id = int(np.argmin(dists))
            best_dist = dists[best_id]
            best_center = foot_mids[best_id]

            if best_center is not None and (max_contact_dist is None or best_dist <= max_contact_dist):
                bc_int = (int(best_center[0]), int(best_center[1]))
                cv2.line(out, (bx, by), bc_int, (255,255,0), 2)
                text = f"{best_dist:.1f}"
                text_pos = (bx - 10, by - 10)
                cv2.putText(
                    out,
                    text,
                    text_pos,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(255,255,0),
                    thickness=1,
                    lineType=cv2.LINE_AA
                )
                contact_points.append({"pt": bc_int, "dist": best_dist, "foot": "left" if best_id == 0 else "right"})
        # ---------------------------------------------------------------------------- #

        # draw contact points on the rendered (right) view as well
        if contact_points:
            x_offset = img.shape[1]  # width of the original frame
            for cp in contact_points:
                cx, cy = cp["pt"]
                cv2.circle(out, (int(cx + x_offset), int(cy)), 6, (0, 0, 255), -1)
                cv2.putText(
                    out,
                    f"{cp['dist']:.1f}",
                    (int(cx + x_offset + 8), int(cy) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA
                )
        else:
            x_offset = img.shape[1]
            if draw_contacts and i < len(raw_flags) and (left_flags[i] or right_flags[i]):
                cv2.putText(
                    out,
                    "smoothed contact",
                    (int(10 + x_offset), 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    out,
                    "no contact",
                    (int(10 + x_offset), 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

        writer.append_data(out)

    writer.close()
