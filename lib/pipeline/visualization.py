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

def visualize_tram(seq_folder, annotations, floor_scale=5, bin_size=-1, max_faces_per_bin=30000):
    img_folder = f'{seq_folder}/images'
    hps_folder = f'{seq_folder}/hps'
    imgfiles = sorted(glob(f'{img_folder}/*.jpg'))
    hps_files = sorted(glob(f'{hps_folder}/*.npy'))
    
    
    fname_to_id, bboxes_by_id = load_bboxes(annotations)
 
    

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
    CAM_BACK = -2.0     # metres you want to retreat

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
    writer = imageio.get_writer(f'{seq_folder}/tram_output.mp4', fps=30, mode='I', 
                                format='FFMPEG', macro_block_size=1)
    img = cv2.imread(imgfiles[0])
    renderer = Renderer(img.shape[1], img.shape[0], img_focal-100, 'cuda', 
                        smpl.faces, bin_size=bin_size, max_faces_per_bin=max_faces_per_bin)
    renderer.set_ground(scale, cx.item(), cz.item())
    
    
    # ---------------------------------------------------------------
    disk_radius  = 10       # pixels          

    

    for i in tqdm(range(len(imgfiles))):
        img = cv2.imread(imgfiles[i])[:, :, ::-1].copy()
        
        
        # overlay bboxes (Based on COCO, will change later)
        fname = os.path.basename(imgfiles[i])
        img_id = fname_to_id.get(fname)
        bbox_centers = []
        if img_id is not None:
            for x, y, w, h in bboxes_by_id.get(img_id, []):
                x, y, w, h = map(int, (x, y, w, h))
                bx, by = x + w//2, y + h//2
                bbox_centers.append((bx, by)) 
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)

        
        
        verts_list = track_verts[i]
        if len(verts_list)>0:
            verts_list = torch.stack(track_verts[i])[:,None].to('cuda')
            verts_list -= offset
            
            tid = track_tid[i]
            verts_colors = torch.stack([colors[t] for t in tid]).to('cuda')


        # ---------------------------------------------------------------------------- #
        # 3-D world → camera coordinates
        pix_int = None 
        if len(track_joints[i]):
            j_world = torch.stack(track_joints[i]).to(device) - offset   # (N,24,3)

            R = view_cam_R[i]                                   # (3,3)
            T = view_cam_T[i]                                   # (3,)

            j_cam = torch.einsum('ij,nkj->nki', R, j_world) + T # (N,24,3)

            x = img_focal * (j_cam[..., 0] / j_cam[..., 2]) + img.shape[1] / 2
            y = img_focal * (j_cam[..., 1] / j_cam[..., 2]) + img.shape[0] / 2
            pix_int = torch.stack([x, y], -1).round().to(torch.int32)     # (N,24,2)


        # ---------------------------------------------------------------------------- #
    
    
        faces = renderer.faces.clone().squeeze(0)
        cameras, lights = renderer.create_camera_from_cv(view_cam_R[[i]], 
                                                        view_cam_T[[i]])
        rend = renderer.render_with_ground_multiple(verts_list, faces, verts_colors, 
                                                    cameras, lights)
        
        out = np.concatenate([img, rend], axis=1)
        
        
        joints_list = []
        # ---------------------------------------------------------------------------- #
        # draw the ankles and foor joints
        if pix_int is not None:
            la = pix_int[0, 10]   # first body, joint 10   
            cv2.circle(out,
                    (la[0].item(), la[1].item()),
                    disk_radius,
                    (0, 0, 255),   # yellow in BGR
                    thickness=-1)
            
    
        if pix_int is not None:
            ra = pix_int[0, 11]
            cv2.circle(out,
                    (ra[0].item(), ra[1].item()),
                    disk_radius,
                    (0, 255, 0),   # yellow in BGR
                    thickness=-1)


        if pix_int is not None:
            lf = pix_int[0, 7]   
            cv2.circle(out,
                    (lf[0].item(), lf[1].item()),
                    disk_radius,
                    (0, 255, 255),   # yellow in BGR
                    thickness=-1)
            

        if pix_int is not None:
            rf = pix_int[0, 8]                               
            cv2.circle(out,
                    (rf[0].item(), rf[1].item()),
                    disk_radius,
                    (255, 0, 0),   # yellow in BGR
                    thickness=-1)
            
        # ---------------------------------------------------------------------------- #
        # get the mid point between those joints on each leg. This seems to be more robust in case of joint misalignment 
        joints_list = []
        if pix_int is not None and pix_int.shape[0] > 0:
            # left foot box: joint 10 ↔ ankle 7
            joints_list.append((pix_int[0,10], pix_int[0,7]))
            # right foot box: joint 11 ↔ ankle 8
            joints_list.append((pix_int[0,11], pix_int[0,8]))

        # ---------------------------------------------------------------------------- #
        # now for each bbox, find the nearest foot‐box center and draw a line
        for bx, by in bbox_centers:
            best_dist   = float('inf')
            best_center = None
            for (j1, j2) in joints_list:
                # midpoint of this foot‐ankle pair
                fx = (int(j1[0]) + int(j2[0])) // 2
                fy = (int(j1[1]) + int(j2[1])) // 2
                # Euclidean distance to bbox center
                d = ((bx - fx)**2 + (by - fy)**2)**0.5
                if d < best_dist:
                    best_dist   = d
                    best_center = (fx, fy)
            if best_center is not None:
                cv2.line(out, (bx, by), best_center, (255,255,0), 2)
                text = f"{best_dist:.1f}"
                # choose a position a few pixels above the bbox center to put the distance in pixels:
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
        # ---------------------------------------------------------------------------- #

        
        writer.append_data(out)

    writer.close()