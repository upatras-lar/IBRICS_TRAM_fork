import numpy as np
import torch, time, os, argparse
from glob import glob
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from lib.vis.traj import traj_filter
from lib.models.smpl import SMPL


# ────────────────────────────────────────────────────────────────────────
def stream_tram_joints(
        seq_folder: str,
        sphere_radius: float = 0.025,
        fps: int = 30
):
    """Real-time MeshCat streamer for TRAM – works on MeshCat 0.3.x."""
    # 1. files -----------------------------------------------------------
    imgfiles  = sorted(glob(f'{seq_folder}/images/*.jpg'))
    hps_files = sorted(glob(f'{seq_folder}/hps/*.npy'))
    T = len(imgfiles)
    if T == 0:
        raise RuntimeError("No images found – check seq_folder path.")
    N = len(hps_files)                 # number of tracks (people)

    # 2. model -----------------------------------------------------------
    device = 'cuda'
    smpl   = SMPL().to(device)

    # palette (RGB, drop alpha if present)
    pal = np.loadtxt("data/colors.txt")
    if pal.ndim == 1:
        pal = pal[None, :]
    if pal.shape[1] > 3:
        pal = pal[:, :3]
    if len(pal) < N:                   # repeat colours if not enough rows
        pal = np.vstack([pal] * ((N + len(pal) - 1) // len(pal)))

    # 3. camera extrinsics cam→world ------------------------------------
    cam  = np.load(f"{seq_folder}/camera.npy", allow_pickle=True).item()
    R_cw = torch.tensor(cam["world_cam_R"])        # (T,3,3)
    t_cw = torch.tensor(cam["world_cam_T"])        # (T,3)

    # 4. compute joints for every track & frame -------------------------
    per_frame = [[None] * N for _ in range(T)]     # list[T][N] → (24,3) or None
    print("↺  Pre-computing SMPL joints …")
    lowest_y = []

    for tid, npy in enumerate(tqdm(hps_files, desc="tracks")):
        data       = np.load(npy, allow_pickle=True).item()
        frames     = data["frame"]                 # (Ni,)
        Rmat       = data["pred_rotmat"].to(device)
        betas      = data["pred_shape"].to(device)
        trans_raw  = data["pred_trans"].to(device)

        betas = betas.mean(0, keepdim=True).expand_as(betas)
        trans = trans_raw.squeeze()

        out = smpl(body_pose     = Rmat[:, 1:],
                   global_orient = Rmat[:, [0]],
                   betas         = betas,
                   transl        = trans,
                   pose2rot      = False,
                   default_smpl  = True)

        verts_c, joints_c = out.vertices, out.joints[:, :24]

        Rc, tc = R_cw[frames].to(device), t_cw[frames].to(device)
        verts_w  = torch.einsum("bij,bnj->bni", Rc, verts_c)  + tc[:, None]
        joints_w = torch.einsum("bij,bnj->bni", Rc, joints_c) + tc[:, None]
        verts_w, joints_w = traj_filter(verts_w.cpu(), joints_w.cpu())

        lowest_y.append(verts_w[:, :, 1].min())

        for j, f in enumerate(frames.tolist()):
            per_frame[f][tid] = joints_w[j].numpy()           # (24,3)

    # ground height → shift so y==0 is on the floor
    floor_y = float(np.min(lowest_y))
    for f in range(T):
        for tid in range(N):
            if per_frame[f][tid] is not None:
                per_frame[f][tid][:, 1] -= floor_y

    # 5. MeshCat scene setup -------------------------------------------
    vis = meshcat.Visualizer().open()
    vis.delete()

    vis["ground"].set_object(
        g.Box([4, 0.002, 4]),
        g.MeshLambertMaterial(color=0x555555, opacity=0.3, transparent=True)
    )

    sphere_geom = g.Sphere(radius=sphere_radius)

    # create one sphere per track × joint
    for tid in range(N):
        r, g_, b = (pal[tid, :] * 255).astype(int)
        col_hex  = (b << 16) + (g_ << 8) + r
        material = g.MeshLambertMaterial(color=int(col_hex))
        for jid in range(24):
            vis[f"t{tid}_j{jid}"].set_object(sphere_geom, material)

    hidden = tf.translation_matrix([1e6, 1e6, 1e6])  # off-screen

    # 6. streaming loop -------------------------------------------------
    print("▶︎  Streaming frames – Ctrl-C to stop.")
    try:
        while True:
            for f in range(T):
                frame = per_frame[f]
                for tid in range(N):
                    for jid in range(24):
                        node = vis[f"t{tid}_j{jid}"]
                        if frame[tid] is None:
                            node.set_transform(hidden)          # hide actor
                        else:
                            joint = frame[tid][jid]
                            node.set_transform(tf.translation_matrix(joint))
                time.sleep(1.0 / fps)   # real-time pace
    except KeyboardInterrupt:
        print("\n⏹  Stopped.")


# ─── CLI ----------------------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("seq_folder", help="TRAM sequence folder")
    ap.add_argument("--fps", type=int, default=30, help="playback frame-rate")
    args = ap.parse_args()

    if not os.path.isdir(args.seq_folder):
        raise SystemExit("sequence folder does not exist")

    stream_tram_joints(args.seq_folder, fps=args.fps)
