import numpy as np
import torch, time, os, argparse
from glob import glob
from tqdm import tqdm

import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

from lib.vis.traj import traj_filter
from lib.models.smpl import SMPL

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

def bone_transform(a, b, eps=1e-8):
    """
    Return 4x4 transform that maps a unit cylinder aligned with +Y (height=1, centered at origin)
    to a cylinder connecting points a -> b.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    v = b - a
    L = np.linalg.norm(v)
    if L < eps:
        # Degenerate: hide at origin
        return tf.translation_matrix([1e6, 1e6, 1e6])

    mid = 0.5 * (a + b)
    y = v / L  # desired +Y direction

    # Build rotation matrix that takes +Y to direction y
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    axis = np.cross(up, y)
    s = np.linalg.norm(axis)
    c = float(np.dot(up, y))
    if s < 1e-8:
        # up and y are parallel (or anti-parallel)
        if c > 0:
            R = np.eye(3)
        else:
            # 180 deg around X
            R = tf.rotation_matrix(np.pi, [1, 0, 0])[:3, :3]
    else:
        axis /= s
        R = tf.rotation_matrix(np.arctan2(s, c), axis)[:3, :3]

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = mid

    # Scale: cylinder height is along +Y. MeshCat's Cylinder is centered, height=1.
    S = np.eye(4)
    S[1, 1] = L  # scale along Y to match bone length

    return T @ S

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

    # ------ frame conversion:  (x, y, z)_SMPL  →  (z, x, y)_MeshCat
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

    sphere_geom = g.Sphere(radius=sphere_radius)

    hidden = tf.translation_matrix([1e6, 1e6, 1e6])  # off-screen

    # ---- CHANGED: bones are cylinders, transforms only (no set_object per frame)
    bone_nodes = {}        # (tid, bid) -> node
    bone_radius = sphere_radius * 0.35
    unit_cyl = g.Cylinder(1.0, bone_radius)  # height=1, aligned with +Y, centered at origin
    # bones: deep blue to complement the warm joint colors
    bone_mat_blue = g.MeshLambertMaterial(color=0x1a4dff)

    for tid in range(N):
        r, g_, b = (pal[tid, :] * 255).astype(int)
        col_hex  = int((b << 16) + (g_ << 8) + r)
        joint_mat = g.MeshLambertMaterial(color=col_hex)
        bone_mat  = bone_mat_blue

        # joints
        for jid in range(24):
            world[f"t{tid}_j{jid}"].set_object(sphere_geom, joint_mat)

        # bones
        for bid, (p, c) in enumerate(BONES):
            node = world[f"t{tid}_b{bid}"]
            bone_nodes[(tid, bid)] = node
            node.set_object(unit_cyl, bone_mat)

    # 6. streaming loop -------------------------------------------------
    period = 1.0 / fps
    next_t = time.perf_counter()
    print("▶︎  Streaming frames – Ctrl-C to stop.")
    try:
        while True:
            for f in range(T):
                frame = per_frame[f]

                for tid in range(N):
                    if frame[tid] is None:
                        # hide joints + bones
                        for jid in range(24):
                            world[f"t{tid}_j{jid}"].set_transform(hidden)
                        for bid in range(len(BONES)):
                            bone_nodes[(tid, bid)].set_transform(hidden)
                        continue

                    # joints
                    for jid in range(24):
                        joint = frame[tid][jid]
                        world[f"t{tid}_j{jid}"].set_transform(tf.translation_matrix(joint))

                    # bones (transform-only)
                    for bid, (p, c) in enumerate(BONES):
                        a = frame[tid][p]
                        b = frame[tid][c]
                        bone_nodes[(tid, bid)].set_transform(bone_transform(a, b))

                next_t += period
                sleep_time = next_t - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)

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
