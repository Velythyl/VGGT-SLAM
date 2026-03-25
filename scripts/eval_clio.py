import numpy as np
import struct
import os
import re
import yaml
from termcolor import colored
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

def get_scale_from_yaml(transforms_file, scene_name):
    """Loads scene-specific transform and scale from YAML."""
    with open(transforms_file, "r") as file:
        data = yaml.safe_load(file)
    return float(data[scene_name]['scale'])

def get_id_from_image_name(image_name):
    """Parses frame ID from filename (e.g., 'frame_0012.jpg' -> 12)."""
    name_no_ext = os.path.splitext(image_name)[0]
    digits = re.findall(r'\d+', name_no_ext)
    return int(digits[-1]) if digits else None

def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)

def read_images_binary_with_names(path_to_model_file):
    """Reads COLMAP images.bin. Returns {id: (center_xyz, filename)}."""
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_props = read_next_bytes(fid, 64, "idddddddi")
            qvec = np.array(binary_props[1:5])
            tvec = np.array(binary_props[5:8])
            
            image_name = ""
            char = read_next_bytes(fid, 1, "c")[0]
            while char != b"\x00":
                image_name += char.decode("utf-8")
                char = read_next_bytes(fid, 1, "c")[0]
            
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            fid.seek(24 * num_points2D, 1) # Skip 2D points

            # Convert World-to-Camera to Camera Center (World Coordinates)
            rot = R.from_quat([qvec[1], qvec[2], qvec[3], qvec[0]])
            center = -rot.as_matrix().T @ tvec
            
            extracted_id = get_id_from_image_name(image_name)
            if extracted_id is not None:
                images[extracted_id] = (center, image_name)
    return images

def read_est_trajectory(file_path):
    """Reads est trajectory (frame_id x y z ...)."""
    traj_points = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            fid = int(float(parts[0]))
            traj_points[fid] = np.array([float(x) for x in parts[1:4]])
    return traj_points

def get_transform_from_yaml(transforms_file, scene_name):
    """Loads scene-specific transform and scale from YAML."""
    with open(transforms_file, "r") as file:
        data = yaml.safe_load(file)
    return np.array(data[scene_name]['T']), float(data[scene_name]['scale'])

def compute_sim3(p_a, p_b):
    """
    Computes Sim3 sfrom p_a to p_b
    """
    mu_c = np.mean(p_a, axis=0)
    mu_e = np.mean(p_b, axis=0)
    
    p_c_centered = p_a - mu_c
    p_e_centered = p_b - mu_e
    
    n = p_a.shape[0]
    sigma = (1/n) * (p_e_centered.T @ p_c_centered)
    
    U, D, V_T = np.linalg.svd(sigma)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(V_T) < 0:
        S[2, 2] = -1
        
    R_align = U @ S @ V_T
    var_c = (1/n) * np.trace(p_c_centered.T @ p_c_centered)
    scale = (1/var_c) * np.trace(np.diag(D) @ S)
    t_align = mu_e - scale * (R_align @ mu_c)
    
    return scale, R_align, t_align

def visualize_trajectory_plt(poses_colmap, poses_aligned):
    """Matplotlib 3D plot for trajectory alignment verification."""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.plot(poses_colmap[:, 0], poses_colmap[:, 1], poses_colmap[:, 2], c='black', label='COLMAP ground truth', linewidth=2)
    ax.plot(poses_aligned[:, 0], poses_aligned[:, 1], poses_aligned[:, 2], c='g', label='Esimated Aligned Trajectory')

    ax.set_title('Sim3 Alignment Verification')
    ax.legend()
    plt.show()


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Evaluate estimated bboxes on Clio dataset")
    parser.add_argument('--vis', action='store_true', help='Visualize trajectories and scene')
    parser.add_argument('--scene', type=str, choices=['cubicle', 'apartment', 'office'], required=True, help='Scene to evaluate')
    parser.add_argument('--est_traj', type=str, required=True, help='Path to estimated trajectory text file')
    args = parser.parse_args()

    # --- CONFIGURATION ---

    if args.scene == "cubicle":
        CONFIG = {
            "colmap_bin": "/home/dominic/cam_ws/imgs/cubicle/sparse/0/images.bin",
            "colmap_mesh": "/home/dominic/cam_ws/imgs/cubicle/dense/meshed-poisson.ply",
            "gt_yaml": "/home/dominic/Documents/vggt/temp/tasks_cubicle.yaml",
            "scene_tf_yaml": "/home/dominic/Documents/vggt/scripts/scene_transform.yaml",
            "scene_name": "cubicle"
        }
    elif args.scene == "apartment":
        CONFIG = {
            "colmap_bin": "/home/dominic/cam_ws/imgs/apartment_full/sparse/0/images.bin",
            "colmap_mesh": "/home/dominic/cam_ws/imgs/clio_datasets/apartment/dense/meshed-poisson.ply",
            "gt_yaml": "/home/dominic/Documents/vggt/temp/tasks_apartment.yaml",
            "scene_tf_yaml": "/home/dominic/Documents/vggt/scripts/scene_transform.yaml",
            "scene_name": "apartment"
        }
    elif args.scene == "office":
        CONFIG = {
            "colmap_bin": "/home/dominic/cam_ws/imgs/office/sparse/0/images.bin",
            "colmap_mesh": "/home/dominic/cam_ws/imgs/clio_datasets/office/dense/meshed-poisson.ply",
            "gt_yaml": "/home/dominic/Documents/vggt/temp/tasks_office.yaml",
            "scene_tf_yaml": "/home/dominic/Documents/vggt/scripts/scene_transform.yaml",
            "scene_name": "office"
        }

    # Load Data
    CONFIG["est_traj"] = args.est_traj
    colmap_data = read_images_binary_with_names(CONFIG["colmap_bin"])
    est_data = read_est_trajectory(CONFIG["est_traj"])

    # Match Frames
    common_ids = sorted(list(set(colmap_data.keys()).intersection(set(est_data.keys()))))
    if len(common_ids) < 3:
        raise ValueError(f"Insufficient overlap. Found only {len(common_ids)} common frames.")

    poses_colmap = np.array([colmap_data[fid][0] for fid in common_ids])
    poses_est = np.array([est_data[fid] for fid in common_ids])

    # Compute initial Sim3 from trajectory (poses)
    s_est_colmap, R_est_colmap, t_est_colmap = compute_sim3(poses_est, poses_colmap)

    s_colmap_world = get_scale_from_yaml(CONFIG["scene_tf_yaml"], CONFIG["scene_name"])

    # Calculate Error

    poses_est_aligned = (s_est_colmap * (R_est_colmap @ poses_est.T).T) + t_est_colmap
    error = np.linalg.norm(poses_est_aligned - poses_colmap, axis=1).mean() * s_colmap_world

    print(colored("ATE: " + str(error), 'green'))


    if args.vis:
        visualize_trajectory_plt(poses_colmap, poses_est_aligned)