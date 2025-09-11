# check the mesh for init rotation and scaling
# fps sampling
import torch
import numpy as np
import os
import glob
import trimesh
from pytorch3d.ops import sample_farthest_points
from pytorch3d.transforms import matrix_to_quaternion
from tqdm import tqdm

from dress4d_utils import extract_files


from scipy.spatial import KDTree
def distCUDA2(points):
    points_np = points.detach().cpu().float().numpy()
    dists, inds = KDTree(points_np).query(points_np, k=4)
    meanDists = (dists[:, 1:] ** 2).mean(1)
    return torch.tensor(meanDists, dtype=points.dtype, device=points.device)

# %%

def get_init_attr(verts, faces, sample_ratio, eps=1e-6):
    v0 = verts[faces[:, 0]]  # 第一个顶点
    v1 = verts[faces[:, 1]]  # 第二个顶点
    v2 = verts[faces[:, 2]]  # 第三个顶点
    
    # 计算每个面的法向量
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=1)  # 叉积得到法向量
    face_normals = face_normals / face_normals.norm(dim=1, keepdim=True).clamp(min=eps)  # 归一化法向量
    points = (v0 + v1 + v2) / 3.0  # (M, 3) # center points on each face

    # fps to reduce sampling points
    if sample_ratio < 1.0:
        num_points = max(1, int(points.shape[0] * sample_ratio))
        selected_points, selected_idx = sample_farthest_points(points.unsqueeze(0), K=num_points)
        selected_points = selected_points.squeeze(0)
        selected_idx = selected_idx.squeeze(0)
        
        selected_face_normals = face_normals[selected_idx]
        selected_v0 = v0[selected_idx]
        selected_v1 = v1[selected_idx]

    else:
        selected_points = points
        selected_face_normals = face_normals
        selected_v0 = v0
        selected_v1 = v1

    # Compute tangent & bitangent for each selected point
    tangent = (selected_v1 - selected_v0)
    tangent = tangent / tangent.norm(dim=1, keepdim=True).clamp(min=eps)

    bitangent = torch.cross(selected_face_normals, tangent, dim=1)
    bitangent = bitangent / bitangent.norm(dim=1, keepdim=True).clamp(min=eps)

    # Rotation matrix: (tangent, bitangent, normal)
    rot_mats = torch.stack([tangent, bitangent, selected_face_normals], dim=-1)  # (K, 3, 3)
    init_rotations = matrix_to_quaternion(rot_mats)  # (K, 4)
    bad_rot = (init_rotations.norm(dim=1) < 0.9) | (init_rotations.norm(dim=1) > 1.1)
    init_rotations[bad_rot] = torch.tensor([1, 0, 0, 0], dtype=init_rotations.dtype, device=init_rotations.device)  # identity rotation

    # Scaling
    dist2 = torch.clamp_min(distCUDA2(points), eps)
    init_scalings = torch.sqrt(dist2)[..., None].repeat(1, 3)

    return selected_points, init_rotations, init_scalings

# %%
import plotly.graph_objects as go

def vis_points(xyz, colors='blue'):
    fig = go.Figure(data=[go.Scatter3d(
            x=xyz[:, 0],
            y=xyz[:, 1],
            z=xyz[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=colors,  
                opacity=0.8
            )
        )])

    fig.update_layout(
        title_text=f'Body Part',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    fig.show()

def check_rot_scale(init_rotations, init_scalings):
    flag = False
    # 检查 NaN / Inf
    if torch.isnan(init_rotations).any() or torch.isinf(init_rotations).any():
        print("rotation 里有 NaN 或 Inf")
        flag = True

    if torch.isnan(init_scalings).any() or torch.isinf(init_scalings).any():
        print("scaling 里有 NaN 或 Inf")
        flag = True

    # 检查 rotation 是否是 unit quaternion（范数接近 1）
    rot_norms = init_rotations.norm(dim=1)
    bad_rot = (rot_norms < 0.9) | (rot_norms > 1.1)
    if bad_rot.any():
        print(f"rotation 四元数范数异常: {rot_norms[bad_rot][:10]}")
        flag = True

    # 检查 scaling 是否太大或太小
    scale_min = init_scalings.min().item()
    scale_max = init_scalings.max().item()
    if scale_min < 1e-5 or scale_max > 1e2:
        print(f"scaling 范围异常: min={scale_min}, max={scale_max}")
        flag = True

    return flag

# %%
device = 'cuda'
folders = extract_files('.datasets/4ddress')
check_res = []
for folder in tqdm(folders):
    cloth_path = os.path.join(folder['process_folder'], 'Meshes_cloth')
    mesh_path = glob.glob(f"{cloth_path}/unpose*.obj")[0]
    cloth_mesh = trimesh.load(mesh_path)
    cloth_faces = torch.from_numpy(cloth_mesh.faces).long()
    cloth_points = torch.from_numpy(cloth_mesh.vertices).float()

    select_cloth_points, cloth_rotation, cloth_scaling = get_init_attr(cloth_points.to(device), cloth_faces.to(device), sample_ratio=0.1) # 146984 face num
    # vis_points(select_cloth_points)
    flag = check_rot_scale(cloth_rotation, cloth_scaling)
    
    if flag:
        check_res.append(folder['process_folder'])

print("有问题的文件夹:")
for f in check_res:
    print(f)