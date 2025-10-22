import os
import glob
import numpy as np
import torch
from tqdm import tqdm
import pickle
from pathlib import Path

from pytorch3d.transforms import matrix_to_axis_angle

import random
random.seed(42)


SELECT_DIR = ["batch8"]

def extract_files(root_folder, select_dir=SELECT_DIR, select_dir2=['images0', 'images1', 'images2']):
    

    # gather all take directories of each subject
    subject_dir_list = []
    for batch_name in sorted(os.listdir(root_folder)):
        if batch_name is not None and batch_name not in select_dir:
            continue
        sub_dir = sorted(os.listdir(os.path.join(root_folder, batch_name)))
        batch_dir = os.path.join(root_folder, batch_name, sub_dir[0])
        for item_name in sorted(os.listdir(batch_dir)):
            if item_name not in select_dir2:
                continue
            subject_dir_list.append(os.path.join(batch_dir, item_name))

    res = []
    # 先统计所有 param 文件路径
    all_param_files = []
    for item_dir in subject_dir_list:
        param_dir = os.path.join(item_dir, "param_smpl")
        for item in sorted(os.listdir(param_dir)):
            all_param_files.append(os.path.join(param_dir, item))

    # all_param_files = all_param_files[:300] # random.sample(all_param_files, 300)

    for param_path in tqdm(all_param_files, desc="Loading all subject data in huge100k"):
 
        identity_name = os.path.basename(param_path).split(".")[0]
        smplx_param_path = os.path.join(os.path.dirname(param_path).replace('param_smpl', 'param'), f'{identity_name}.npy')
        images_path = os.path.join(os.path.dirname(param_path).replace('param_smpl', 'images'), identity_name)

        smpl_param = pickle.load(open(param_path, 'rb'))
        pose = torch.cat([smpl_param['global_orient'], smpl_param['body_pose']], dim=0) # (24,3,3)
        pose = matrix_to_axis_angle(pose).reshape(1, -1) # (1, 72)
        beta = smpl_param['betas'].unsqueeze(0) # (1, 10)
        if smpl_param['transl'] is not None:
            transl = smpl_param['transl'].reshape(1, -1)
        else:
            transl = torch.zeros((1, 3))

        camera_info = np.load(smplx_param_path, allow_pickle=True).item()['poses']

        img_file = sorted(glob.glob(os.path.join(images_path, '*.png')))[0]
        front_view_idx = Path(img_file).stem.split("view_")[-1]
        cam_idx = int(front_view_idx)
        
        intrinsic_params = camera_info[cam_idx][1]  # fx, fy, cx, cy
        extrinsic_params = camera_info[cam_idx][0] # R|T


        R = extrinsic_params[:3, :3].numpy().astype(np.float32)
        T = extrinsic_params[:3, 3].numpy().astype(np.float32)
        K = np.array([
        [intrinsic_params[0], 0, intrinsic_params[2]],
        [0, intrinsic_params[1], intrinsic_params[3]],
        [0,  0,  1]
        ], dtype=np.float32)

        R = torch.from_numpy(R).unsqueeze(0)
        T = torch.from_numpy(T).unsqueeze(0)
        K = torch.from_numpy(K).unsqueeze(0)

        res.append({
            'process_folder': param_path,
            'camera_view': [front_view_idx],
            'camera_params': (K, R, T),
            'path_image': [img_file],
            'pose': pose,
            'beta': beta,
            'transl': transl
        })


    return res


def extract_files_tmp(root_folder, select_dir=SELECT_DIR):
    

    # gather all take directories of each subject
    subject_dir_list = []
    for batch_name in sorted(os.listdir(root_folder)):
        if batch_name is not None and batch_name not in select_dir:
            continue
        sub_dir = sorted(os.listdir(os.path.join(root_folder, batch_name)))
        batch_dir = os.path.join(root_folder, batch_name, sub_dir[0])
        for item_name in sorted(os.listdir(batch_dir)):
            subject_dir_list.append(os.path.join(batch_dir, item_name))

    res = []
    # 先统计所有 param 文件路径
    all_param_files = []
    for item_dir in subject_dir_list:
        param_dir = os.path.join(item_dir, "param_smpl")
        for item in sorted(os.listdir(param_dir)):
            all_param_files.append(os.path.join(param_dir, item))

    for param_path in tqdm(all_param_files, desc="Loading all subject data in huge100k"):
 
        identity_name = os.path.basename(param_path).split(".")[0]

        cloth_path = os.path.join(os.path.dirname(param_path).replace('param_smpl', 'Meshes_cloth'), identity_name)
        if not os.path.exists(cloth_path):

            smplx_param_path = os.path.join(os.path.dirname(param_path).replace('param_smpl', 'param'), f'{identity_name}.npy')
            images_path = os.path.join(os.path.dirname(param_path).replace('param_smpl', 'images'), identity_name)

            smpl_param = pickle.load(open(param_path, 'rb'))
            pose = torch.cat([smpl_param['global_orient'], smpl_param['body_pose']], dim=0) # (24,3,3)
            pose = matrix_to_axis_angle(pose).reshape(1, -1) # (1, 72)
            beta = smpl_param['betas'].unsqueeze(0) # (1, 10)
            if smpl_param['transl'] is not None:
                transl = smpl_param['transl'].reshape(1, -1)
            else:
                transl = torch.zeros((1, 3))

            camera_info = np.load(smplx_param_path, allow_pickle=True).item()['poses']

            img_file = sorted(glob.glob(os.path.join(images_path, '*.png')))[0]
            front_view_idx = Path(img_file).stem.split("view_")[-1]
            cam_idx = int(front_view_idx)
            
            intrinsic_params = camera_info[cam_idx][1]  # fx, fy, cx, cy
            extrinsic_params = camera_info[cam_idx][0] # R|T


            R = extrinsic_params[:3, :3].numpy().astype(np.float32)
            T = extrinsic_params[:3, 3].numpy().astype(np.float32)
            K = np.array([
            [intrinsic_params[0], 0, intrinsic_params[2]],
            [0, intrinsic_params[1], intrinsic_params[3]],
            [0,  0,  1]
            ], dtype=np.float32)

            R = torch.from_numpy(R).unsqueeze(0)
            T = torch.from_numpy(T).unsqueeze(0)
            K = torch.from_numpy(K).unsqueeze(0)

            res.append({
                'process_folder': param_path,
                'camera_view': [front_view_idx],
                'camera_params': (K, R, T),
                'path_image': [img_file],
                'pose': pose,
                'beta': beta,
                'transl': transl
            })


    return res
    
def compute_projections(xyz, K, R, T):
    """
    计算 3D 点在图像平面上的投影，使用 4×4 内外参矩阵合并计算。

    参数：
        xyz: (B, N, 3) - 3D 点，世界坐标系下的点云。
        intrinsics: (B, 3, 3) - 相机内参矩阵。
        train_poses: (B, 4, 4) - 相机外参矩阵 (世界到相机变换 W2C)。
        image_height: int - 图像高度，用于 Y 轴翻转。
        correct_principal: bool - 是否修正主点偏移。

    返回：
        projected_points: (B, N, 2) - 2D 像素坐标。
    """
    B, N, _ = xyz.shape
    xyz = xyz.float()

    intrinsics = K.unsqueeze(0)
    # 构造相机外参矩阵 (4, 4)
    extrinsics = torch.eye(4, dtype=torch.float32).unsqueeze(0)  # (1, 4, 4)
    extrinsics[:, :3, :3] = R.unsqueeze(0) # 旋转部分
    extrinsics[:, :3, 3:] = T.view(1,3,1)  # 平移部分

    # **Step 1: 构造 4×4 内参矩阵**
    K_pad = torch.eye(4, device=intrinsics.device, dtype=intrinsics.dtype).repeat(B, 1, 1)
    K_pad[:, :3, :3] = intrinsics  # 嵌入 3×3 内参


    # **Step 2: 计算投影矩阵 P = K_pad × W2C**
    P = torch.bmm(K_pad, extrinsics)  # (B, 4, 4)

    # **Step 3: 将 3D 点扩展为齐次坐标 (B, N, 4)**
    ones = torch.ones((B, N, 1), device=xyz.device, dtype=xyz.dtype)
    xyz_homogeneous = torch.cat([xyz, ones], dim=2)  # (B, N, 4)

    # **Step 4: 直接应用 P 进行投影变换**
    projected_homogeneous = torch.bmm(xyz_homogeneous, P.transpose(1, 2))  # (B, N, 4)

    # **Step 5: 归一化 (除以 z)**
    eps = 1e-8
    projected_points = projected_homogeneous[:, :, :2] / (projected_homogeneous[:, :, 2:3] + eps)


    return projected_points.squeeze(0)  # (N, 2)

def get_waist_line_from_smpl(smpl_layer, folder):
    """
    joints: (24, 3) SMPL joints in camera coordinates
    cam_intrinsics: (3, 3) 相机内参矩阵
    return: (u, v) 腰部在图像坐标的点
    """
    beta = folder['beta']
    pose = folder['pose']
    transl = folder['transl']
    camera_params = folder['camera_params']
    joints = smpl_layer.skeleton_forward(beta=beta.cuda(), theta=pose.cuda(), transl=transl.cuda(),
                                        get_skin=False)[0].cpu()
    
    # pelvis (0), left_hip (1), right_hip (2)
    pelvis = joints[0]
    left_hip = joints[1]
    right_hip = joints[2]

    waist_3d = (left_hip + right_hip) / 2.0  # 也可以直接用 pelvis

    # 投影到图像
    waist_2d = compute_projections(waist_3d.view(1,1,-1), camera_params[0], camera_params[1], camera_params[2])

    return waist_2d.round().view(-1)


def adjust_segmentation_map(seg1, smpl_layer, folder):
    seg1 = seg1.clone()

    B, H, W = seg1.shape

    for b in range(B):
        s1 = seg1[b]

        waist = get_waist_line_from_smpl(smpl_layer, folder[b])
        waist_y = waist[1]   # y 坐标（图像行数）

        for cls, thresh in [(2,100), (5,50), (6,50), (7,50), (9,50), (12,50), (18,50)]:
            if (s1 == cls).sum() < thresh:
                s1[s1 == cls] = 25

        # 2. dress=6 → 拆成 upper=5 和 skirts=12
        if (s1 == 6).any():
            mask_dress = (s1 == 6)

            yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
            yy = yy.to(mask_dress.device)
            mask_upper = mask_dress & (yy < waist_y)
            mask_skirt = mask_dress & (yy >= waist_y)

            s1[mask_upper] = 5
            s1[mask_skirt] = 12
            s1[s1 == 6] = 25
            
        # 1. outer=7 存在 → upper=5 改成 25
        if (s1 == 7).any() and (s1 == 5).any():
            cnt_outer = (s1 == 7).sum().item()
            cnt_upper = (s1 == 5).sum().item()
            if cnt_outer >= cnt_upper:
                # 保留 outer=7，把 upper=5 改成25
                s1[(s1 == 5)] = 25
            else:
                # 保留 upper=5，把 outer=7 合并为 upper
                s1[s1 == 7] = 5

        
        # 3. pants=9 与 skirts=12 同时出现 → 按面积选择大类
        if (s1 == 9).any() and (s1 == 12).any():
            cnt_pants = (s1 == 9).sum().item()
            cnt_skirt = (s1 == 12).sum().item()

            if cnt_pants >= cnt_skirt:
                # 保留 pants=9，把 skirt=12 合并为 pants
                s1[s1 == 12] = 9
            else:
                # 保留 skirts=12，把 pants=9 合并为 skirts
                s1[s1 == 9] = 12
        
        seg1[b] = s1

    return seg1
