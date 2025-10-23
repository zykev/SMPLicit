import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
import cv2
import pickle
import torch
import glob
import trimesh
from fit_SMPLicit.utils.sdf import create_grid
from fit_SMPLicit.utils import projection
from fit_SMPLicit.options.image_fitting_options import FitOptions
# import kaolin
import time
import tqdm
import shutil

from fit_SMPLicit.utils import image_fitting

from huge100k_utils import extract_files
from dress4d_utils import get_depth_map, compute_projections, get_mesh_render, combine_meshes, get_02v_pose, point_to_mesh_distance
from huge100k_utils import adjust_segmentation_map

import sys
import os

# 获取当前脚本所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将 submodule 的路径加入 Python 搜索路径
HP_PATH = os.path.join(ROOT_DIR, "submodules", "human_parsing")
sys.path.insert(0, HP_PATH)  # 插入到开头，优先搜索

from submodules.human_parsing.evaluate_simple import get_segmentation_map
# from submodules.human_parsing.sapiens_seg import get_segmentation_sapiens



import SMPLicit
import os
# cuda = 2
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

fitoptions = FitOptions()
_opt = fitoptions.parse()


SMPLicit_Layer = SMPLicit.SMPLicit()
SMPLicit_Layer = SMPLicit_Layer.cuda()

# Initialize SMPL-Related stuff:
SMPL_Layer = SMPLicit_Layer.SMPL_Layer
smpl_faces = torch.LongTensor(SMPL_Layer.faces).cuda()
v_pose = get_02v_pose()


cool_latent_reps = np.load('fit_SMPLicit/utils/z_gaussians.npy')

print("PROCESSING:")
# print(files)

folders = extract_files('.datasets/HuGe100K')
# folders = [folders[293]]

segmentation_maps = get_segmentation_map(folders, image_size=[896, 640])
# segmentation_sapiens = get_segmentation_sapiens(folders, image_size=[896, 640])

assert len(folders) == len(segmentation_maps)

# segmentation_maps = adjust_segmentation_map(segmentation_maps, segmentation_sapiens, minor_adj=False)

segmentation_maps = adjust_segmentation_map(segmentation_maps, SMPL_Layer, folders)

for idx, folder in enumerate(folders):
    print('Processing folder:', folder['process_folder'])
    # set save folder
    identity_name = os.path.basename(folder['process_folder']).split(".")[0]
    save_folder = os.path.join(os.path.dirname(folder['process_folder']).replace('param_smpl', 'Meshes_cloth'), identity_name)
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder, exist_ok=True)  
    # print(save_folder)

    segmentation = segmentation_maps[idx]
    segmentation = segmentation.cpu().numpy() # (h, w) 0,1,2,

    # 更新配置中的分割图信息
    _opt = fitoptions.set_segmentation(segmentation)

    if len(_opt.labels) == 0 or (5 in _opt.labels and 7 in _opt.labels) or (9 in _opt.labels and 12 in _opt.labels):
        if len(_opt.labels) == 0:
            reason = "Labels empty"
        if 5 in _opt.labels and 7 in _opt.labels:
            reason = "Both uppercloth and coat present"
        if 9 in _opt.labels and 12 in _opt.labels:
            reason = "Both pants and skirt present"
        print(f"Exclude reason {reason} for folder: {folder['process_folder']}")
        # 以追加方式写入 txt
        with open('tmp/problem_folder.txt', 'a') as f:
            f.write(folder['process_folder'] + ',' + reason + '\n')
        continue

    
    pose = folder['pose'] # (1, 72)
    beta = folder['beta'] # (1, 10)
    transl = folder['transl'] # (1, 3)
    # --- 2. 主循环：遍历每个foloder的每张图片进行处理 ---

    for path_image, select_view in zip(folder['path_image'], folder['camera_view']):
        identity_id = os.path.basename(path_image).replace('.png', '')

        # 读取图片和SMPL
        input_image = cv2.imread(path_image)

        posed_meshes = []
        unposed_meshes = []
        cloth_latents = {}
        # posed_normals = []
        # colors = []

        threshold = 245
        mask = ~np.all(input_image >= threshold, axis=-1) # 所有通道都大于等于阈值 -> False, 其他 -> True
        instance_segmentation = mask.astype(np.uint8)       # (H, W)


        # 生成基础 SMPL 身体和深度图
        SMPL_Layer = SMPL_Layer.cuda()
        v_posed = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), transl=transl.cuda(),
                                        get_skin=True)[0][0].cpu().data.numpy()

        mesh_smpl = trimesh.Trimesh(v_posed, smpl_faces.cpu().data.numpy())

        K, R, T = folder['camera_params']
        image_size = np.array(segmentation.shape)

        depth_image_smpl = get_depth_map(v_posed, smpl_faces, K, R, T, image_size=image_size)
        depth_image_smpl = depth_image_smpl.cpu().data.numpy()
        

        try:
            # Optimizating clothes one at a time
            # --- 4. 衣物优化循环：逐件拟合衣物 ---
            for cloth_optimization_index in _opt.labels:
                _opt = fitoptions.update_optimized_cloth(cloth_optimization_index)

                # 获取 T-pose 下的 SMPL 身体顶点和骨骼关节点
                J, v = SMPL_Layer.skeleton(beta.cuda(), require_body=True)
                SMPL_Layer = SMPL_Layer.cuda()
                # 根据配置，选择是在标准 T-pose 还是在微调过的 repose 姿态下进行推理
                if _opt.repose:
                    v_inference = SMPL_Layer.forward(beta=beta.cuda(), theta=_opt.pose_inference_repose.cuda(), get_skin=True)[0]
                else:
                    v_inference = v
                

                # Sample points uniformly in predefined 3D space of clothing:
                # 在一个预定义的 3D 边界框内创建均匀的采样点
                coords, mat = create_grid(_opt.resolution, _opt.resolution, _opt.resolution, np.array(_opt.b_min), np.array(_opt.b_max))
                coords = coords.reshape(3, -1).T

                coords_tensor = torch.FloatTensor(coords)

                # Remove unnecessary points that are too far from body and are never occupied anyway:
                # 过滤掉离身体表面太远的采样点，以提高效率
                unsigned_distance = point_to_mesh_distance(v_inference[0].cuda(), smpl_faces.cuda(), coords_tensor.cuda())

                if cloth_optimization_index == 2:
                    valid = unsigned_distance < 0.1
                else:
                    #valid = unsigned_distance < 0.001
                    valid = unsigned_distance < 0.01
                coords = coords[valid.cpu().data.numpy()]
                coords_tensor = coords_tensor[valid.cpu()]

                # Re-Pose to SMPL's Image pose:
                # --- 4.2. 将采样点从 T-pose 变换到目标姿态 (蒙皮/Skinning) ---
                if cloth_optimization_index in [6, 9, 12]:
                    # lower body
                    unposed_verts = coords_tensor
                    model_trimesh = trimesh.Trimesh(unposed_verts, [], process=False)
                    model_trimesh = image_fitting.unpose_and_deform_cloth(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer, transl=transl)
                    posed_coords = model_trimesh.vertices
                else:
                    # TODO: Move this to image utils script
                    SMPL_Layer = SMPL_Layer.cpu()
                    posed_coords = np.zeros((len(coords), 3))
                    for i in range(len(coords)//_opt.step + 1):
                        unposed_verts = coords_tensor[i*_opt.step:(i+1)*_opt.step]
                        _, batch_unposed_coords = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J.cpu(), v.cpu(), unposed_verts.unsqueeze(0), neighbors=10, transl=transl)
                        posed_coords[i*_opt.step:(i+1)*_opt.step] = batch_unposed_coords[0].cpu().data.numpy()
                    SMPL_Layer = SMPL_Layer.cuda()

                # Convert 3D Points to original image space (X,Y are aligned to image)
                # --- 4.3. 将 3D 点投影到 2D 图像空间 ---
                coords_2d = compute_projections(torch.from_numpy(posed_coords).unsqueeze(0), K, R, T)

                # Remove vertices outside of the image
                # 移除投影到图像外部的点
                coords_2d, valid = image_fitting.remove_outside_vertices(coords_2d, input_image)
                coords = coords[valid]
                coords_tensor = coords_tensor[valid]
                posed_coords = posed_coords[valid]

                # Move to image coordinates
                coords_2d = np.int32(coords_2d)[:, :2]


                # TODO: MOVE THIS TO IMAGE UTILS SCRIPT:
                # Find array of indices per pixel:
                # --- 4.4. 构建优化所需的约束 (正/负样本) ---
                # 这是一个关键步骤，为每个像素构建其对应的三维点集和真值标签
                array_pixels = []
                array_gt = []
                condys = []
                for y in range(input_image.shape[1]):
                    cond2 = coords_2d[:, 0] == y
                    condys.append(cond2)
                # 遍历图像的每个像素
                for x in range(input_image.shape[0]):
                    cond1 = coords_2d[:, 1] == x
                    # Faster iteration:
                    if not cond1.max(): 
                        continue
                    for y in range(input_image.shape[1]):
                        cond2 = condys[y]
                        if not cond2.max():
                            continue
                        if instance_segmentation[x, y] == 0 and cloth_optimization_index != 2:
                            continue
                        indices = np.where(np.logical_and(cond1, cond2))[0]
                        if len(indices) == 0:
                            continue

                        depth_smpl = depth_image_smpl[x, y]
                        if depth_smpl == 0:
                            depth_smpl = np.inf

                        indices = indices[coords[indices, 2] < depth_smpl]
                        if len(indices) == 0:
                            continue
                        if segmentation[x, y] in _opt.other_labels:
                            continue

                        for i in indices: # 一个像素点可能对应好几个3d坐标
                            array_pixels.append(i)
                            array_gt.append(_opt.clamp_value - (segmentation[x, y] == cloth_optimization_index)*_opt.clamp_value)

                array_pixels = np.array(array_pixels)
                array_gt = np.array(array_gt)

                if len(array_pixels) < 200:
                    continue

                # NOTE: Initialize upper cloth to open jacket's parameters helps convergence when we have detected jacket's segmentation label
                # --- 4.5. 初始化并执行优化循环 ---
                # 为当前衣物初始化可优化的参数（形状和风格的潜向量）
                if cloth_optimization_index == 7: #7, Coat
                    parameters = image_fitting.OptimizationCloth(_opt.num_params_style, _opt.num_params_shape, cool_latent_reps[3][6:])
                else:
                    parameters = image_fitting.OptimizationCloth(_opt.num_params_style, _opt.num_params_shape)
                optimizer = torch.optim.Adam(parameters.parameters(), lr=_opt.lr)


                # Optimization loop:
                # TODO: Add weights to options file:
                weight_occ = 1
                weight_reg = 6
                frames = []
                decay = (_opt.lr - _opt.lr_decayed)/_opt.iterations

                # 加载聚类文件，用于位置编码
                clusters = np.load(SMPLicit_Layer._opt.path_cluster_files + '/' + _opt.clusters, allow_pickle=True)
                for i in tqdm.tqdm(range(_opt.iterations), desc=f"cloth {_opt.labels_name[int(cloth_optimization_index)]}"):
                    # Select random number of vertices:
                    indices = np.arange(len(array_pixels))
                    np.random.shuffle(indices)

                    # Get differentiable style and shape vectors:
                    style, shape = parameters.forward()

                    # Iterate over these samples
                    loss = torch.FloatTensor([0]).cuda()
                    loss_positives = torch.FloatTensor([0]).cuda()
                    loss_negatives = torch.FloatTensor([0]).cuda()
                    for index_sample in range(_opt.index_samples):
                        inds_points = array_pixels[indices[index_sample]]
                        gt = array_gt[indices[index_sample]]
                        points = torch.FloatTensor(coords_tensor[inds_points]).cuda()

                        # Positional encoding
                        # 计算位置编码：输入点相对于 SMPL 聚类中心的位置差
                        input_position_points = points[None, None, :] - v_inference[:, clusters[_opt.num_clusters]].cuda()
                        input_position_points = input_position_points.reshape(1, -1, _opt.num_clusters*3)

                        # Forward pass:
                        # 前向传播：将位置编码和潜向量输入SMPLicit模型，预测SDF值
                        if cloth_optimization_index == 18:   #18: # Shoe
                            empty_tensor = torch.zeros(1,0).cuda()
                            pred_dists = SMPLicit_Layer.forward(_opt.index_cloth, empty_tensor, style, input_position_points)
                        else:
                            pred_dists = SMPLicit_Layer.forward(_opt.index_cloth, shape, style, input_position_points)

                        # Loss:
                        if gt == _opt.clamp_value:
                            loss = loss + torch.abs(pred_dists - _opt.clamp_value).max()*_opt.weight_negatives
                            loss_negatives = loss_negatives + torch.abs(pred_dists - _opt.clamp_value).max()*_opt.weight_negatives
                        else:
                            lp = torch.min(pred_dists).mean()*_opt.weight_positives

                            loss_positives = loss_positives + lp
                            loss = loss + lp

                    # Hyperparameters for fitting T-Shirt and Jacket better. This might require a little bit of tweaking, and challenging images might converge wrongly
                    # 添加正则化损失，防止潜向量过大导致形状怪异
                    if cloth_optimization_index == 5: # uppercloth
                        reg = torch.abs(style).mean()*10 + torch.abs(shape).mean()
                    elif cloth_optimization_index == 7: # coat
                        center_z_style = torch.FloatTensor(cool_latent_reps[3][6:]).cuda()
                        center_z_cut = torch.FloatTensor(cool_latent_reps[3][:6]).cuda()
                        reg = (torch.abs(style - center_z_style).mean() + torch.abs(shape - center_z_cut).mean())*4
                    else:
                        reg = torch.abs(style).mean() + torch.abs(shape).mean()

                    loss = loss*weight_occ + reg*weight_reg
                    # Backward:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    for param_group in optimizer.param_groups:
                        param_group['lr'] = _opt.lr - decay*i

                t = time.time()
                # print(f"Optimization loss: {loss.item():.4f}")
                # After training, get final prediction:
                # --- 4.6. 从优化后的隐式场重建显式网格 ---
                style, shape = parameters.forward()

                if cloth_optimization_index == 18: # Shoe
                    smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [style[0]], _opt.pose_inference_repose.cuda(), beta.cuda())
                    cloth_latents.update({_opt.labels_name[int(cloth_optimization_index)]: style[0].cpu().data.numpy()})
                else:
                    smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [torch.cat((shape, style), 1).cpu().data.numpy()[0]], _opt.pose_inference_repose.cuda(), beta.cuda())
                    cloth_latents.update({_opt.labels_name[int(cloth_optimization_index)]: torch.cat((shape, style), 1)[0].cpu().data.numpy()})

                # smooth_normals = model_trimesh.vertex_normals.copy()
                # normals = model_trimesh.vertex_normals.copy()

                # color meshes:
                color = np.array(_opt.color[0], dtype=np.uint8)
                model_trimesh.visual.vertex_colors = np.tile(color, (model_trimesh.vertices.shape[0], 1))

                # Unpose+Pose if it's lower body, and pose directly if it's upper body:
                # 将 T-pose 下的衣物网格，通过蒙皮算法穿到目标姿态上
                if cloth_optimization_index in [9, 12]:
                    posed_trimesh = image_fitting.unpose_and_deform_cloth_w_normals(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer, transl=transl, return_unpose=True)
                else:
                    posed_trimesh = image_fitting.batch_posing_w_normals(model_trimesh, pose, J, v, SMPL_Layer, transl=transl)
                
                # adjust model_trimesh from smpl T pose to virtuvian pose
                model_trimesh = image_fitting.batch_posing_w_normals(model_trimesh, v_pose, J, v, SMPL_Layer)

                # Smooth meshes:
                # 对最终的网格进行平滑处理，使其更自然
                posed_trimesh = trimesh.smoothing.filter_laplacian(posed_trimesh,lamb=0.5)
                model_trimesh = trimesh.smoothing.filter_laplacian(model_trimesh,lamb=0.5)

                # Save predictions before rendering all of them together
                unposed_meshes.append(model_trimesh)
                posed_meshes.append(posed_trimesh)
                # posed_normals.append(normals)
                # colors.append(_opt.color[0][:3])

            # --- save unposed and posed meshes ---
            combined_unposed = combine_meshes(unposed_meshes)
            combined_posed = combine_meshes(posed_meshes)

            combined_unposed.export(os.path.join(save_folder, 'unposed-' + identity_id + '-' + select_view + '.obj'))
            combined_posed.export(os.path.join(save_folder, 'posed-' + identity_id + '-' + select_view + '.obj'))
            np.savez(os.path.join(save_folder, 'latents-' + identity_id + '-' + select_view + '.npz'), **cloth_latents)

            # render posed meshes to images
            posed_image = get_mesh_render(combined_posed, K, R, T, image_size=image_size)
            # unposed_image = get_multi_mesh_render(unposed_meshes, K, R, T, image_size=image_size)
            plt.imsave(os.path.join(save_folder, 'render-' + identity_id + '-' + select_view + '.png'), posed_image)

        except:
            print(f"Exception for folder: {folder['process_folder']}")
            # 以追加方式写入 txt
            with open('tmp/problem_folder.txt', 'a') as f:
                f.write(folder['process_folder'] + ',' + 'Exception\n')
            continue
