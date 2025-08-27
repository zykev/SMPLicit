import numpy as np
import matplotlib.pyplot as plt
# plt.ion()
import cv2
import pickle
import torch
import glob
import trimesh
from utils.sdf import create_grid, eval_grid_octree, eval_grid
from utils import projection
from options.image_fitting_options import FitOptions
import kaolin
import time
import tqdm
from utils import image_fitting

import sys
import os

# 获取当前文件夹的上级目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 将平行目录加入 Python 搜索路径
sys.path.append(os.path.join(project_root, "SMPLicit"))

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

files = glob.glob(_opt.image_folder + '/*' + _opt.image_extension)
files.sort()

cool_latent_reps = np.load('fit_SMPLicit/utils/z_gaussians.npy')

print("PROCESSING:")
print(files)

def compute_sdf_from_mesh(mesh: kaolin.rep.SurfaceMesh, points: torch.Tensor) -> torch.Tensor:
    """
    根据 GitHub issue 中的官方推荐方法，实现从 Kaolin SurfaceMesh 计算 SDF 的函数。

    该函数通过组合 `point_to_mesh_distance` 和 `check_sign` 来计算符号距离。

    参数:
        mesh (kaolin.rep.SurfaceMesh): 输入的网格对象。
                                    为了保证符号计算准确，该网格必须是水密的 (watertight)。
        points (torch.Tensor):     需要计算 SDF 的点云，形状为 (N, 3)。

    返回:
        torch.Tensor: 一个形状为 (N,) 的张量，包含每个点的 SDF 值。
                      负值代表点在网格内部，正值代表在外部。
    """
    # 确保网格的顶点和面片与点云在同一个设备上 (例如 'cuda')
    # Kaolin 的函数通常需要一个批处理维度，所以我们使用 unsqueeze(0)
    vertices_batch = mesh.vertices.to(points.device).unsqueeze(0)
    faces = mesh.faces.to(points.device)
    points_batch = points.to(vertices_batch.device).unsqueeze(0)

    # 1. 提取出每个三角面片的顶点坐标
    # 这个是 point_to_mesh_distance 和 check_sign 都需要的输入
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices_batch, faces)

    # 2. 计算点到网格表面的无符号距离（的平方）
    # 遵循图中建议，这是第一步
    distance_sq, face_indices, _ = kaolin.metrics.mesh.point_to_mesh_distance(
        points_batch, face_vertices
    )
    
    # 3. 对距离的平方进行开方，得到真实的距离
    # 遵循图中 "be careful to apply torch.sqrt" 的建议
    distance = torch.sqrt(distance_sq)

    # 4. 判断点的符号（内外）
    # 遵循图中建议，这是第二步
    # 注意：check_sign 要求网格是水密的 (watertight)
    is_inside = kaolin.ops.mesh.check_sign(points_batch, face_vertices)
    
    # 5. 结合距离和符号
    # 将布尔值的 is_inside (True for inside) 转换为数值符号 (-1 for inside, 1 for outside)
    sign = torch.where(is_inside, -1.0, 1.0).to(distance.device)
    
    signed_distance = sign * distance
    
    # 返回结果时去掉批处理维度，使其与输入点的维度匹配
    return signed_distance.squeeze(0)


# --- 2. 主循环：遍历每张图片进行处理 ---
for _file in files:
    _file = str.split(_file[:-4], '/')[-1]
    path_image = _opt.image_folder + _file + _opt.image_extension
    path_smpl_prediction = _opt.smpl_prediction_folder + _file + '_prediction_result.pkl'
    path_segmentation = _opt.cloth_segmentation_folder + _file + '.png'
    path_instance_segmentation =  _opt.instance_segmentation_folder + _file + '.png'

    # 读取图片和SMPL
    input_image = cv2.imread(path_image)
    smpl_prediction = pickle.load(open(path_smpl_prediction, 'rb'))['pred_output_list']

    posed_meshes = []
    posed_normals = []
    colors = []
    all_camscales = []
    all_camtrans = []
    all_toplefts = []
    all_scaleratios = []
    done_people = []
    #for index_fitting in range(1):
    
    # --- 3. 内层循环：遍历图片中的每一个人 ---
    for index_fitting in range(len(smpl_prediction)):
        segmentation = cv2.imread(path_segmentation, 0)
        instance_segmentation = cv2.imread(path_instance_segmentation, 0)
        # 获取当前人的边界框信息
        topleft = smpl_prediction[index_fitting]['bbox_top_left']
        scale_ratio = smpl_prediction[index_fitting]['bbox_scale_ratio']

        offset = 40
        min_y = max(0,int(topleft[1]+offset/scale_ratio))
        max_y = min(input_image.shape[0],int(topleft[1]+(224-offset)/scale_ratio))
        min_x = max(0,int(topleft[0]+offset/scale_ratio))
        max_x = min(input_image.shape[1],int(topleft[0]+(224-offset)/scale_ratio))
        person_instance_segm = instance_segmentation[min_y:max_y,min_x:max_x]
        try:
            index_instancesegm = np.bincount(person_instance_segm.reshape(-1))[1:].argmax() + 1
        except:
            continue

        if index_instancesegm in done_people:
            continue
        done_people.append(index_instancesegm)

        # 在衣物分割图中，将不属于当前实例的像素全部置为0（背景）
        assert index_instancesegm > 0
        segmentation[instance_segmentation!=index_instancesegm] = 0

        # 清理分割图中的小噪声块
        if (segmentation == 2).sum() < 100:
            segmentation[segmentation == 2] = 25
        if (segmentation == 5).sum() < 50:
            segmentation[segmentation == 5] = 25
        if (segmentation == 7).sum() < 50:
            segmentation[segmentation == 7] = 25
        if (segmentation == 9).sum() < 50:
            segmentation[segmentation == 9] = 25
        if (segmentation == 12).sum() < 50:
            segmentation[segmentation == 12] = 25
        if (segmentation == 18).sum() < 50:
            segmentation[segmentation == 18] = 25

        # Image crop params:
        camScale = smpl_prediction[index_fitting]['pred_camera'][0] 
        camTrans = smpl_prediction[index_fitting]['pred_camera'][1:]
        topleft = smpl_prediction[index_fitting]['bbox_top_left']
        scale_ratio = smpl_prediction[index_fitting]['bbox_scale_ratio']
        pose = torch.FloatTensor(smpl_prediction[index_fitting]['pred_body_pose'])
        beta = torch.FloatTensor(smpl_prediction[index_fitting]['pred_betas'])

        # Visualization:
        #m = trimesh.Trimesh(smpl_prediction[index_fitting]['pred_vertices_smpl'], smpl_prediction[index_fitting]['faces'])
        #m.show()

        # 更新配置中的分割图信息
        _opt = fitoptions.set_segmentation(segmentation)

        # 生成基础 SMPL 身体和深度图
        SMPL_Layer = SMPL_Layer.cuda()
        v_posed = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), 
                                     get_skin=True)[0][0].cpu().data.numpy()
        mesh_smpl = trimesh.Trimesh(v_posed, smpl_faces.cpu().data.numpy())

        depth_image_smpl = image_fitting.render_depth_image(mesh_smpl, camScale, camTrans, topleft, scale_ratio, input_image)

        # Prepare final mesh, and we will keep concatenating vertices/faces later on, while updating normals and colors:
        vertices_smpl = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), get_skin=True)[0][0].cpu().data.numpy()
        m = trimesh.Trimesh(vertices_smpl, smpl_faces.cpu().data.numpy())
        posed_meshes.append(m)
        posed_normals.append(m.vertex_normals)
        colors.append([220, 220, 220]) # 身体颜色设为灰色
        all_camscales.append(camScale)
        all_camtrans.append(camTrans)
        all_toplefts.append(topleft)
        all_scaleratios.append(scale_ratio)

        # Optimizating clothes one at a time
        # --- 4. 衣物优化循环：逐件拟合衣物 ---
        for cloth_optimization_index in _opt.labels:
            _opt = fitoptions.update_optimized_cloth(cloth_optimization_index)

            # Get SMPL mesh in kaolin:
            #SMPL_Layer = SMPL_Layer.cpu()
            # 获取 T-pose 下的 SMPL 身体顶点和骨骼关节点
            J, v = SMPL_Layer.skeleton(beta.cuda(), require_body=True)
            SMPL_Layer = SMPL_Layer.cuda()
            # 根据配置，选择是在标准 T-pose 还是在微调过的 repose 姿态下进行推理
            if _opt.repose:
                v_inference = SMPL_Layer.forward(beta=beta.cuda(), theta=_opt.pose_inference_repose.cuda(), get_skin=True)[0]
            else:
                v_inference = v
            # smpl_mesh = kaolin.rep.TriangleMesh.from_tensors(v_inference[0].cuda(), smpl_faces.cuda())
            # 将 SMPL 网格转换为 Kaolin 格式，以计算 SDF
            smpl_mesh = kaolin.rep.SurfaceMesh(vertices = [v_inference[0].cuda()], faces=[smpl_faces.cuda()])
            
            #! smpl_mesh_sdf = kaolin.conversions.trianglemesh_to_sdf(smpl_mesh)
            smpl_mesh_sdf = compute_sdf_from_mesh(smpl_mesh)

            # Sample points uniformly in predefined 3D space of clothing:
            # 在一个预定义的 3D 边界框内创建均匀的采样点
            coords, mat = create_grid(_opt.resolution, _opt.resolution, _opt.resolution, np.array(_opt.b_min), np.array(_opt.b_max))
            coords = coords.reshape(3, -1).T

            coords_tensor = torch.FloatTensor(coords)

            # Remove unnecessary points that are too far from body and are never occupied anyway:
            # 过滤掉离身体表面太远的采样点，以提高效率
            unsigned_distance = torch.abs(smpl_mesh_sdf(coords_tensor.cuda()))

            if cloth_optimization_index == 2:
                valid = unsigned_distance < 0.1
            else:
                #valid = unsigned_distance < 0.001
                valid = unsigned_distance < 0.01
            coords = coords[valid.cpu().data.numpy()]
            coords_tensor = coords_tensor[valid.cpu()]

            # Re-Pose to SMPL's Image pose:
            # --- 4.2. 将采样点从 T-pose 变换到目标姿态 (蒙皮/Skinning) ---
            if cloth_optimization_index == 9 or cloth_optimization_index == 12:
                unposed_verts = coords_tensor
                model_trimesh = trimesh.Trimesh(unposed_verts, [], process=False)
                model_trimesh = image_fitting.unpose_and_deform_cloth(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer)
                posed_coords = model_trimesh.vertices
            else:
                # TODO: Move this to image utils script
                SMPL_Layer = SMPL_Layer.cpu()
                posed_coords = np.zeros((len(coords), 3))
                for i in range(len(coords)//_opt.step + 1):
                    unposed_verts = coords_tensor[i*_opt.step:(i+1)*_opt.step]
                    _, batch_unposed_coords = SMPL_Layer.deform_clothed_smpl_usingseveralpoints(pose, J.cpu(), v.cpu(), unposed_verts.unsqueeze(0), neighbors=10)
                    posed_coords[i*_opt.step:(i+1)*_opt.step] = batch_unposed_coords[0].cpu().data.numpy()
                SMPL_Layer = SMPL_Layer.cuda()

            # Convert 3D Points to original image space (X,Y are aligned to image)
            # --- 4.3. 将 3D 点投影到 2D 图像空间 ---
            coords_2d = image_fitting.project_points(posed_coords, camScale, camTrans, topleft, scale_ratio, input_image)

            # Remove vertices outside of the image
            # 移除投影到图像外部的点
            coords_2d, valid = image_fitting.remove_outside_vertices(coords_2d, input_image)
            coords = coords[valid]
            coords_tensor = coords_tensor[valid]
            posed_coords = posed_coords[valid]

            # Move to image coordinates
            coords_2d = np.int32(coords_2d)[:, :2]

            # Find positive/negative gt:
            dists = np.zeros(len(coords_2d))
            dists[segmentation[coords_2d[:, 1], coords_2d[:, 0]] == 1] = 0

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

                    array_pixels.append(indices)
                    array_gt.append(_opt.clamp_value - (segmentation[x, y] == cloth_optimization_index)*_opt.clamp_value)

            array_pixels = np.array(array_pixels)
            array_gt = np.array(array_gt)

            if len(array_pixels) < 200:
                continue

            # NOTE: Initialize upper cloth to open jacket's parameters helps convergence when we have detected jacket's segmentation label
            # --- 4.5. 初始化并执行优化循环 ---
            # 为当前衣物初始化可优化的参数（形状和风格的潜向量）
            if cloth_optimization_index == 7:
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
            for i in tqdm.tqdm(range(_opt.iterations)):
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
                    input_position_points = points[:, None] - v_inference[:, clusters[_opt.num_clusters]].cuda()
                    input_position_points = input_position_points.reshape(1, -1, _opt.num_clusters*3)

                    # Forward pass:
                    # 前向传播：将位置编码和潜向量输入SMPLicit模型，预测SDF值
                    if cloth_optimization_index == 18: # Shoe
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
                if cloth_optimization_index == 5:
                    reg = torch.abs(style).mean()*10 + torch.abs(shape).mean()
                elif cloth_optimization_index == 7:
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

            # After training, get final prediction:
            # --- 4.6. 从优化后的隐式场重建显式网格 ---
            style, shape = parameters.forward()

            if cloth_optimization_index == 18: # Shoe
                smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [style[0]], _opt.pose_inference_repose.cuda(), beta.cuda())
            else:
                smpl_mesh, model_trimesh = SMPLicit_Layer.reconstruct([_opt.index_cloth], [torch.cat((shape, style), 1).cpu().data.numpy()[0]], _opt.pose_inference_repose.cuda(), beta.cuda())

            smooth_normals = model_trimesh.vertex_normals.copy()
            normals = model_trimesh.vertex_normals.copy()

            # Unpose+Pose if it's lower body, and pose directly if it's upper body:
            # 将 T-pose 下的衣物网格，通过蒙皮算法穿到目标姿态上
            if cloth_optimization_index == 9 or cloth_optimization_index == 12:
                model_trimesh, normals = image_fitting.unpose_and_deform_cloth_w_normals(model_trimesh, _opt.pose_inference_repose.cpu(), pose, beta.cpu(), J, v, SMPL_Layer, normals)
            else:
                model_trimesh, normals = image_fitting.batch_posing_w_normals(model_trimesh, normals, pose, J, v, SMPL_Layer)

            # Smooth meshes:
            # 对最终的网格进行平滑处理，使其更自然
            model_trimesh = trimesh.smoothing.filter_laplacian(model_trimesh,lamb=0.5)

            # Save predictions before rendering all of them together
            posed_meshes.append(model_trimesh)
            posed_normals.append(normals)
            colors.append(_opt.color[0][:3])
            all_camscales.append(camScale)
            all_camtrans.append(camTrans)
            all_toplefts.append(topleft)
            all_scaleratios.append(scale_ratio)

    # --- 5. 最终渲染与视频输出 ---
    t = time.time()
    original_vertices = []
    for i in range(len(posed_meshes)):
        original_vertices.append(posed_meshes[i].vertices.copy())

    frames = [input_image]*10
    nframes = 100
    angles_y = np.arange(0, 360, 360/nframes)
    diffuminated_image = np.uint8(input_image.copy()*0.4)

    # To avoid person intersections in rendering, just assign person ordering according to person's scale:
    uniqueratios = np.unique(all_scaleratios)
    order = []
    for i in range(len(all_scaleratios)):
        which_smpl = np.where(all_scaleratios[i] == uniqueratios)[0][0]
        order.append(which_smpl/2.)

    # Render frames of persons rotating around Y axis:
    print("Rendering video")
    #! 
    renderer = image_fitting.meshRenderer()
    for i in tqdm.tqdm(range(len(angles_y))):
        angle_y = angles_y[i]
        c = np.cos(angle_y*np.pi/180)
        s = np.sin(angle_y*np.pi/180)
        rotmat_y = np.array([[c, 0, s], [0, 1, 0], [-1*s, 0, c]])

        for j in range(len(posed_meshes)):
            posed_meshes[j].vertices = np.matmul(rotmat_y, original_vertices[j].copy().T).T
            posed_meshes[j].vertices[:, 2] += order[j]

        renderImg_normals = image_fitting.render_image_projection_multiperson_wrenderer(diffuminated_image, posed_meshes, posed_normals, colors, all_camscales, all_camtrans, all_toplefts, all_scaleratios, mode='normals', renderer=renderer)
        if i == 0:
            for i in range(20):
                frames.append(renderImg_normals)
            diffuminated_image = np.uint8(diffuminated_image*0)
        frames.append(renderImg_normals)

    # Save video:
    print("Saving output video")
    image_fitting.save_video(frames, 'video_fits/%d_result_video'%t, freeze_first=5, freeze_last=5, framerate=24)
