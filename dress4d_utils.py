import os
import pickle
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import trimesh
import kaolin
import glob


from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, 
    RasterizationSettings, 
    MeshRasterizer,
    MeshRenderer, 
    PointLights, 
    SoftPhongShader,
)
from pytorch3d.renderer.mesh import TexturesVertex
from pytorch3d.utils import cameras_from_opencv_projection


SURFACE_LABEL = ['skin', 'hair', 'shoe', 'upper', 'lower', 'outer']
SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])
VIEWS = ['0004', '0028', '0052', '0076']  # left, back, right, front
# grey, orange, purple, red, green, blue


def extract_files(root_folder, subject_outfit= ['Inner', 'Outer'], select_view = '0076'):
    process_folders = []
    for subject_id in sorted(os.listdir(root_folder)):
        subject_dir = os.path.join(root_folder, subject_id)
        for outfit in subject_outfit:
            outfit_dir = os.path.join(subject_dir, outfit)
            if os.path.exist(outfit_dir):
                take_dir_list = sorted(os.listdir(outfit_dir))
                for take_id in take_dir_list:
                    take_dir = os.path.join(outfit_dir, take_id)
                    process_folders.append(take_dir)
            else:
                continue

    res = []
    for process_folder in process_folders:
        # process folder is one task for one outfit in one subject
        # print('Processing folder: ', process_folder)
        path_camera = os.path.join(process_folder, 'Capture/cameras.pkl')
        path_image = os.path.join(process_folder, 'Capture/', select_view, 'images')
        path_smpl_prediction = os.path.join(process_folder, 'SMPL')
        # path_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'images')
        # path_instance_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'masks')


        img_files = sorted(glob.glob(os.path.join(path_image, '*.png')))
        img_files = [img_files[0]]
        # mask_files = sorted(glob.glob(os.path.join(path_instance_segmentation, '*.png')))
        smpl_files = sorted(glob.glob(os.path.join(path_smpl_prediction, '*_smpl.pkl')))
        smpl_files = [smpl_files[0]]
        # seg_files = sorted(glob.glob(os.path.join(path_segmentation, '*.png')))

        assert len(img_files) == len(smpl_files)

        # load camera
        K, R, T = get_cameras(camera_path=path_camera, cam_name=select_view, W=1280, H=940)

        res.append({
            'process_folder': process_folder,
            'camera_view': select_view,
            'camera_params': (K, R, T),
            'path_image': img_files,
            'path_smpl': smpl_files,
        })

    return res

def extract_multiview_files(root_folder, subject_outfit= ['Inner', 'Outer'], select_views = None):
    process_folders = []
    for subject_id in os.listdir(root_folder):
        subject_dir = os.path.join(root_folder, subject_id)
        for outfit in subject_outfit:
            outfit_dir = os.path.join(subject_dir, outfit)
            take_dir_list = sorted(os.listdir(outfit_dir))
            for take_id in take_dir_list:
                take_dir = os.path.join(outfit_dir, take_id)
                process_folders.append(take_dir)

    res = []
    for process_folder in process_folders:
        # process folder is one task for one outfit in one subject
        # print('Processing folder: ', process_folder)
        path_camera = os.path.join(process_folder, 'Capture/cameras.pkl')
        all_views_img_files = []
        all_views_cam_params = []
        select_views = select_views if select_views is not None else VIEWS
        for select_view in select_views:
        	# load data for each view
            path_image = os.path.join(process_folder, 'Capture/', select_view, 'images')
            path_smpl_prediction = os.path.join(process_folder, 'SMPL')
            # path_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'images')
            # path_instance_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'masks')


            img_files = sorted(glob.glob(os.path.join(path_image, '*.png')))
            img_files = [img_files[0]]
            # mask_files = sorted(glob.glob(os.path.join(path_instance_segmentation, '*.png')))
            smpl_files = sorted(glob.glob(os.path.join(path_smpl_prediction, '*_smpl.pkl')))
            smpl_files = [smpl_files[0]]
            # seg_files = sorted(glob.glob(os.path.join(path_segmentation, '*.png')))

            assert len(img_files) == len(smpl_files)

            # load camera
            K, R, T = get_cameras(camera_path=path_camera, cam_name=select_view, W=1280, H=940)
            all_views_img_files.append(img_files)
            all_views_cam_params.append((K, R, T))
        img_files = list(map(list, zip(*all_views_img_files)))
        
        res.append({
            'process_folder': process_folder,
            'camera_view': VIEWS,
            'camera_params': all_views_cam_params,
            'path_image': img_files,
            'path_smpl': smpl_files,
        })

    return res


def seg_to_label(segmentation, colors=SURFACE_LABEL_COLOR):
    """
    segmentation: (H, W, 3) uint8 RGB
    colors: (num_classes, 3) uint8 RGB
    返回 (H, W) 的类别索引 map，背景是 0
    """
    h, w, _ = segmentation.shape
    # 扩展维度 (H, W, 1, 3) 和 (1, 1, C, 3) 做广播
    seg_exp = segmentation[:, :, None, :]        # (H, W, 1, 3)
    colors_exp = colors[None, None, :, :]        # (1, 1, C, 3)

    # 比较每个像素与每个颜色
    matches = np.all(seg_exp == colors_exp, axis=-1)  # (H, W, C) 布尔矩阵

    # 找到匹配的类别索引，从 1 开始，背景为 0
    label_map = np.zeros((h, w), dtype=np.uint8)
    # np.argmax 返回第一个最大值索引，但要先判断有没有匹配
    matched_any = matches.any(axis=-1)
    label_map[matched_any] = np.argmax(matches[matched_any], axis=-1) + 1


    # # 分左右鞋
    # shoe_idx = 3  # 原 shoe
    # shoe_mask = label_map == shoe_idx
    # shoe_mask_uint8 = shoe_mask.astype(np.uint8) * 255

    # num_labels, labels_cc = cv2.connectedComponents(shoe_mask_uint8)
    # H, W = label_map.shape
    # shoe_centers = []
    # for i in range(1, num_labels):
    #     ys, xs = np.where(labels_cc == i)
    #     center_x = xs.mean()
    #     shoe_centers.append((i, center_x))

    # shoe_centers.sort(key=lambda x: x[1])  # 按 x 坐标排序
    # left_shoe_label = shoe_centers[0][0]
    # right_shoe_label = shoe_centers[1][0]

    # # 分配新类别索引
    # final_label_map = np.zeros_like(label_map)  # 背景默认 0
    # final_label_map[label_map == 1] = 1  # skin
    # final_label_map[label_map == 2] = 2  # hair
    # final_label_map[(labels_cc == left_shoe_label)] = 3  # leftshoe
    # final_label_map[(labels_cc == right_shoe_label)] = 4 # rightshoe

    # # upper+outer 合并
    # final_label_map[label_map == 4] = 5  # upper
    # final_label_map[label_map == 5] = 6  # lower

    # # lower
    # final_label_map[label_map == 6] = 7  # outer

    return label_map


def get_cameras(camera_path='4ddress_sample/cameras.pkl', cam_name='0076', W=1280, H=940):
    camera_info = pickle.load(open(camera_path, "rb"))
    K = np.array(camera_info[cam_name]["intrinsics"], dtype=np.float32)
    extrinsic = camera_info[cam_name]["extrinsics"]
    R = np.array(extrinsic[:, :3], dtype=np.float32)
    T = np.array(extrinsic[:, 3], dtype=np.float32)

    M = np.eye(3, dtype=np.float32)
    M[0, 2] = (K[0, 2] - W / 2) / K[0, 0]
    M[1, 2] = (K[1, 2] - H / 2) / K[1, 1]
    K[0, 2] = W / 2
    K[1, 2] = H / 2
    R = M @ R
    T = M @ T.reshape(3, 1)

    R = torch.from_numpy(R).unsqueeze(0)
    T = torch.from_numpy(T).permute(1, 0)
    K = torch.from_numpy(K).unsqueeze(0)

    return K, R, T


def get_depth_map(vertices_pose, body_model_faces, K, R, T, image_size):
    device = torch.device("cuda:0")

    # 假设 verts: [V, 3]，faces: [F, 3]
    if type(vertices_pose) is np.ndarray:
        vertices_pose = torch.from_numpy(vertices_pose).to(device)
    if type(body_model_faces) is np.ndarray:
        body_model_faces = torch.from_numpy(body_model_faces).to(device)
    if type(image_size) is np.ndarray:
        image_size = torch.from_numpy(image_size).to(device)
    verts = vertices_pose.unsqueeze(0).clone()
    faces = body_model_faces.unsqueeze(0)
    image_size = image_size.unsqueeze(0)


    # Step 2: 构建 Meshes 对象
    meshes = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=torch.ones_like(verts)))

    # Step 3: 相机设置（默认视角）
    pytorch_cameras = cameras_from_opencv_projection(R, T, K, image_size)

    cameras = PerspectiveCameras(focal_length=pytorch_cameras.focal_length,
                                principal_point=pytorch_cameras.principal_point,
                                R=pytorch_cameras.R,
                                T=pytorch_cameras.T,
                                image_size=image_size,
                                in_ndc=True,
                                device=device)



    # Step 4: 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=(int(image_size[0, 0]), int(image_size[0, 1])),
        blur_radius=1e-6,
        faces_per_pixel=1,
        perspective_correct=True
    )



    # 渲染获取 depth 和 silhouette
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

    fragments = rasterizer(meshes)

    depth_map = fragments.zbuf[0, ..., 0]  # [H, W], 深度图


    return depth_map


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


# mesh renderer
def get_mesh_render(mesh, K, R, T, image_size):
    device = torch.device("cuda:0")

    # 假设 verts: [V, 3]，faces: [F, 3]
    verts = torch.from_numpy(mesh.vertices).float().to(device)
    faces = torch.from_numpy(mesh.faces).long().to(device)
    textures = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float().to(device) / 255.0

    if type(image_size) is not torch.Tensor:
        image_size = torch.from_numpy(image_size).to(device)

    verts = verts.unsqueeze(0).clone()
    faces = faces.unsqueeze(0)
    image_size = image_size.unsqueeze(0)


    # Step 2: 构建 Meshes 对象
    meshes = Meshes(verts=verts, faces=faces, textures=TexturesVertex(verts_features=textures.unsqueeze(0)))

    # Step 3: 相机设置（默认视角）
    pytorch_cameras = cameras_from_opencv_projection(R, T, K, image_size)

    cameras = PerspectiveCameras(focal_length=pytorch_cameras.focal_length,
                                principal_point=pytorch_cameras.principal_point,
                                R=pytorch_cameras.R,
                                T=pytorch_cameras.T,
                                image_size=image_size,
                                in_ndc=True,
                                device=device)



    # Step 4: 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=(int(image_size[0, 0]), int(image_size[0, 1])),
        blur_radius=1e-6,
        faces_per_pixel=1,
        perspective_correct=True
    )

    # Place a point light in front of the person. 
    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    # Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
    # interpolate the texture uv coordinates for each vertex, sample from a texture image and 
    # apply the Phong lighting model
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )
    # 渲染获取 images
    images = renderer(meshes)

    return images.squeeze(0)[..., :3].cpu().numpy()  # [H, W, 3], 图像



def get_multi_mesh_render(mesh_list, K, R, T, image_size):
    """
    渲染一个 list 的 trimesh 对象，每个 mesh 用固定颜色。
    
    mesh_list: list of trimesh.Trimesh
    """

    device = torch.device("cuda:0")

    verts_ls, faces_ls, textures_ls = [], [], []
    face_offset = 0

    for i, mesh in enumerate(mesh_list):
        verts = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).long().to(device)
        textures = torch.from_numpy(mesh.visual.vertex_colors[:, :3]).float().to(device) / 255.0

        verts_ls.append(verts)
        faces_ls.append(faces + face_offset)
        textures_ls.append(textures)

        face_offset += verts.shape[0]

    # 拼成 Meshes
    verts_ls = torch.cat(verts_ls, dim=0).unsqueeze(0)     # [1, V, 3]
    faces_ls = torch.cat(faces_ls, dim=0).unsqueeze(0)     # [1, F, 3]
    textures_ls = torch.cat(textures_ls, dim=0).unsqueeze(0)   # [1, V, 3]
    meshes = Meshes(
        verts=verts_ls,
        faces=faces_ls,
        textures=TexturesVertex(verts_features=textures_ls)
    )

    if type(image_size) is not torch.Tensor:
        image_size = torch.tensor(image_size, device=device).unsqueeze(0)

    # 相机
    pytorch_cameras = cameras_from_opencv_projection(R, T, K, image_size)
    cameras = PerspectiveCameras(
        focal_length=pytorch_cameras.focal_length,
        principal_point=pytorch_cameras.principal_point,
        R=pytorch_cameras.R,
        T=pytorch_cameras.T,
        image_size=image_size,
        in_ndc=True,
        device=device
    )

    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=(int(image_size[0, 0]), int(image_size[0, 1])),
        blur_radius=1e-6,
        faces_per_pixel=1,
        perspective_correct=True
    )

    lights = PointLights(device=device, location=[[0.0, 0.0, 2.0]])

    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras, 
            raster_settings=raster_settings
        ),
        shader=SoftPhongShader(
            device=device, 
            cameras=cameras,
            lights=lights
        )
    )

    # 渲染
    images = renderer(meshes)

    return images.squeeze(0)[..., :3].cpu().numpy()


def compute_udf_from_mesh(mesh: kaolin.rep.SurfaceMesh, points: torch.Tensor) -> torch.Tensor:
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
    vertices_batch = mesh.vertices[0].to(points.device).unsqueeze(0).contiguous()
    faces = mesh.faces[0].to(points.device, dtype=torch.int64).contiguous()
    points_batch = points.to(vertices_batch.device).unsqueeze(0).contiguous()

    # 1. 提取出每个三角面片的顶点坐标
    # 这个是 point_to_mesh_distance 和 check_sign 都需要的输入
    face_vertices = kaolin.ops.mesh.index_vertices_by_faces(vertices_batch, faces)

    # 2. 计算点到网格表面的无符号距离（的平方）
    # 遵循图中建议，这是第一步
    distance_sq, face_indices, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
        points_batch, face_vertices
    )
    
    # 3. 对距离的平方进行开方，得到真实的距离
    # 遵循图中 "be careful to apply torch.sqrt" 的建议
    distance = torch.sqrt(distance_sq)

    # 4. 判断点的符号（内外）
    # 遵循图中建议，这是第二步
    # 注意：check_sign 要求网格是水密的 (watertight)
    # is_inside = kaolin.ops.mesh.check_sign(vertices_batch, faces, points_batch)
    
    # 5. 结合距离和符号
    # 将布尔值的 is_inside (True for inside) 转换为数值符号 (-1 for inside, 1 for outside)
    # sign = torch.where(is_inside, -1.0, 1.0).to(distance.device)
    
    # signed_distance = sign * distance
    
    # 返回结果时去掉批处理维度，使其与输入点的维度匹配
    return distance.squeeze(0)


def combine_meshes(meshes):
    """
    将多个 trimesh 对象合并为一个，保留顶点颜色。
    """
    combined_vertices = []
    combined_faces = []
    combined_colors = []
    offset = 0

    for m in meshes:
        combined_vertices.append(m.vertices)
        combined_faces.append(m.faces + offset)
        combined_colors.append(m.visual.vertex_colors)
        offset += len(m.vertices)

    combined_vertices = np.vstack(combined_vertices)
    combined_faces = np.vstack(combined_faces)
    combined_colors = np.vstack(combined_colors)

    combined_mesh = trimesh.Trimesh(
        vertices=combined_vertices,
        faces=combined_faces,
        vertex_colors=combined_colors
    )

    return combined_mesh


from scipy.spatial.transform import Rotation

def get_02v_pose(num_joints=24):
    """
    Get SMPL pose parameters (1, 72) for Vitruvian A-pose.
    
    Args:
        Jtr (torch.Tensor): Joint locations of shape (24 or 55, 3)
    
    Returns:
        torch.Tensor: SMPL pose (1, 72), axis-angle representation
    """

    # Initialize axis-angle pose with zeros (no rotation)
    smpl_pose = torch.zeros((1, num_joints * 3), dtype=torch.float32)

    # Define rotation matrices
    rot45p = Rotation.from_euler('z', 45, degrees=True).as_matrix()
    rot45n = Rotation.from_euler('z', -45, degrees=True).as_matrix()

    # Convert to axis-angle
    aa45p = torch.tensor(Rotation.from_matrix(rot45p).as_rotvec(), dtype=torch.float32)
    aa45n = torch.tensor(Rotation.from_matrix(rot45n).as_rotvec(), dtype=torch.float32)

    # Left hip (index 1 in SMPL)
    smpl_pose[0, 1*3:1*3+3] = aa45p
    # Right hip (index 2 in SMPL)
    smpl_pose[0, 2*3:2*3+3] = aa45n

    return smpl_pose