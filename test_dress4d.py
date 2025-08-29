
# %%
import os
import pickle
import numpy as np
import torch
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import trimesh


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
# grey, orange, purple, red, green, blue


# %%

# load pytorch3d cameras from parameters: intrinsics, extrinsics
def load_pytorch_cameras(camera_params, camera_list, image_shape):
    # init camera_dict
    camera_dict = dict()
    # process all camera within camera_list
    for camera_id in camera_list:
        # assign camera intrinsic and extrinsic matrices
        intrinsic = torch.tensor((camera_params[camera_id]["intrinsics"]), dtype=torch.float32).cuda()
        extrinsic = torch.tensor(camera_params[camera_id]["extrinsics"], dtype=torch.float32).cuda()
        # assign camera image size
        image_size = torch.tensor([image_shape[0], image_shape[1]], dtype=torch.float32).unsqueeze(0).cuda()

        # assign camera parameters
        f_xy = torch.cat([intrinsic[0:1, 0], intrinsic[1:2, 1]], dim=0).unsqueeze(0)
        p_xy = intrinsic[:2, 2].unsqueeze(0)
        R = extrinsic[:, :3].unsqueeze(0)
        T = extrinsic[:, 3].unsqueeze(0)
        # coordinate system adaption to PyTorch3D
        R[:, :2, :] *= -1.0
        # camera position in world space -> world position in camera space
        T[:, :2] *= -1.0
        R = torch.transpose(R, 1, 2)  # row-major
        # assign Pytorch3d PerspectiveCameras
        camera_dict[camera_id] = PerspectiveCameras(focal_length=f_xy, principal_point=p_xy, R=R, T=T, in_ndc=False, image_size=image_size).cuda()
    # assign Pytorch3d RasterizationSettings
    raster_settings = RasterizationSettings(image_size=image_shape, blur_radius=0.0, faces_per_pixel=1, max_faces_per_bin=80000)
    return camera_dict, raster_settings

# render pixel labels(nv, h, w) from mesh labels(nvt, ), faces(nft, 3), uvs(nvt, 2), and render_masks(nv, h, w)
def render_mesh_pixel_labels(labels, faces, render_rasts, render_masks, surface_labels):
    # get dimensions
    n_views, h, w = render_masks.shape[:3]
    # init label_votes (nv, h, w, nl) for multi-view images
    label_votes = torch.zeros((n_views, h, w, len(surface_labels))).cuda()
    # render labels for all views
    for nv in range(n_views):
        # get render pix_points, pix_to_faces, and pix_bary_coords
        pix_points = torch.nonzero(render_masks[nv])
        pix_to_faces = render_rasts.pix_to_face[nv][pix_points[:, 0], pix_points[:, 1]]
        pix_bary_coords = render_rasts.bary_coords[nv][pix_points[:, 0], pix_points[:, 1]]
        # project one pixel to one face if exists
        for nf in range(1):
            # get pixel_points(nvp, 2)
            pixel_points = pix_points[pix_to_faces[:, nf] >= 0]
            # get pixel_faces(nvp, ), pixel_bary_coords(nvp, 3), and pixel_faces_labels(nvp, 3)
            pixel_faces = pix_to_faces[pix_to_faces[:, nf] >= 0, nf] - faces.shape[0] * nv
            pixel_bary_coords = pix_bary_coords[pix_to_faces[:, nf] >= 0, nf]
            pixel_faces_labels = labels[faces[pixel_faces, :]]
            # assign votes to surface labels: skin, upper, lower, hair, shoe, outer
            for nl in range(len(surface_labels)):
                # loop over all face vertices
                for n in range(pixel_faces_labels.shape[-1]):
                    nl_labels = pixel_faces_labels[:, n] == nl
                    if torch.count_nonzero(nl_labels) == 0: continue
                    label_votes[nv, pixel_points[nl_labels][:, 0], pixel_points[nl_labels][:, 1], nl] += 1 * pixel_bary_coords[nl_labels, n]
    # collect render_labels from label_votes, filter label without votes
    render_labels = torch.max(label_votes, dim=-1).indices
    render_labels[torch.sum(label_votes, dim=-1) == 0] = -1
    return render_labels, label_votes

# render pixel labels (nv, h, w) to colors (nv, h, w, 3)
def render_pixel_label_colors(labels):
    # init parser_images with white background
    images = np.ones((*labels.shape[:3], 3)) * 255
    # loop over all parser images
    for nv in range(images.shape[0]):
        for nl in range(len(SURFACE_LABEL)):
            images[nv][labels[nv] == nl] = SURFACE_LABEL_COLOR[nl]
    return images.astype(np.uint8)

# extract label meshes for the entire subj_outfit_seq
def subj_outfit_seq_render_pixel_labels():
    # locate subj_outfit_seq_dir
    # # -------------------- Load Capture Cameras -------------------- # #
    # load camera_params
    image_shape = (1280, 940)
    camera_list = ['0076']
    camera_params = pickle.load(open('4ddress_sample/cameras.pkl', "rb"))
    # load pytorch3d camera_agents and raster_settings
    camera_agents, raster_settings = load_pytorch_cameras(camera_params, camera_list, image_shape)

    # # -------------------- Render Labels For All Cameras Frames -------------------- # #
    # process all frame

    # # -------------------- Load Scan Mesh and Label -------------------- # #
    # load scan_mesh to pytorch3d for rasterize
    scan_data = pickle.load(open('4ddress_sample/cloth_segmentation/mesh-f00006.pkl', "rb"))
    th_verts = torch.tensor(scan_data['vertices'], dtype=torch.float32).unsqueeze(0)
    th_faces = torch.tensor(scan_data['faces'], dtype=torch.long).unsqueeze(0)
    scan_mesh = Meshes(th_verts, th_faces).cuda()
    # load scan_labels
    scan_labels = pickle.load(open('4ddress_sample/cloth_segmentation/label-f00006.pkl', "rb"))['scan_labels']
    scan_labels = torch.tensor(scan_labels).cuda()

    # # -------------------- Process ALL Capture Cameras -------------------- # #
    for cam_id, camera_agent in camera_agents.items():
        # get capture_rasts from camera and scan_mesh
        capture_rasts = MeshRasterizer(cameras=camera_agent, raster_settings=raster_settings)(scan_mesh)
        # render capture_mask(h, w) and capture_mask_image(h, w)
        capture_mask = capture_rasts.pix_to_face[0, :, :, 0] > -1

        # # -------------------- Render Vertex Labels -------------------- # #
        # render scan_labels(nvt, ) to multi_view capture_labels(nv, h, w) and capture_labels_votes(nv, h, w, nl)
        capture_labels, capture_labels_votes = render_mesh_pixel_labels(
            scan_labels, th_faces.squeeze(0).cuda(), capture_rasts, capture_mask.unsqueeze(0), SURFACE_LABEL)
        # render multi-view render_labels(nv, h, w) to render_labels_images(nv, h, w, 3)
        capture_labels_images = render_pixel_label_colors(capture_labels.cpu().numpy())[0]

        Image.fromarray(capture_labels_images).save('4ddress_sample/cloth_segmentation/cloth-f00006.png')

# %%
# create cloth segmentation for 4ddress_sample
subj_outfit_seq_render_pixel_labels()


# %%
segmentation = cv2.imread('4ddress_sample/cloth_segmentation/cloth-f00006.png') # (h, w, 3)
segmentation = cv2.cvtColor(segmentation, cv2.COLOR_BGR2RGB)  # 转成 RGB

# %%
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


    # 分左右鞋
    shoe_idx = 3  # 原 shoe
    shoe_mask = label_map == shoe_idx
    shoe_mask_uint8 = shoe_mask.astype(np.uint8) * 255

    num_labels, labels_cc = cv2.connectedComponents(shoe_mask_uint8)
    H, W = label_map.shape
    shoe_centers = []
    for i in range(1, num_labels):
        ys, xs = np.where(labels_cc == i)
        center_x = xs.mean()
        shoe_centers.append((i, center_x))

    shoe_centers.sort(key=lambda x: x[1])  # 按 x 坐标排序
    left_shoe_label = shoe_centers[0][0]
    right_shoe_label = shoe_centers[1][0]

    # 分配新类别索引
    final_label_map = np.zeros_like(label_map)  # 背景默认 0
    final_label_map[label_map == 1] = 1  # skin
    final_label_map[label_map == 2] = 2  # hair
    final_label_map[(labels_cc == left_shoe_label)] = 3  # leftshoe
    final_label_map[(labels_cc == right_shoe_label)] = 4 # rightshoe

    # upper+outer 合并
    final_label_map[label_map == 4] = 5  # upper
    final_label_map[label_map == 5] = 6  # lower

    # lower
    final_label_map[label_map == 6] = 7  # outer

    return final_label_map

label_map = seg_to_label(segmentation)

color_array = np.vstack([np.array([[255, 255, 255]], dtype=np.uint8), 
                         SURFACE_LABEL_COLOR,
                         np.array([255, 255, 0], dtype=np.uint8)])

label_map_visual = color_array[label_map]

import matplotlib.pyplot as plt
plt.figure(figsize=(6,6))
plt.imshow(label_map_visual)
plt.axis('off')
plt.title('Segmentation Map with Left/Right Shoes')
plt.show()

# ['bg', 'skin', 'hair', 'leftshoe', 'rightshoe', 'upper', 'lower', 'outer']
# white, grey, orange, purple, red, green, blue, yellow
# 0, 1, 2, 3, 4, 5, 6, 7

# %%

smpl_prediction = pickle.load(open('4ddress_sample/smpl_prediction/mesh-f00006_smpl.pkl', 'rb'))

global_orient = torch.from_numpy(smpl_prediction['global_orient']) # (3,)
body_pose = torch.from_numpy(smpl_prediction['body_pose']) # (69,)
pose = torch.cat((global_orient, body_pose), 0).unsqueeze(0) # (1, 72)
beta = torch.from_numpy(smpl_prediction['betas']).unsqueeze(0) # (1, 10)
transl = torch.from_numpy(smpl_prediction['transl']).unsqueeze(0) # (1, 3)


# %%

def get_cameras(camera_path='4ddress_sample/cameras.pkl', W=1280, H=940):
    camera_info = pickle.load(open(camera_path, "rb"))
    cam_name = '0076'
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

# %%
import SMPLicit

# 生成基础 SMPL 身体和深度图
# Initialize SMPL-Related stuff:
SMPLicit_Layer = SMPLicit.SMPLicit()
SMPLicit_Layer = SMPLicit_Layer.cuda()
SMPL_Layer = SMPLicit_Layer.SMPL_Layer
smpl_faces = torch.LongTensor(SMPL_Layer.faces).cuda()

SMPL_Layer = SMPL_Layer.cuda()
v_posed = SMPL_Layer.forward(beta=beta.cuda(), theta=pose.cuda(), 
                                get_skin=True)[0][0] # [V, 3]
v_posed += transl.cuda()

K, R, T = get_cameras()
image_size = np.array([1280, 940])

# %%

depth_map = get_depth_map(v_posed, smpl_faces, K, R, T, image_size=image_size)




# %%

# 假设 depth_map 是 (H, W) 的 Tensor 或 numpy array
# 如果是 Tensor，先转 numpy
if 'torch' in str(type(depth_map)):
    depth_map = depth_map.cpu().numpy()

image = cv2.imread('4ddress_sample/images/capture-f00006.png') # (h, w, 3)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转成 RGB

# 方法 1: 灰度显示
plt.imshow(depth_map, cmap='gray')
plt.colorbar(label='Depth')
plt.title('Depth Map (Gray)')
plt.axis('off')
plt.show()

# 方法 2: 伪彩色显示，便于看深度变化
plt.imshow(depth_map, cmap='viridis')
plt.colorbar(label='Depth')
plt.title('Depth Map (Viridis)')
plt.axis('off')
plt.show()

plt.imshow(image)
plt.axis('off')
plt.show()


# %%
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


# %%
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

# %%
mesh_list = []
for i in range(6):
    mesh_list.append(trimesh.load('tmp/unposed_' + str(i) + '.obj'))

images = get_multi_mesh_render(mesh_list, K, R, T, image_size)


# %%
# 渲染单个 mesh
mesh = trimesh.load('tmp/unposed_1.obj')
images = get_mesh_render(mesh, K, R, T, image_size)

image = cv2.imread('4ddress_sample/images/capture-f00006.png') # (h, w, 3)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转成 RGB

# %%
fig, axes = plt.subplots(1, 2, figsize=(10, 5))  # 一行两列
axes[0].imshow(image)
axes[0].axis("off")
axes[0].set_title("Image")

axes[1].imshow(images)
axes[1].axis("off")
axes[1].set_title("Mesh render")

plt.tight_layout()
plt.show()


# %%
mesh = trimesh.load('tmp/unposed_1.obj')
mesh.is_watertight

# %%
import open3d as o3d
import open3d_examples as o3dtut

def check_properties(name, mesh):
    mesh.compute_vertex_normals()

    edge_manifold = mesh.is_edge_manifold(allow_boundary_edges=True)
    edge_manifold_boundary = mesh.is_edge_manifold(allow_boundary_edges=False)
    vertex_manifold = mesh.is_vertex_manifold()
    self_intersecting = mesh.is_self_intersecting()
    watertight = mesh.is_watertight()
    orientable = mesh.is_orientable()

    print(name)
    print(f"  edge_manifold:          {edge_manifold}")
    print(f"  edge_manifold_boundary: {edge_manifold_boundary}")
    print(f"  vertex_manifold:        {vertex_manifold}")
    print(f"  self_intersecting:      {self_intersecting}")
    print(f"  watertight:             {watertight}")
    print(f"  orientable:             {orientable}")

    # geoms = [mesh]
    # if not edge_manifold:
    #     edges = mesh.get_non_manifold_edges(allow_boundary_edges=True)
    #     geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 0)))
    # if not edge_manifold_boundary:
    #     edges = mesh.get_non_manifold_edges(allow_boundary_edges=False)
    #     geoms.append(o3dtut.edges_to_lineset(mesh, edges, (0, 1, 0)))
    # if not vertex_manifold:
    #     verts = np.asarray(mesh.get_non_manifold_vertices())
    #     pcl = o3d.geometry.PointCloud(
    #         points=o3d.utility.Vector3dVector(np.asarray(mesh.vertices)[verts]))
    #     pcl.paint_uniform_color((0, 0, 1))
    #     geoms.append(pcl)
    # if self_intersecting:
    #     intersecting_triangles = np.asarray(
    #         mesh.get_self_intersecting_triangles())
    #     intersecting_triangles = intersecting_triangles[0:1]
    #     intersecting_triangles = np.unique(intersecting_triangles)
    #     print("  # visualize self-intersecting triangles")
    #     triangles = np.asarray(mesh.triangles)[intersecting_triangles]
    #     edges = [
    #         np.vstack((triangles[:, i], triangles[:, j]))
    #         for i, j in [(0, 1), (1, 2), (2, 0)]
    #     ]
    #     edges = np.hstack(edges).T
    #     edges = o3d.utility.Vector2iVector(edges)
    #     geoms.append(o3dtut.edges_to_lineset(mesh, edges, (1, 0, 1)))
    # o3d.visualization.draw_geometries(geoms, mesh_show_back_face=True)

check_properties("unposed_1", o3d.io.read_triangle_mesh('tmp/unposed_1.obj'))
# %%

import open3d as o3d
mesh = o3d.io.read_triangle_mesh('tmp/unposed_5.obj') 
mesh.is_watertight()

# %%

# 1. 将原 mesh 转为点云
mesh = o3d.io.read_triangle_mesh("tmp/unposed_0.obj")
mesh.compute_vertex_normals()
pcd = mesh.sample_points_uniformly(number_of_points=50000)  # 采样点云

# 2. Poisson 重建
poisson_mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
    pcd, depth=10)  # depth 越大，细节越高

# 3. 根据密度裁剪掉离散点
vertices_to_keep = densities > np.quantile(densities, 0.01)
poisson_mesh = poisson_mesh.select_by_index(np.where(vertices_to_keep)[0])

# 4. 保存
o3d.io.write_triangle_mesh("tmp/unposed_revise_0.obj", poisson_mesh)
# %%
