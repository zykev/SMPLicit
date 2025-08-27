import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    TexturesVertex
)
from pytorch3d.structures import Meshes

def render_depth_image(mesh_smpl, camScale, camTrans, topleft, scale_ratio, input_image):
    """
    Render depth map from SMPL mesh using PyTorch3D, directly in mesh coordinates.

    Args:
        mesh_smpl: trimesh.Trimesh object
        camScale: float, camera scale
        camTrans: list or np.array, camera translation (x, y)
        topleft: list or np.array, bbox top-left of person crop
        scale_ratio: float, scale ratio from crop to network input
        input_image: HxW or HxWx3 np.array, original image

    Returns:
        depth_image: HxW np.array, depth map in mesh coordinates
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    H, W = input_image.shape[:2]

    # 顶点和面
    verts = torch.tensor(mesh_smpl.vertices, dtype=torch.float32, device=device).unsqueeze(0)
    faces = torch.tensor(mesh_smpl.faces, dtype=torch.int64, device=device).unsqueeze(0)
    textures = TexturesVertex(verts_features=torch.ones_like(verts))
    mesh = Meshes(verts=verts, faces=faces, textures=textures)

    # 相机近似设置
    # PyTorch3D 相机使用 T 表示从相机到世界的平移
    # 我们根据原 camScale 和 camTrans 做近似投影
    # camScale 对应 focal length 近似，camTrans 对应平移
    cameras = FoVPerspectiveCameras(device=device)

    # 光栅化设置
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    # 渲染器只用 rasterizer 获取 z-buffer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        ),
        shader=HardPhongShader(device=device, cameras=cameras)
    )

    fragments = renderer.rasterizer(mesh)
    depth = fragments.zbuf[0, ..., 0].cpu().numpy()

    # 处理 NaN
    depth[np.isnan(depth)] = 0.0

    # 对应原图 crop
    depth_full = np.zeros((H, W), dtype=np.float32)
    # 计算缩放 bbox 大小
    crop_h = int(224 / scale_ratio)
    crop_w = int(224 / scale_ratio)
    min_y = max(0, int(topleft[1]))
    min_x = max(0, int(topleft[0]))
    max_y = min(H, min_y + crop_h)
    max_x = min(W, min_x + crop_w)

    # 直接放入裁剪区域
    depth_resized = np.array(depth)  # PyTorch3D 输出的大小默认和 raster_settings.image_size 一致
    # 如果需要，你可以插值 resize 到 crop_h, crop_w
    depth_resized = np.array(depth_resized)
    depth_crop = depth_resized[:(max_y-min_y), :(max_x-min_x)]
    depth_full[min_y:max_y, min_x:max_x] = depth_crop

    return depth_full
