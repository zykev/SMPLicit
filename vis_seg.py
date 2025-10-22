# %%
import sys
import os

# 获取当前脚本所在目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 将 submodule 的路径加入 Python 搜索路径
HP_PATH = os.path.join(ROOT_DIR, "submodules", "human_parsing")
sys.path.insert(0, HP_PATH)  # 插入到开头，优先搜索

from submodules.human_parsing.evaluate_simple import get_segmentation_map
from submodules.human_parsing.sapiens_seg import get_segmentation_sapiens

from dress4d_utils import adjust_segmentation_map

from thuman_utils import extract_files

import SMPLicit

import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

SMPLicit_Layer = SMPLicit.SMPLicit()
SMPLicit_Layer = SMPLicit_Layer.cuda()

# Initialize SMPL-Related stuff:
SMPL_Layer = SMPLicit_Layer.SMPL_Layer

def get_palette(num_cls):
    """ Returns the color map for visualizing the segmentation mask.
    Args:
        num_cls: Number of classes
    Returns:
        The color map
    """
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3
    return palette

def visualize_segmentation_batch(seg_batch: torch.Tensor, num_classes: int, ncol=20, figsize_per_image=(4,4), save_path=None):
    """
    Batch 可视化 segmentation
    Args:
        seg_batch: (B, H, W) torch.Tensor 或 numpy array
        num_classes: 类别总数，用于生成 palette
        ncol: 每行显示多少张图
        figsize_per_image: 每张图大小 (width, height)
        save_path: 可选，保存路径
    """
    if torch.is_tensor(seg_batch):
        seg_batch = seg_batch.cpu().numpy()

    B, H, W = seg_batch.shape
    nrow = (B + ncol - 1) // ncol

    palette = get_palette(num_classes)

    plt.figure(figsize=(figsize_per_image[0]*ncol, figsize_per_image[1]*nrow))

    for i in range(B):
        seg_img = Image.fromarray(seg_batch[i].astype(np.uint8), mode='P')
        seg_img.putpalette(palette)

        plt.subplot(nrow, ncol, i+1)
        plt.imshow(seg_img)
        plt.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    # plt.show()

# %%
folders = extract_files('.datasets/THuman')

segmentation_maps = get_segmentation_map(folders, image_size=[1024, 1024])
segmentation_maps_sapiens = get_segmentation_sapiens(folders, image_size=[1024, 1024])

# %%
# segmentation_maps_adjusted = adjust_segmentation_map(segmentation_maps, SMPL_Layer, folders)
segmentation_maps_adjusted = adjust_segmentation_map(segmentation_maps, segmentation_maps_sapiens, minor_adj=True)

visualize_segmentation_batch(segmentation_maps, num_classes=21, save_path='tmp/segmap_thuman.png')

visualize_segmentation_batch(segmentation_maps_sapiens, num_classes=21, save_path='tmp/segmap_thuman_sapiens.png')

# %%
visualize_segmentation_batch(segmentation_maps_adjusted, num_classes=21, save_path='tmp/segmap_adj_thuman.png')

"""
# %%
# 类别信息
labels = [
    "Background",
    "Hat", "Hair", "Glove", "Sunglasses", "Upper-Clothes", "Dress", "Coat",
    "Socks", "Pants", "Torso-Skin", "Scarf", "Skirt", "Face", "Left Arm",
    "Right Arm", "Left Leg", "Right Leg", "Left Shoe", "Right Shoe",
    "Exclude"
]

num_classes = len(labels)
palette = get_palette(num_classes)

# 创建调色板图像
color_img = np.zeros((50, num_classes * 50, 3), dtype=np.uint8)
for i in range(num_classes):
    color = palette[i*3:i*3+3]
    color_img[:, i*50:(i+1)*50, :] = color

plt.figure(figsize=(num_classes*0.5, 2))
plt.imshow(color_img)
plt.axis('off')

# 加类别标签
for i, label in enumerate(labels):
    plt.text(i*50 + 25, 55, label, rotation=90, ha='center', va='top', fontsize=8)

plt.show()

"""

# %%
import matplotlib.image as mpimg
import math
import matplotlib.pyplot as plt
import os

def visualize_images(image_paths, ncol=20, figsize_per_image=(4, 4), save_path=None):

    # 以组图形式可视化所有图片
    # Args:
    #     image_paths: 图片路径列表
    #     ncol: 每行的列数
    #     figsize_per_image: 每张图占的大小 (宽, 高)
    #     save_path: 可选，保存路径
 
    num_images = len(image_paths)
    nrow = math.ceil(num_images / ncol)

    fig, axes = plt.subplots(nrow, ncol, figsize=(figsize_per_image[0] * ncol,
                                                  figsize_per_image[1] * nrow))

    # 如果只有一行或一列，axes不是二维，需要 reshape
    axes = axes.flatten() if isinstance(axes, (list, np.ndarray)) else [axes]

    for i, ax in enumerate(axes):
        if i < num_images:
            img = mpimg.imread(image_paths[i])
            ax.imshow(img)
            ax.axis("off")
        else:
            ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)

# visualize render images
render_image_pths = []
for folder in folders:
    # cloth_path = os.path.join(folder['process_folder'], 'Meshes_cloth')
    # identity_name = os.path.basename(folder['process_folder']).split(".")[0]
    # cloth_path = os.path.join(os.path.dirname(folder['process_folder']).replace('param_smpl', 'Meshes_cloth'), identity_name)
    cloth_path = os.path.join('.datasets/THuman', 'Meshes_cloth', folder['process_folder'])
    img_path = glob.glob(f"{cloth_path}/render*.png")
    if len(img_path) > 0:
        img_path = img_path[0]
        render_image_pths.append(img_path)

visualize_images(render_image_pths, ncol=20, figsize_per_image=(4, 4), save_path='tmp/renders_huge100k.png')


# %%
# ori_images = []
# for folder in folders:
#     img_path = folder['path_image'][0]
#     # mask_path = img_path.replace('images', 'masks')

#     # 读取图像和掩码
#     image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
#     # mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
#     # mask = mask != 0
#     # image[~mask] = 255.
#     image = image / 255.

#     image = torch.from_numpy(image).permute(2, 0, 1).float()  # (3, H, W)


#     ori_images.append(image)

# ori_images = torch.stack(ori_images, dim=0)  # (B, 3, H, W)

# B = ori_images.shape[0]
# ncol = 20
# nrow = math.ceil(B / ncol)

# plt.figure(figsize=(ncol*2, nrow*2))
# for i in range(B):
#     plt.subplot(nrow, ncol, i+1)
#     plt.imshow(ori_images[i].permute(1, 2, 0).numpy())
#     plt.axis('off')

# plt.tight_layout()
# plt.savefig('tmp/ori_images_huge100k.png', dpi=150)




# # %%
# path_ls = []
# for folder in folders:
#     path_ls.append(folder['process_folder'])
# # %%
# path_ls.index('.datasets/4ddress/00187/Outer/Take14')
# # %%


