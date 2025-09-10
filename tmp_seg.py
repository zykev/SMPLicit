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

from dress4d_utils import extract_files


import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt



# def extract_files(root_folder, subject_outfit= ['Inner', 'Outer'], select_view = '0004'):
#     process_folders = []
#     for subject_id in sorted(os.listdir(root_folder)):
#         if subject_id in ['00148', '00149_1', '00149_2']:
#             subject_dir = os.path.join(root_folder, subject_id)
#             for outfit in subject_outfit:
#                 outfit_dir = os.path.join(subject_dir, outfit)
#                 if os.path.exists(outfit_dir):
#                     take_dir_list = sorted(os.listdir(outfit_dir))
#                     for take_id in take_dir_list:
#                         take_dir = os.path.join(outfit_dir, take_id)
#                         process_folders.append(take_dir)
#                 else:
#                     continue

#     res = []
#     for process_folder in process_folders:
#         # process folder is one task for one outfit in one subject
#         # print('Processing folder: ', process_folder)
#         path_image = os.path.join(process_folder, 'Capture/', select_view, 'images')
#         path_smpl_prediction = os.path.join(process_folder, 'SMPL')
#         # path_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'images')
#         # path_instance_segmentation = os.path.join(process_folder, 'Capture/', select_view, 'masks')


#         img_files = sorted(glob.glob(os.path.join(path_image, '*.png')))
#         img_files = [img_files[0]]
#         # mask_files = sorted(glob.glob(os.path.join(path_instance_segmentation, '*.png')))
#         smpl_files = sorted(glob.glob(os.path.join(path_smpl_prediction, '*_smpl.pkl')))
#         smpl_files = [smpl_files[0]]
#         # seg_files = sorted(glob.glob(os.path.join(path_segmentation, '*.png')))

#         assert len(img_files) == len(smpl_files)

#         res.append({
#             'process_folder': process_folder,
#             'camera_view': select_view,
#             'path_image': img_files,
#             'path_smpl': smpl_files,
#         })

#     return res


def adjust_segmentation_map(seg1, seg2):
    seg1 = seg1.clone()

    B, H, W = seg1.shape

    for b in range(B):
        s1 = seg1[b]
        s2 = seg2[b]

        for cls, thresh in [(2,100), (5,50), (6,50), (7,50), (9,50), (12,50), (18,50)]:
            if (s1 == cls).sum() < thresh:
                s1[s1 == cls] = 25

         # 2. dress=6 → 拆成 upper=5 和 skirts=12
        if (s1 == 6).any():
            mask_dress = (s1 == 6)

            mask_upper = mask_dress & (s2 == 2)   # dress ∩ upper cloth
            mask_skirt = mask_dress & (s2 == 1)   # dress ∩ lower cloth

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
        # 4. 没有 outer=7，但有 upper=5
        if (s1 == 5).any() and not (s1 == 7).any():
            s1[(s1 == 5) & (s2 != 2)] = 25
            s1[(s1 == 5) | (s2 == 2)] = 5
        
        if (s1 == 7).any() and not (s1 == 5).any():
            s1[(s1 == 7) & (s2 != 2)] = 25
            # s1[(s1 == 7) | (s2 == 2)] = 7
        
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
        if (s1 == 12).any() and not (s1 == 9).any():
            # 5. 没有 pants=9，但有 skirts=12
            s1[(s1 == 12) & (s2 != 1)] = 25
            s1[(s1 == 12) | (s2 == 1)] = 12
        
        if (s1 == 9).any() and not (s1 == 12).any():
            # 6. 没有 skirts=12，但有 pants=9
            s1[(s1 == 9) & (s2 != 1)] = 25
            s1[(s1 == 9) | (s2 == 1)] = 9

        seg1[b] = s1

    return seg1

# 只拟合upper, pants, skirts
# def adjust_segmentation_map(seg1, seg2):
#     seg1 = seg1.clone()

#     B, H, W = seg1.shape

#     for b in range(B):
#         s1 = seg1[b]
#         s2 = seg2[b]

#         for cls, thresh in [(2,100), (5,50), (6,50), (7,50), (9,50), (12,50), (18,50)]:
#             if (s1 == cls).sum() < thresh:
#                 s1[s1 == cls] = 25

#          # 2. dress=6 → 拆成 upper=5 和 skirts=12
#         if (s1 == 6).any():
#             s1[s1 == 9] = 6
#             s1[s1 == 12] = 6
#             mask_dress = (s1 == 6)

#             mask_upper = mask_dress & (s2 == 2)   # dress ∩ upper cloth
#             mask_skirt = mask_dress & (s2 == 1)   # dress ∩ lower cloth

#             s1[mask_upper] = 5
#             s1[mask_skirt] = 12
            
#         # 1. outer=7 存在 → upper=7 改成 5
#         s1[s1 == 7] = 5
#         s1[(s1 == 5) & (s2 != 2)] = 25
#         s1[(s1 == 5) | (s2 == 2)] = 5

        
#         # 3. pants=9 与 skirts=12 同时出现 → 按面积选择大类
#         if (s1 == 9).any() and (s1 == 12).any():
#             cnt_pants = (s1 == 9).sum().item()
#             cnt_skirt = (s1 == 12).sum().item()

#             if cnt_pants >= cnt_skirt:
#                 # 保留 pants=9，把 skirt=12 合并为 pants
#                 s1[s1 == 12] = 9
#             else:
#                 # 保留 skirts=12，把 pants=9 合并为 skirts
#                 s1[s1 == 9] = 12
#         if (s1 == 12).any() and not (s1 == 9).any():
#             # 5. 没有 pants=9，但有 skirts=12
#             s1[(s1 == 12) & (s2 != 1)] = 25
#             s1[(s1 == 12) | (s2 == 1)] = 12
        
#         if (s1 == 9).any() and not (s1 == 12).any():
#             # 6. 没有 skirts=12，但有 pants=9
#             s1[(s1 == 9) & (s2 != 1)] = 25
#             s1[(s1 == 9) | (s2 == 1)] = 9

#         seg1[b] = s1

#     return seg1

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
folders = extract_files('.datasets/4ddress')

segmentation_maps = get_segmentation_map(folders)
segmentation_maps_sapiens = get_segmentation_sapiens(folders)

# %%
segmentation_maps_adjusted = adjust_segmentation_map(segmentation_maps, segmentation_maps_sapiens)

visualize_segmentation_batch(segmentation_maps, num_classes=21, save_path='tmp/segmap.png')


# %%
visualize_segmentation_batch(segmentation_maps_adjusted, num_classes=21, save_path='tmp/segmap_adj.png')


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
# %%
