import numpy as np

def get_training_arrays_per_image(_opt, input_image, depth_image_smpl, segmentation, instance_segmentation, coords_2d, coords, cloth_optimization_index):
    """
    为单张图像获取训练数组
    Args:
        _opt: 配置选项
        input_image: 输入图像 (H, W, 3)
        depth_image_smpl: SMPL 深度图 (H, W)
        segmentation: 分割图 (H, W)，每个像素的值为类别标签
        instance_segmentation: 实例分割图 (H, W)，每个像素的值为实例ID
        coords_2d: 3D点对应的2D像素坐标 (N, 2)
        coords: 3D点坐标 (N, 3)
        cloth_optimization_index: 当前优化的衣物类别索引
    Returns:
        array_pixels: 参与优化的3D点索引 (M,)
        array_gt: 对应的目标值 (M,)
    """
    H, W = input_image.shape[:2]

    # step1: 建立像素到 indices 的映射 (不排序，不打乱)
    pixel_map = {}
    for i, (y, x) in enumerate(coords_2d):
        key = (x, y)  # 注意你原始代码的循环是 (x,y)
        if key not in pixel_map:
            pixel_map[key] = []
        pixel_map[key].append(i)

    array_pixels = []
    array_gt = []

    # step2: 严格按照 (x,y) 双循环顺序来遍历
    for x in range(H):
        for y in range(W):
            if (x, y) not in pixel_map:
                continue
            if instance_segmentation[x, y] == 0 and cloth_optimization_index != 2:
                continue
            if segmentation[x, y] in _opt.other_labels:
                continue

            depth_smpl = depth_image_smpl[x, y]
            if depth_smpl == 0:
                depth_smpl = np.inf

            inds = np.array(pixel_map[(x, y)], dtype=np.int32)
            inds = inds[coords[inds, 2] < depth_smpl]
            if len(inds) == 0:
                continue

            gt_val = _opt.clamp_value - (segmentation[x, y] == cloth_optimization_index) * _opt.clamp_value
            array_pixels.extend(inds.tolist())
            array_gt.extend([gt_val] * len(inds))

    array_pixels = np.array(array_pixels, dtype=np.int32)
    array_gt = np.array(array_gt, dtype=np.float32)


    return array_pixels, array_gt


def get_training_arrays(_opt, input_image, depth_image_smpl, segmentation, instance_segmentation, coords_2d, coords, cloth_optimization_index):
    H, W = input_image.shape[:2]
    coords_2d = coords_2d.astype(np.int32)

    # 将 coords_2d 转换成 pixel_keys, 用于排序
    pixel_keys = coords_2d[:, 0] * W + coords_2d[:, 1]  # row*x, col*y

    # depth 和 seg 同样扁平化
    seg_flat = segmentation.ravel()
    ins_flat = instance_segmentation.ravel()
    depth_flat = depth_image_smpl.ravel()

    linear_idx = np.ravel_multi_index((coords_2d[:,0], coords_2d[:,1]), (H, W))

    # 每个点对应的像素条件一次性筛选
    mask = ((ins_flat[linear_idx] != 0) | (cloth_optimization_index == 2)) & (~np.isin(seg_flat[linear_idx], _opt.other_labels))

    # depth 条件
    depth_vals = depth_flat[linear_idx].copy()
    depth_vals[depth_vals == 0] = np.inf
    mask &= coords[:,2] < depth_vals

    # 先筛选有效点
    valid_inds = np.where(mask)[0]

    # 为了保持原来的循环顺序，先根据像素 key 排序
    sorted_order = np.argsort(pixel_keys[valid_inds])
    valid_inds = valid_inds[sorted_order]

    array_pixels = valid_inds.astype(np.int32)

    seg_vals = seg_flat[linear_idx[array_pixels]]
    array_gt = np.where(seg_vals == cloth_optimization_index, 0.0, _opt.clamp_value).astype(np.float32)

    return array_pixels, array_gt

SURFACE_LABEL_COLOR = np.array([[128, 128, 128], [255, 128, 0], [128, 0, 255], [180, 50, 50], [50, 180, 50], [0, 128, 255]])

def visual_seg(segmentation):
    color_array = np.vstack([np.array([[255, 255, 255]], dtype=np.uint8), 
                         SURFACE_LABEL_COLOR,
                         np.array([255, 255, 0], dtype=np.uint8)])

    label_map_visual = color_array[segmentation]

    return label_map_visual

