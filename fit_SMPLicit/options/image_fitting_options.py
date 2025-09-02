import argparse
import torch
import os
import numpy as np

# HUMAN PARSING LABELS:
# 1 -> Hat
# 2 -> Hair
# 3 -> Glove
# 4 -> Sunglasses,
# 5 -> Upper-Clothes, *
# 6 -> Dress,
# 7 -> Coat, *
# 8 -> Socks,
# 9 -> Pants, *
# 10 -> Torso-Skin
# 11 -> Scarf
# 12 -> Skirt *
# 13 -> Face
# 14 -> Left Arm
# 15 -> Right Arm
# 16 -> Left Leg
# 17 -> Right Leg
# 18 -> Left Shoe *
# 19 -> Right Shoe

# 4ddress
# 0 -> background, white
# 1 -> skin, grey
# 2 -> hair, orange *
# 3 -> left shoe, purple *
# 4 -> right shoe, red
# 5 -> upper clothes, green *
# 6 -> lower clothes, blue *
# 7 -> outer clothes, yellow *

class FitOptions():
    def __init__(self):
        self._parser = argparse.ArgumentParser()

    # def set_segmentation(self, segmentation):
    #     self.segmentation = segmentation

    #     labels = np.unique(self.segmentation)
    #     labels = np.intersect1d(labels, [5, 7, 9, 12, 18])
    #     self._opt.labels = labels
    #     return self._opt
    
    def set_segmentation(self, segmentation):
        self.segmentation = segmentation

        labels = np.unique(self.segmentation)
        labels = np.intersect1d(labels, [5, 6, 7]) # [2, 3, 5, 6, 7]
        self._opt.labels = labels
        return self._opt
    
    def set_labels(self):

        self._opt.labels = [5, 6, 7]

        return self._opt


    def initialize(self):
        # self._parser.add_argument('--image_folder', type=str, default='fit_SMPLicit/data/images/', help='folder with input images')
        # self._parser.add_argument('--smpl_prediction_folder', type=str, default='fit_SMPLicit/data/smpl_prediction/', help='folder with input images')
        # self._parser.add_argument('--cloth_segmentation_folder', type=str, default='fit_SMPLicit/data/cloth_segmentation/', help='folder with input images')
        # self._parser.add_argument('--instance_segmentation_folder', type=str, default='fit_SMPLicit/data/instance_segmentation/', help='folder with input images')
        # self._parser.add_argument('--image_extension', type=str, default='.jpg', help='image extension (.png, .jpg, etc)')
        # self._parser.add_argument('--camera_folder', type=str, default='fit_SMPLicit/data/cameras/', help='folder with input cameras')

        self._parser.add_argument('--root_folder', type=str, default='.datasets/4ddress/', help='folder with input images')
        self._parser.add_argument('--camera_view', type=str, default='0076', help='camera view to use (0004, 0028, 0052, 0076)')
        self._parser.add_argument('--save_folder', type=str, default='.datasets/4ddress_cloth/', help='folder to save unposed meshes')
        self._parser.add_argument('--image_folder', type=str, default='4ddress_sample/images/', help='folder with input images')
        self._parser.add_argument('--smpl_prediction_folder', type=str, default='4ddress_sample/smpl_prediction/', help='folder with input images')
        self._parser.add_argument('--cloth_segmentation_folder', type=str, default='4ddress_sample/cloth_segmentation/', help='folder with input images')
        self._parser.add_argument('--instance_segmentation_folder', type=str, default='4ddress_sample/instance_segmentation/', help='folder with input images')
        self._parser.add_argument('--image_extension', type=str, default='.png', help='image extension (.png, .jpg, etc)')
        self._parser.add_argument('--camera_folder', type=str, default='4ddress_sample/cameras.pkl', help='folder with input cameras')

        # Optimization parameters:
        self._parser.add_argument('--lr', type=float, default=0.01)
        self._parser.add_argument('--lr_decayed', type=float, default=0.0003)
        self._parser.add_argument('--step', type=int, default=10000)
        self._parser.add_argument('--iterations', type=int, default=10) # Decrease to 100 if it's too slow
        self._parser.add_argument('--index_samples', type=int, default=100)
        self._parser.add_argument('--is_train', type=bool, default=False)
        self._parser.add_argument('--do_videos', type=bool, default=False)
        self._parser.add_argument('--resolution', type=int, default=64)

        # General options:
        self._parser.add_argument('--clusters', type=str, default='')
        self._parser.add_argument('--num_clusters', type=int, default=500)
        self._parser.add_argument('--clamp_value', type=float, default=0.5)
        self._parser.add_argument('--num_params_style', type=int, default=0)
        self._parser.add_argument('--num_params_shape', type=int, default=0)
        self._parser.add_argument('--other_labels', type=str, default='')
        self._parser.add_argument('--repose', type=bool, default=False)
        self._parser.add_argument('--b_min', type=str, default='')
        self._parser.add_argument('--b_max', type=str, default='')
        self._parser.add_argument('--color', type=str, default='')
        
        self._initialized = True

    def parse(self):
        self.initialize()
        self._opt = self._parser.parse_args()
        return self._opt

    def update_optimized_cloth(self, optimization_index):
        #self._opt.lr = 0.01

        # TODO: ADD INITIALIZATION PARAMETERS (eg skirt different than pants):
        if optimization_index == 5: # upperbody, ori: 5 T-Shirt
            self._opt.index_cloth = 0
            self._opt.clusters = 'indexs_clusters_tshirt_smpl.npy'
            self._opt.num_clusters = 500
            self._opt.clamp_value =  0.1
            self._opt.num_params_style = 12
            self._opt.num_params_shape = 6
            self._opt.other_labels = [9, 11, 7, 2]
            self._opt.b_min = [-0.7, -0.2, -0.3]
            self._opt.b_max = [0.7, 0.6, 0.3]
            self._opt.weight_positives = 3
            self._opt.weight_negatives = 3
            self._opt.color = [[160, 220, 160, 255]]
            self._opt.res = 128
            self._opt.repose = False
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()

        elif optimization_index == 7: # ori: 7 Coat
            self._opt.index_cloth = 0
            self._opt.clusters = 'indexs_clusters_tshirt_smpl.npy'
            self._opt.num_clusters = 500
            self._opt.clamp_value =  0.1
            self._opt.num_params_style = 12
            self._opt.num_params_shape = 6
            self._opt.other_labels = [11, 2]
            self._opt.b_min = [-0.8, -0.2, -0.3]
            self._opt.b_max = [0.8, 0.6, 0.3]
            self._opt.weight_positives = 10
            self._opt.weight_negatives = 3
            self._opt.color = [[120, 120, 120, 255]]
            self._opt.res = 128
            self._opt.repose = False
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()

        elif optimization_index == 6: # lowerbody, ori 9, Pants
            self._opt.index_cloth = 1
            self._opt.clusters = 'clusters_lowerbody.npy'
            self._opt.num_clusters = 500
            self._opt.clamp_value =  0.1
            self._opt.num_params_style = 12
            self._opt.num_params_shape = 6
            self._opt.other_labels = [5, 7]
            self._opt.b_min = [-0.2, -1.2, -0.3]
            self._opt.b_max = [0.2, -0.2, 0.3]
            self._opt.weight_positives = 10
            self._opt.weight_negatives = 3
            self._opt.color = [[90, 120, 180, 255]]
            self._opt.res = 128
            self._opt.repose = True
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()
            self._opt.pose_inference_repose[0, 5] = 0.04
            self._opt.pose_inference_repose[0, 8] = -0.04

        elif optimization_index == 12: # Skirts
            self._opt.index_cloth = 2
            self._opt.clusters = 'clusters_lowerbody.npy'
            self._opt.num_clusters = 500
            self._opt.clamp_value =  0.1
            self._opt.num_params_style = 12
            self._opt.num_params_shape = 6
            self._opt.other_labels = [5, 7]
            self._opt.b_min = [-0.2, -1.2, -0.25]
            self._opt.b_max = [0.2, 0.1, 0.25]
            self._opt.weight_positives = 600
            self._opt.weight_negatives = 3
            self._opt.color = [[120, 180, 255, 255]]
            self._opt.res = 128
            self._opt.repose = True
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()
            self._opt.pose_inference_repose[0, 5] = 0.04
            self._opt.pose_inference_repose[0, 8] = -0.04

        elif optimization_index == 3: # ori 18, Shoe
            self._opt.index_cloth = 4
            self._opt.clusters = 'clusters_shoes.npy'
            self._opt.num_clusters = 100
            self._opt.clamp_value =  0.01
            self._opt.num_params_style = 4
            self._opt.num_params_shape = 6
            self._opt.other_labels = []
            self._opt.b_min = [-0.1, -1.4, -0.2]
            self._opt.b_max = [0.25, -0.6, 0.3]
            self._opt.weight_positives = 5
            self._opt.weight_negatives = 3
            self._opt.color = [[100, 100, 100, 255]]
            self._opt.res = 128
            self._opt.repose = False
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()

        elif optimization_index == 2: # Hair
            self._opt.index_cloth = 3
            self._opt.clusters = 'clusters_hairs.npy'
            self._opt.num_clusters = 500
            self._opt.num_params_style = 12
            self._opt.num_params_shape = 6
            self._opt.other_labels = [1, 11]
            hair_offset = np.array([[0, -1.32, 0.02]])
            self._opt.b_min = [-0.3, 0.9, -0.35] + hair_offset[0]
            self._opt.b_max = [0.3, 2., 0.35] + hair_offset[0]
            # Weight for positives much lower in this case, since model already tends to generate more hair than necessary
            # That's essentially because most hair train data was long hair
            self._opt.weight_positives = 5#20
            self._opt.weight_negatives = 1
            self._opt.color = [[220, 220, 160, 255]]
            self._opt.res = 128
            self._opt.repose = False
            self._opt.pose_inference_repose = torch.zeros(1, 72).cuda()

        return self._opt
