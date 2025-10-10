import os
import glob
import numpy as np
import torch
import json
import pickle

SELECT_DIR = [f"{i:04d}" for i in range(526)]
def extract_files(root_folder, image_size = [1024, 1024], select_dir=SELECT_DIR):
    res = []
    for subject_id in sorted(os.listdir(os.path.join(root_folder, 'process'))):
        if select_dir is not None and subject_id in select_dir:
            subject_dir = os.path.join(root_folder, 'process', subject_id)
            
            path_camera = os.path.join(subject_dir, 'camera.json')
            path_image = os.path.join(subject_dir, 'image', '00.png')
            path_smpl_prediction = os.path.join(root_folder, 'THuman2.0_smpl', f'{subject_id}_smpl.pkl')

            img_files = [path_image]
            smpl_files = [path_smpl_prediction]

            assert len(img_files) == len(smpl_files)

            # load camera
            with open(path_camera, 'r') as f:
                camera_info = json.load(f)
            cam_name = '00'

            R = torch.as_tensor(camera_info[cam_name]['R'], dtype=torch.float32)
            T = torch.as_tensor(camera_info[cam_name]['T'], dtype=torch.float32)
            K = torch.as_tensor(camera_info[cam_name]['K'], dtype=torch.float32)


            R = R.unsqueeze(0)
            T = T.unsqueeze(0)
            K = K.unsqueeze(0)
            
            # load smpl param
            smpl_params = pickle.load(open(path_smpl_prediction, 'rb'))

            betas = torch.as_tensor(smpl_params['betas'], dtype=torch.float32)
            global_orient = torch.as_tensor(smpl_params['global_orient'], dtype=torch.float32)
            body_pose = torch.as_tensor(smpl_params['body_pose'], dtype=torch.float32)

            transl = torch.as_tensor(smpl_params['transl'], dtype=torch.float32)
            # scale = torch.as_tensor(smpl_params['scale'], dtype=torch.float32)

            pose = torch.cat([global_orient, body_pose.reshape(1, -1)], dim=-1)

            res.append({
                'process_folder': subject_id,
                'camera_view': ['00'],
                'camera_params': (K, R, T),
                'path_image': img_files,
                'path_smpl': smpl_files,
                'pose': pose,
                'beta': betas,
                'transl': transl,
                # 'scale': scale
            })

    return res
