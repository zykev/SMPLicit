import os
import glob
import numpy as np
import torch

SELECT_DIR = ["my_377", "my_386", "my_387", "my_392", "my_393", "my_394"]
VIEW_MAP = {"my_377": "04", "my_386": "07", "my_387": "04", "my_392": "04", "my_393": "06", "my_394": "04"}
def extract_files(root_folder, image_size = [1024, 1024], select_dir=SELECT_DIR):
    res = []
    for subject_id in sorted(os.listdir(root_folder)):
        if select_dir is not None and subject_id in select_dir:
            subject_dir = os.path.join(root_folder, subject_id)
            
            select_view = VIEW_MAP[subject_id]
            path_camera = os.path.join(subject_dir, 'annots.npy')
            path_image = os.path.join(subject_dir, 'images', select_view)
            path_smpl_prediction = os.path.join(subject_dir, 'smpl_params')


            img_files = sorted(glob.glob(os.path.join(path_image, '*.jpg')))
            img_files = [img_files[0]]
            smpl_files = sorted(glob.glob(os.path.join(path_smpl_prediction, '*.npy')))
            smpl_files = [smpl_files[0]]

            assert len(img_files) == len(smpl_files)

            # load camera
            camera_info = np.load(path_camera, allow_pickle=True).item()

            K = np.array(camera_info['cams']['K'][int(select_view)], dtype=np.float32)
            # D = np.array(camera_info['cams']['D'][int(select_view)], dtype=np.float32)
            R = np.array(camera_info['cams']['R'][int(select_view)], dtype=np.float32)
            T = np.array(camera_info['cams']['T'][int(select_view)], dtype=np.float32) / 1000.

            H, W = image_size
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

            res.append({
                'process_folder': subject_dir,
                'camera_view': [select_view],
                'camera_params': (K, R, T),
                'path_image': img_files,
                'path_smpl': smpl_files,
            })

    return res