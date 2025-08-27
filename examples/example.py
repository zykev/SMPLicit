import torch
import SMPLicit
import trimesh
import numpy as np
import os 

SMPLicit_layer = SMPLicit.SMPLicit()
meshes = SMPLicit_layer.reconstruct()

verts = np.concatenate((meshes[0].vertices, meshes[1].vertices))
faces = np.concatenate((meshes[0].faces, meshes[1].faces + len(meshes[0].vertices)))
mesh = trimesh.Trimesh(verts, faces)
# mesh.show()

# 2. 定义要保存的文件夹和文件名
output_folder = '/disk/work/kyzhang/human/4dhuman-develop/SMPLicit/example_save'
output_filename = 'T-Shirt.obj'  # 您可以更改文件名，.obj 是推荐的格式


output_path = os.path.join(output_folder, output_filename)
mesh.export(output_path)

# 6. (可选) 打印一条确认信息，告诉您文件已保存
print(f"网格已成功保存到: {output_path}")
