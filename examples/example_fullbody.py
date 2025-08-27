import torch
import SMPLicit
import trimesh
import numpy as np
import os 

SMPLicit_layer = SMPLicit.SMPLicit()

upperbody_Z = np.zeros(18)
pants_Z = np.zeros(18)
hair_Z = np.zeros(18)
shoes_Z = np.zeros(4)
Zs = [upperbody_Z, pants_Z, hair_Z, shoes_Z]
meshes = SMPLicit_layer.reconstruct(model_ids=[0, 1, 3, 4], Zs=Zs)

verts = np.zeros((0, 3))
faces = np.zeros((0, 3))
for mesh in meshes:
    faces = np.concatenate((faces, mesh.faces + len(verts)))
    verts = np.concatenate((verts, mesh.vertices))

mesh = trimesh.Trimesh(verts, faces)
# mesh.show()

output_folder = '/disk/work/kyzhang/human/4dhuman-develop/SMPLicit/example_save'
output_filename = 'fullbody.obj'  # 您可以更改文件名，.obj 是推荐的格式


output_path = os.path.join(output_folder, output_filename)
mesh.export(output_path)

# 6. (可选) 打印一条确认信息，告诉您文件已保存
print(f"网格已成功保存到: {output_path}")
