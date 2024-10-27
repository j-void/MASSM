import glob
import os
from tqdm import tqdm
import shapeworks as sw
import ShapeCohortGen
import os
import numpy as np
import argparse
import os
import glob
import math
import vtk
import numpy as np
import json
from torch.nn import functional as F
import torch
from collections import defaultdict

"""
Preprocess the total segmentator dataset
========================================

1. Load the data from the complete_shapes.csv file
2. Resample the data to the same size (768x672x1344)
3. Resize to 256x224x448
4. Save the data in the correct format

The script saves the data in the {save_path} - change accordingly
Also the path to - complete_shapes of total segmentator
"""

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

save_path = "/home/sci/janmesh/DeepSSM/Janmesh_Data"
anatomies = ['heart_ventricle_left', 'heart_ventricle_right', 'heart_atrium_left', \
    'heart_atrium_right', 'lung_lower_lobe_right', 'spleen', 'lung_upper_lobe_left', 'lung_upper_lobe_right']

make_dir(os.path.join(save_path, "images"))
for ca in anatomies:
    make_dir(os.path.join(save_path, "segmentations", ca))
    make_dir(os.path.join(save_path, "meshes", ca))

import pandas as pd
df = pd.read_csv("/home/sci/janmesh/DeepSSM/DeepSSM_pose/datasets/total_segmentator/complete_shapes.csv")
rows = len(df.axes[0])

out_dict = defaultdict(list)

max_shape = np.array([768, 672, 1344])
rz_size = max_shape//3

iso_spacing = [1, 1, 1]

def transform_volume(img):
    img.resample([1, 1, 1], sw.InterpolationType.Linear)
    img.setOrigin([0,0,0])
    if img.dims()[2] <= 200:
        return img, True
    img_tensor = torch.tensor(img.toArray().transpose()).unsqueeze(0).unsqueeze(0)
    xpad = (max_shape[0]-img_tensor.shape[2])/2
    ypad = (max_shape[1]-img_tensor.shape[3])/2
    zpad = (max_shape[2]-img_tensor.shape[4])/2
    img_tensor = F.pad(img_tensor, (math.floor(zpad), math.ceil(zpad), math.floor(ypad), \
        math.ceil(ypad), math.floor(xpad), math.ceil(xpad)), "constant", 0)
    img_tensor = F.interpolate(img_tensor, [rz_size[0], rz_size[1], rz_size[2]], mode='trilinear')
    img_arr = img_tensor.squeeze().detach().numpy()
    img_new = sw.Image(img_arr.transpose().copy())
    return img_new, False

def worker(i):
    current_row = df.iloc[i]
    img = sw.Image(current_row['new_image'])
    img, if_skip = transform_volume(img)
    if if_skip:
        return
    bn = os.path.basename(current_row['new_image']).replace(".nii.gz", ".nrrd")
    img_path = os.path.join(save_path, "images", bn)
    if os.path.exists(img_path) == False:
        img.write(img_path)
    seg_dict = {}
    mesh_dict = {}
    all_empty = True
    for ca in anatomies:
        if pd.isna(current_row[ca]):
            seg_dict[ca] = ""
            mesh_dict[ca] = ""
            continue
        all_empty = False
        seg = sw.Image(current_row[ca])
        seg, _ = transform_volume(seg)
        seg.isolate()
        seg.binarize()
        seg_path = os.path.join(save_path, "segmentations", ca, bn)
        seg.write(seg_path)
        seg_dict[ca] = seg_path
        seg.antialias(30).computeDT(0).gaussianBlur(1.5)
        mesh_path = os.path.join(save_path, "meshes", ca, bn.replace(".nrrd", ".vtk"))
        seg_mesh = seg.toMesh(0)
        seg_mesh.fillHoles()
        seg_mesh.smooth()
        seg_mesh.remesh(10000, 1.0)
        seg_mesh.write(mesh_path)
        mesh_dict[ca] = mesh_path
    if all_empty:
        return
    sf = {
        'image': img_path,
        'segmentations': seg_dict,
        'meshes': mesh_dict
    }
    return sf

from joblib import Parallel, delayed

out = Parallel(n_jobs=16)(delayed(worker)(i) for i in tqdm(range(rows), desc="Iter: "))

out_new = [i for i in out if i is not None]
out_dict["files"] = out_new

with open('/home/sci/janmesh/DeepSSM/DeepSSM_pose/datasets/total_segmentator/selected_shapes.json', 'w') as fp:
    json.dump(out_dict, fp, indent=4)