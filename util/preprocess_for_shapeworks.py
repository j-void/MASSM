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
from ShapeCohortGen.CohortGenUtils import *
from scipy.spatial.transform import Rotation as R
import matplotlib.tri as mtri
import trimesh
import json

"""
Grooming stage of shapeworks before optimization
===============================================

"""

def make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path

if __name__ == "__main__":
    
    with open("/home/sci/janmesh/DeepSSM/DeepSSM_pose/datasets/total_segmentator/selected_shapes.json", 'r') as f:
        data = json.load(f) 
    
    file_list = data['files']

    shape_label = "heart_atrium_left"
    main_path = "/home/sci/janmesh/DeepSSM/Janmesh_Data"

    properties_dir = make_dir(os.path.join(main_path, "properties", shape_label))
    ssm_output_dir = make_dir(os.path.join(main_path, "ssm_output", shape_label))
    groom_dir = make_dir(os.path.join(main_path, "groomed", shape_label)) 
    ref_dir = make_dir(os.path.join(main_path, "references"))
    
    seg_list = []
    seg_names = []
    seg_paths = []

    for sf in file_list:
        if sf['type'] != "train":
            continue
        shape_filename = sf["segmentations"][shape_label]
        if len(shape_filename) == 0:
            continue
        print('Loading: ' + shape_filename)
        shape_name = os.path.basename(shape_filename).replace('.nrrd', '')
        shape_seg = sw.Image(shape_filename)
        iso_value = 0.5  
        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.crop(bounding_box)
        
        seg_names.append(shape_name)
        seg_list.append(shape_seg)
        seg_paths.append(shape_filename)

        antialias_iterations = 30   # number of iterations for antialiasing
        shape_seg.antialias(antialias_iterations).binarize()
        pad_size = 10    # number of voxels to pad for each dimension
        pad_value = 0   # the constant value used to pad the segmentations
        shape_seg.pad(pad_size, pad_value)
    
    ref_index = sw.find_reference_image_index(seg_list)
    ref_name = seg_names[ref_index]
    ref_seg = seg_list[ref_index]

    with open(os.path.join(ref_dir, shape_label+".txt"), 'w') as f:
        out_ = "Reference found: " + str(ref_name)
        f.write(out_)


    """
    Now we can loop over all of the segmentations again to find the rigid
    alignment transform and compute a distance transform
    """
    antialias_iterations = 30
    error_names = []
    for seg, name in zip(seg_list, seg_names):
        print('Finding alignment transform from ' + name + ' to ' + ref_name)
        antialias_iterations = 30   # number of iterations for antialiasing
        iso_spacing = [1, 1, 1]
        iso_value = 0.1
        seg.antialias(antialias_iterations)
        rigidTransform = seg.createRigidRegistrationTransform(ref_seg, iso_value, 200)
        seg.applyTransform(rigidTransform,
                                ref_seg.origin(),  ref_seg.dims(),
                                ref_seg.spacing(), ref_seg.coordsys(),
                                sw.InterpolationType.Linear)
        seg.binarize()
        bounding_box = sw.ImageUtils.boundingBox([seg], 0.5).pad(2)
        # print(bounding_box, seg, name)
        seg.crop(bounding_box).pad(10, 0)
        seg.antialias(30).computeDT(0).gaussianBlur(1.5)
        seg_mesh = seg.toMesh(0)
        seg_mesh.fillHoles()
        seg_mesh.smooth()
        seg_mesh.remesh(10000, 1.0)
        seg_mesh.write(os.path.join(groom_dir, name+".vtk"))
        rigid_transform = sw.utils.getVTKtransform(rigidTransform)

        pName = os.path.join(properties_dir, name+".npz")
        np.savez(pName, rigid_transform=np.array(rigid_transform))



    print('------------ Processing for test ------------')


    test_seg_list = []
    test_seg_names = []
    


    for sf in file_list:
        if sf['type'] != "test":
            continue
        shape_filename = sf["segmentations"][shape_label]
        if len(shape_filename) == 0:
            continue
        print('Loading: ' + shape_filename)
        shape_name = os.path.basename(shape_filename).replace('.nrrd', '')
        shape_seg = sw.Image(shape_filename)
        iso_value = 0.5  
        bounding_box = sw.ImageUtils.boundingBox([shape_seg], iso_value).pad(2)
        shape_seg.crop(bounding_box)
        
        test_seg_names.append(shape_name)
        test_seg_list.append(shape_seg)

        antialias_iterations = 30   # number of iterations for antialiasing
        shape_seg.antialias(antialias_iterations).binarize()
        pad_size = 10    # number of voxels to pad for each dimension
        pad_value = 0   # the constant value used to pad the segmentations
        shape_seg.pad(pad_size, pad_value)
        
        
    

    
    antialias_iterations = 30

    for seg, name in zip(test_seg_list, test_seg_names):
        print('Finding alignment transform from ' + name + ' to ' + ref_name)
        antialias_iterations = 30   # number of iterations for antialiasing
        iso_spacing = [1, 1, 1]
        iso_value = 0.1
        seg.antialias(antialias_iterations)
        rigidTransform = seg.createRigidRegistrationTransform(ref_seg, iso_value, 200)
        
        seg.applyTransform(rigidTransform,
                                ref_seg.origin(),  ref_seg.dims(),
                                ref_seg.spacing(), ref_seg.coordsys(),
                                sw.InterpolationType.Linear)
        seg.binarize()
        bounding_box = sw.ImageUtils.boundingBox([seg], 0.5).pad(2)
        seg.crop(bounding_box).pad(10, 0)
        seg.antialias(30).computeDT(0).gaussianBlur(1.5)
        seg_mesh = seg.toMesh(0)
        seg_mesh.fillHoles()
        seg_mesh.smooth()
        seg_mesh.remesh(10000, 1.0)
        seg_mesh.write(os.path.join(groom_dir, name+".vtk"))

        rigid_transform = sw.utils.getVTKtransform(rigidTransform)

        pName = os.path.join(properties_dir, name+".npz")
        np.savez(pName, rigid_transform=np.array(rigid_transform))



    print('------------ Done ------------')
    print(error_names)