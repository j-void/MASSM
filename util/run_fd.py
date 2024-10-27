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
from numpy.linalg import inv
import json

"""
Run shapeworks using fixed domain
=================================
For test shapes 

"""

def get_particles(model_path):
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return(data)

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
    project_location = make_dir(os.path.join(main_path, "ssm_output", shape_label))
    
    train_groomed_files = []
    train_rigid_transforms_files = []
    test_groomed_files = []
    test_rigid_transforms_files = []
    
    for sf in file_list:
        if sf['type'] == "train" or sf['type'] == "val":
            shape_filename = sf["meshes"][shape_label]
            if len(shape_filename) == 0:
                continue
            train_groomed_files.append(shape_filename.replace("meshes", "groomed"))
            train_rigid_transforms_files.append(shape_filename.replace("meshes", "properties").replace(".vtk", ".npz"))
        elif sf['type'] == "test":
            shape_filename = sf["meshes"][shape_label]
            if len(shape_filename) == 0:
                continue
            test_groomed_files.append(shape_filename.replace("meshes", "groomed"))
            test_rigid_transforms_files.append(shape_filename.replace("meshes", "properties").replace(".vtk", ".npz"))
    
        
    import subprocess



    parameter_dictionary = {
        "number_of_particles": 1024,
        "use_normals": 0,
        "checkpointing_interval" : 200,
        "keep_checkpoints" : 0,
        "iterations_per_split": 100,
        "optimization_iterations" : 1000,
        "starting_regularization" : 10,
        "ending_regularization" : 1,
        "relative_weighting" : 1,
        "initial_relative_weighting" : 0.05,
        "procrustes_interval": 0,
        "procrustes_scaling": 0,
        "save_init_splits": 0,
        "verbosity": 1,
        "use_landmarks": 1,
        "use_fixed_subjects": 1,
        "narrow_band": 1e20,
        "fixed_subjects_column": "fixed",
        "fixed_subjects_choice": "yes"
    }


    spreadsheet_file_name = shape_label

    model_dir = os.path.join(project_location, spreadsheet_file_name+"_particles")

    sw.utils.findMeanShape(model_dir)
    mean_shape_path = os.path.join(model_dir, 'meanshape_local.particles')

    subjects = []
    number_domains = 1

    for i in range(len(train_groomed_files)):
        subject = sw.Subject()
        rel_groom_files = [os.path.join(os.path.abspath(os.getcwd()), train_groomed_files[i])] 
        subject.set_groomed_filenames(rel_groom_files)
        name = os.path.basename(train_groomed_files[i]).replace(".vtk", "_world.particles")
        rel_particle_files = [os.path.join(os.path.abspath(os.getcwd()), os.path.join(model_dir, name))] 
        subject.set_landmarks_filenames(rel_particle_files)
        subject.set_extra_values({"fixed": "yes"})
        subjects.append(subject)



    for i in range(len(test_groomed_files)):
        subject = sw.Subject()
        rel_groom_files = [os.path.join(os.path.abspath(os.getcwd()), test_groomed_files[i])] 
        rel_particle_files = [os.path.join(os.path.abspath(os.getcwd()), mean_shape_path)]
        subject.set_groomed_filenames(rel_groom_files)
        subject.set_landmarks_filenames(rel_particle_files)
        subject.set_extra_values({"fixed": "no"})
        subjects.append(subject)

    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()



    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key, sw.Variant(parameter_dictionary[key]))
        
    project.set_parameters("optimize", parameters)
    spreadsheet_file = os.path.join(project_location, spreadsheet_file_name+".xlsx")
    project.save(spreadsheet_file)

    # print("here", len(subjects))
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)

    
    # # ## Analyze in studio
    # analyze_cmd = ('ShapeWorksStudio ' + spreadsheet_file).split()
    # subprocess.check_call(analyze_cmd)


    print('Transforming particles')    


    for i in range(len(test_rigid_transforms_files)):
        fname = os.path.basename(test_rigid_transforms_files[i])
        wp_path = os.path.join(model_dir, fname.replace(".npz", "_world.particles"))
        mp = np.array(get_particles(wp_path))
        mp = np.append(mp, np.ones((mp.shape[0],1)), axis=1)
        rigid_transform = np.load(test_rigid_transforms_files[i])["rigid_transform"]
        lp = np.einsum('ij,kj->ki',inv(rigid_transform), mp)[:,:3]
        np.savetxt(wp_path.replace("world", "local"), lp)

    for i in range(len(train_rigid_transforms_files)):
        fname = os.path.basename(train_rigid_transforms_files[i])
        wp_path = os.path.join(model_dir, fname.replace(".npz", "_world.particles"))
        mp = np.array(get_particles(wp_path))
        mp = np.append(mp, np.ones((mp.shape[0],1)), axis=1)
        rigid_transform = np.load(train_rigid_transforms_files[i])["rigid_transform"]
        lp = np.einsum('ij,kj->ki',inv(rigid_transform), mp)[:,:3]
        np.savetxt(wp_path.replace("world", "local"), lp)



    
