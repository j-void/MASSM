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
import json

"""
Run shapeworks in incremental mode
=================================

This script is used to run ShapeWorks in incremental mode. It is used to
generate the particle distributions for the training shapes. 

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
    
    
    import subprocess

    train_groomed_files = []
    for sf in file_list:
        if sf['type'] != "train":
            continue
        shape_filename = sf["meshes"][shape_label]
        if len(shape_filename) == 0:
            continue
        train_groomed_files.append(shape_filename.replace("meshes", "groomed"))


    sorting_type = "median"

    if sorting_type == "random":
        print("Randomly sorting.")
        sorted_indices = np.arange(len(train_groomed_files))
        np.random.shuffle(sorted_indices)
    else:
        # Load meshes
        meshes = []
        for mesh_file in train_groomed_files:
            meshes.append(sw.Mesh(mesh_file))
        # Get distance matrix
        print("Finding surface-to-surface distances for sorting...")
        distances = np.zeros((len(meshes), len(meshes)))
        for i in range(len(meshes)):
            for j in range(len(meshes)):
                if i != j:
                    distances[i][j] = np.mean(meshes[i].distance(meshes[j])[0])
        median_index = np.argmin(
            np.sum(distances, axis=0) + np.sum(distances, axis=1))
        # Sort
        if sorting_type == "median":
            print("Sorting using median.")
            sorted_indices = np.argsort(
                distances[median_index] + distances[:, median_index])
        elif sorting_type == "distribution":
            print("Sorting using distribution.")
            sorted_indices = [median_index]
            while len(sorted_indices) < len(train_groomed_files):
                dists = np.sum(distances[sorted_indices], axis=0) + \
                    np.sum(distances[:, sorted_indices], axis=1)
                next_ind = [i for i in np.argsort(
                    dists) if i not in sorted_indices][0]
                sorted_indices.append(next_ind)
        else:
            print("Error: Sorting type unrecognized.")
    sorted_mesh_files = np.array(train_groomed_files)[sorted_indices]

    initial_model_size = 20
    incremental_batch_size = 15
    
    initial_shapes = sorted_mesh_files[:initial_model_size]
    remaining = range(initial_model_size, len(train_groomed_files))
    incremental_batches = [sorted_mesh_files[i:i + incremental_batch_size]
                           for i in range(initial_model_size, len(sorted_mesh_files), incremental_batch_size)]
    batches = [initial_shapes]+incremental_batches
    
    ## Run base optimization
    # Set subjects
    subjects = []
    number_domains = 1
    for i in range(len(batches[0])):
        subject = sw.Subject()
        subject.set_number_of_domains(number_domains)
        rel_groom_files = [os.path.join(os.path.abspath(os.getcwd()), batches[0][i])]
        subject.set_groomed_filenames(rel_groom_files)
        subjects.append(subject)

    # Set project
    project = sw.Project()
    project.set_subjects(subjects)
    parameters = sw.Parameters()

    # # Create a dictionary for all the parameters required by optimization
    parameter_dictionary = {
        "number_of_particles" : 1024,
        "use_normals": 1,
        "checkpointing_interval" : 200,
        "keep_checkpoints" : 0,
        "iterations_per_split" : 1000,
        "optimization_iterations" : 1000,
        "starting_regularization" : 10,
        "ending_regularization" : 1,
        "relative_weighting" : 1,
        "initial_relative_weighting" : 0.05,
        "procrustes_interval" : 0,
        "procrustes_scaling" : 0,
        "save_init_splits" : 0,
        "geodesics_enabled": 1,
        "verbosity" : 1,
        "procrustes" : 0
    }

    # Add param dictionary to spreadsheet
    for key in parameter_dictionary:
        parameters.set(key,sw.Variant([parameter_dictionary[key]]))
    project.set_parameters("optimize",parameters)
    spreadsheet_file_name = shape_label
    spreadsheet_file = os.path.join(project_location, spreadsheet_file_name+".xlsx")
    project.save(spreadsheet_file)

    # Run optimization
    optimize_cmd = ('shapeworks optimize --name ' + spreadsheet_file).split()
    subprocess.check_call(optimize_cmd)

    ## Run incremental optimization

    parameter_dictionary["use_landmarks"] = 1 				
    parameter_dictionary["iterations_per_split"] = 0
    parameter_dictionary["optimization_iterations"] = 1000
    parameter_dictionary["multiscale"] = 0 	

        # Run optimization on each batch
    for batch_index in range(1, len(batches)):
        print("Running incremental optimization " +
              str(batch_index) + " out of " + str(len(batches)-1))
        # Update meanshape
        sw.utils.findMeanShape(os.path.join(project_location, spreadsheet_file_name+"_particles"))
        mean_shape_path = os.path.join(project_location, spreadsheet_file_name+"_particles", 'meanshape_local.particles')
        # Set subjects
        subjects = []
        # Add current shape model (e.g. all previous batches)
        for i in range(0, batch_index):
            for j in range(len(batches[i])):
                subject = sw.Subject()
                subject.set_number_of_domains(1)
                rel_groom_files = [os.path.join(os.path.abspath(os.getcwd()), batches[i][j])]
                subject.set_groomed_filenames(rel_groom_files)
                particle_file = mean_shape_path.replace("meanshape", os.path.basename(batches[i][j]).split(".")[0]) 
                rel_particle_file = [os.path.join(os.path.abspath(os.getcwd()), particle_file)]
                subject.set_landmarks_filenames(rel_particle_file)
                subjects.append(subject)
        # Add new shapes in current batch - intialize with meanshape
        for j in range(len(batches[batch_index])):
            subject = sw.Subject()
            subject.set_number_of_domains(1)
            rel_groom_files = [os.path.join(os.path.abspath(os.getcwd()), batches[batch_index][j])]
            subject.set_groomed_filenames(rel_groom_files)
            rel_particle_file = [os.path.join(os.path.abspath(os.getcwd()), mean_shape_path)]
            subject.set_landmarks_filenames(rel_particle_file)
            subjects.append(subject)
        # Set project
        project = sw.Project()
        project.set_subjects(subjects)
        parameters = sw.Parameters()

        # Add param dictionary to spreadsheet
        for key in parameter_dictionary:
            parameters.set(key, sw.Variant([parameter_dictionary[key]]))
        project.set_parameters("optimize", parameters)
        spreadsheet_file = os.path.join(project_location, spreadsheet_file_name+".xlsx")
        project.save(spreadsheet_file)

        # Run optimization
        optimize_cmd = ('shapeworks optimize --progress --name ' + spreadsheet_file).split()
        subprocess.check_call(optimize_cmd)


    
    

    
    
