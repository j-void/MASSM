import cgi
import numpy as np
import torch
from util.data_loaders import *
import config as cfg
from networks.model_fullts import MultiImage2Shape
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
import random
import nrrd
from collections import defaultdict
import math
import matplotlib.pyplot as plt
from util.net_utils import RMSEparticles, read_dist_mat
import subprocess
import numpy as np
import shutil
import vtk
import shapeworks as sw

from scipy.spatial.transform import Rotation


def makeDir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def RMSE(predicted, ground_truth):
    a = torch.sqrt(torch.mean((predicted - ground_truth)**2, dim=1))
    return torch.mean(a)


def run_inference(model, dataset, reference_paths, reverse_ld, load_name="best", split="train", save_dir="val_result"):

    """
    Run the inference on the given model and dataset.

    Parameters:
    model (nn.Module): The model to be used for inference.
    dataset (Dataset): The dataset to be used for inference.
    reference_paths (dict): A dictionary containing the paths to the reference objects.
    reverse_ld (dict): A dictionary containing the reverse label dict.
    load_name (str): The name of the model to be loaded. Default is "best".
    split (str): The dataset split to be used for inference. Default is "train".
    save_dir (str): The directory to save the results. Default is "val_result".

    Returns:
    errors_dict (dict): A dictionary containing the errors for each label.
    """
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    save_imdt = os.path.join(cfg.checkpoint_dir, save_dir, load_name, split)
    makeDir(save_imdt)
    
    wp_dir = os.path.join(save_imdt, "world", "particles")
    makeDir(wp_dir)
    wm_dir = os.path.join(save_imdt, "world", "mesh")
    makeDir(wm_dir)
    lp_dir = os.path.join(save_imdt, "local", "particles")
    makeDir(lp_dir)
    lm_dir = os.path.join(save_imdt, "local", "mesh")
    makeDir(lp_dir)

    for i in range(len(reverse_ld)):
        makeDir(os.path.join(wp_dir, reverse_ld[i]))
        makeDir(os.path.join(wm_dir, reverse_ld[i]))
        makeDir(os.path.join(lp_dir, reverse_ld[i]))
        makeDir(os.path.join(lm_dir, reverse_ld[i]))
    



    errors_dict = defaultdict(list)
    detection_dict = {"gt":[], "pred":[]}
    
    for idx, (image, heatmap, radius_map, points_world, points_local, detections, center_true, segs_full) in enumerate((tqdm(data_loader, desc=split+": "))):
        image = image.to(cfg.DEVICE)
        heatmap = heatmap.to(cfg.DEVICE)
        points_world = points_world.to(cfg.DEVICE)
        points_local = points_local.to(cfg.DEVICE)
        detections = detections.to(cfg.DEVICE)
        radius_map = radius_map.to(cfg.DEVICE)
        center_true = center_true.to(cfg.DEVICE)

        # Run inference
        centers, pred_centers, radius_full, pred_radius_full, pred_heatmap, pred_points_local, pred_points_world = model.val(image, heatmap, radius_map, detections)
        
        for i in range(radius_full.shape[1]):
            dt = int(detections[0, i].item())
            if dt==1:
                # Calculate the errors for the given anatomy
                errors_dict["radius_"+reverse_ld[i]].append(np.sqrt(model.MSE(radius_full[:,i,:].detach().cpu(), pred_radius_full[:,i,:].detach().cpu()).squeeze().item()))
                errors_dict["center_"+reverse_ld[i]].append(np.sqrt(model.MSE(center_true[:,i,:].detach().cpu(), pred_centers[:,i,:].detach().cpu()).squeeze().item()))
                errors_dict["local_"+reverse_ld[i]].append(RMSE(points_local[:,i,:,:].detach().cpu(), pred_points_local[:,i,:,:].detach().cpu()).squeeze().item())
                errors_dict["world_"+reverse_ld[i]].append(RMSE(points_world[:,i,:,:].detach().cpu(), pred_points_world[:,i,:,:].detach().cpu()).squeeze().item())
                
                # Load the template mesh and particles
                template_mesh = reference_paths[reverse_ld[i]]["mesh_path"]
                template_particles_w = reference_paths[reverse_ld[i]]["wp_path"]
                template_particles_l = reference_paths[reverse_ld[i]]["lp_path"]
        
                # Save the real and predicted particles
                np.savetxt(os.path.join(wp_dir, reverse_ld[i],"real_"+str(idx)+".particles"), points_world[:,i,:,:].detach().cpu().squeeze().numpy())
                np.savetxt(os.path.join(wp_dir, reverse_ld[i],"pred_"+str(idx)+".particles"), pred_points_world[:,i,:,:].detach().cpu().squeeze().numpy())
        
                # Warping to get the mesh from real and predicted particles
                execCommand = ["shapeworks", 
                    "warp-mesh", "--reference_mesh", template_mesh,
                    "--reference_points", template_particles_w,
                    "--target_points" ]
                execCommand.append(os.path.join(wp_dir, reverse_ld[i], "real_"+str(idx)+".particles"))
                execCommand.append(os.path.join(wp_dir, reverse_ld[i], "pred_"+str(idx)+".particles"))
                execCommand.append('--')
                subprocess.check_call(execCommand)
                shutil.move(os.path.join(wp_dir, reverse_ld[i], "real_"+str(idx)+".vtk"), os.path.join(wm_dir, reverse_ld[i], "real_"+str(idx)+".vtk"))
                shutil.move(os.path.join(wp_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"), os.path.join(wm_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"))
        
                # Calculate the distance between the real and predicted meshes
                real_mesh = sw.Mesh(os.path.join(wm_dir, reverse_ld[i], "real_"+str(idx)+".vtk"))
                pred_mesh = sw.Mesh(os.path.join(wm_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"))
        
                distance_values, cell_ids = real_mesh.distance(pred_mesh, method=sw.Mesh.DistanceMethod.PointToCell)
                real_mesh.setField(name='distance', array=distance_values, type=sw.Mesh.FieldType.Point).write(os.path.join(wm_dir,reverse_ld[i], "dist_"+str(idx)+".vtk"))
                average_distance = np.mean(distance_values)
                
                errors_dict["ws2s_"+reverse_ld[i]].append(average_distance)
        
                ### local particles - Do save for local

                np.savetxt(os.path.join(lp_dir, reverse_ld[i], "real_"+str(idx)+".particles"), points_local[:,i,:,:].detach().cpu().squeeze().numpy())
                np.savetxt(os.path.join(lp_dir, reverse_ld[i], "pred_"+str(idx)+".particles"), pred_points_local[:,i,:,:].detach().cpu().squeeze().numpy())
        
                execCommand = ["shapeworks", 
                    "warp-mesh", "--reference_mesh", template_mesh,
                    "--reference_points", template_particles_l,
                    "--target_points" ]
                execCommand.append(os.path.join(lp_dir, reverse_ld[i], "real_"+str(idx)+".particles"))
                execCommand.append(os.path.join(lp_dir, reverse_ld[i], "pred_"+str(idx)+".particles"))
                execCommand.append('--')
                subprocess.check_call(execCommand)
                shutil.move(os.path.join(lp_dir, reverse_ld[i], "real_"+str(idx)+".vtk"), os.path.join(lm_dir, reverse_ld[i], "real_"+str(idx)+".vtk"))
                shutil.move(os.path.join(lp_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"), os.path.join(lm_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"))
        
                real_mesh = sw.Mesh(os.path.join(lm_dir, reverse_ld[i], "real_"+str(idx)+".vtk"))
                pred_mesh = sw.Mesh(os.path.join(lm_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"))
                
                distance_values, cell_ids = real_mesh.distance(pred_mesh, method=sw.Mesh.DistanceMethod.PointToCell)
                real_mesh.setField(name='distance', array=distance_values, type=sw.Mesh.FieldType.Point).write(os.path.join(lm_dir, reverse_ld[i], "dist_"+str(idx)+".vtk"))
                average_distance = np.mean(distance_values)
                
                if split != "test":
                    os.remove(os.path.join(lp_dir, reverse_ld[i],"real_"+str(idx)+".particles"))
                    os.remove(os.path.join(lp_dir, reverse_ld[i],"pred_"+str(idx)+".particles"))
                    os.remove(os.path.join(lm_dir, reverse_ld[i], "real_"+str(idx)+".vtk"))
                    os.remove(os.path.join(lm_dir, reverse_ld[i], "pred_"+str(idx)+".vtk"))
                    os.remove(os.path.join(lm_dir,reverse_ld[i], "dist_"+str(idx)+".vtk"))
        
                errors_dict["ls2s_"+reverse_ld[i]].append(average_distance)
        
                
                
        detection_dict["gt"].append(torch.amax(heatmap, dim=(2, 3, 4)).squeeze().detach().cpu().numpy())
        detection_dict["pred"].append(torch.amax(pred_heatmap, dim=(2, 3, 4)).squeeze().detach().cpu().numpy())

    np.savez(os.path.join(save_imdt, "detections.npz"), gt=np.array(detection_dict["gt"]), pred=np.array(detection_dict["pred"]))
    for key, val in errors_dict.items():
        mean_ = np.mean(np.array(val))
        print(key, mean_)
    
    # Get the best and worst indices for each label and print the statistics
    for i in range(len(reverse_ld)):
        print("Stats for -", reverse_ld[i])
        errors_arr_w = np.array(errors_dict["world_"+reverse_ld[i]])
        idx_w = np.argsort(errors_arr_w)
        print(f"Best index for {split} world points=", idx_w[:3], errors_arr_w[idx_w[:3]])
        print(f"Worst index for {split} world points=", idx_w[-3:], errors_arr_w[idx_w[-3:]])
        errors_arr_wm = np.array(errors_dict["ws2s_"+reverse_ld[i]])
        idx_wm = np.argsort(errors_arr_wm)
        print(f"Best index for {split} surface distance(world)=", idx_wm[:3], errors_arr_wm[idx_wm[:3]])
        print(f"Worst index for {split} surface distance(world)=", idx_wm[-3:], errors_arr_wm[idx_wm[-3:]])
        print(f"Average surface-to-surface distance for {split}(world) = ", np.mean(errors_arr_w))


        errors_arr_l = np.array(errors_dict["local_"+reverse_ld[i]])
        idx_l = np.argsort(errors_arr_l)
        print(f"Best index for {split} local points=", idx_l[:3], errors_arr_l[idx_l[:3]])
        print(f"Worst index for {split} local points=", idx_l[-3:], errors_arr_l[idx_l[-3:]])
        errors_arr_lm = np.array(errors_dict["ls2s_"+reverse_ld[i]])
        idx_lm = np.argsort(errors_arr_lm)
        print(f"Best index for {split} surface distance(local)=", idx_lm[:3], errors_arr_lm[idx_lm[:3]])
        print(f"Worst index for {split} surface distance(local)=", idx_lm[-3:], errors_arr_lm[idx_lm[-3:]])
        print(f"Average surface-to-surface distance for {split}(local) = ", np.mean(errors_arr_lm))

        errors_arr_c = np.array(errors_dict["center_"+reverse_ld[i]])
        idx_c = np.argsort(errors_arr_c)
        print(f"Best index for {split} center=", idx_c[:3], errors_arr_c[idx_c[:3]])
        print(f"Worst index for {split} center=", idx_c[-3:], errors_arr_c[idx_c[-3:]])

        errors_arr_r = np.array(errors_dict["radius_"+reverse_ld[i]])
        idx_r = np.argsort(errors_arr_r)
        print(f"Best index for {split} radius=", idx_r[:3], errors_arr_r[idx_r[:3]])
        print(f"Worst index for {split} radius=", idx_r[-3:], errors_arr_r[idx_r[-3:]])
        

        np.savez(os.path.join(save_imdt, reverse_ld[i]+"_errors.npz"), l=errors_arr_l, lm=errors_arr_lm, \
            c=errors_arr_c, r=errors_arr_r, w=errors_arr_w, wm=errors_arr_wm)

    return errors_dict

def generate_stats(disp_dict, reference_paths, save_imdt, type_="local"):
    """
    Compute and save the mean and standard deviation of the displacements.

    Args:
        disp_dict (dict): Dictionary of displacements for each label.
        reference_paths (dict): Dictionary of paths to reference meshes and particles.
        save_imdt (str): Path to save the mean and standard deviation of displacements.
        type_ (str): Type of displacements (local or world).

    Returns:
        None
    """
    for key in disp_dict.keys():
        dists_array = np.array(disp_dict[key])
        mean_dist = np.mean(dists_array, axis=0)
        std_dist = np.std(dists_array, axis=0)
        
        if type_=="local":
            particles_path = reference_paths[key]["lp_path"]
        else:
            particles_path = reference_paths[key]["wp_path"]
            mesh_path = reference_paths[key]["mesh_path"].replace("meshes", "groomed_meshes")
            
        # Warp the mesh to the particles
        execCommand = ["shapeworks", 
            "warp-mesh", "--reference_mesh", mesh_path,
            "--reference_points", particles_path,
            "--target_points" ]
        execCommand.append(particles_path)
        execCommand.append('--')
        subprocess.check_call(execCommand)
        
        save_path = os.path.join(save_imdt, type_, key+"_data_stat.vtk")
        shutil.move(particles_path.replace(".particles", ".vtk"), save_path)
        
        # Write the mean and standard deviation of the displacements to the mesh
        tmesh = sw.Mesh(save_path)
        tmesh.setField(name='mean', array=mean_dist, type=sw.Mesh.FieldType.Point)\
            .setField(name='std', array=std_dist, type=sw.Mesh.FieldType.Point)\
            .write(save_path)

def plot_errors2(labels, errors, save_path):
    """
    Plot boxplots of errors for each label and save the plot to a file.

    Args:
        labels (list): List of label names.
        errors (list): List of lists of errors corresponding to each label.
        save_path (str): Path to save the plot to.

    Returns:
        CTEs (list): List of mean errors for each label.
    """
    assert len(labels) == len(errors)

    # Calculate mean and standard deviation of errors for each label
    CTEs = [np.mean(np.array(e)) for e in errors]
    stds = [np.std(np.array(e)) for e in errors]

    # Create the boxplot
    x_pos = np.arange(len(labels))

    fig, ax = plt.subplots()
    ax.boxplot(errors, boxprops=dict(color='red'), labels=labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return CTEs


if __name__ == "__main__":
    print("--------- Loading data ---------")
    """
    Load the data and run the inference
    """
    train_dataset = MultiClassDataset_FA(cfg, split="train")
    val_dataset =  MultiClassDataset_FA(cfg, split="val", unique_labels=train_dataset.unique_labels, \
        labels_dict=train_dataset.labels_dict, reverse_ld=train_dataset.reverse_ld)
    test_dataset =  MultiClassDataset_FA(cfg, split="test", unique_labels=train_dataset.unique_labels, \
        labels_dict=train_dataset.labels_dict, reverse_ld=train_dataset.reverse_ld)

    """
    Initialize the networks
    """
    print("--------- Initializing Networks ---------")
    model = MultiImage2Shape(in_channels=1, cfg=cfg, train_data=train_dataset, f_maps=8, num_levels=5)

    """
    Load the best checkpoint
    """
    load_name = "best"
    model.load_pretrained(load_name)
    

    reference_paths = train_dataset.reference_paths
    reverse_ld = train_dataset.reverse_ld

    save_dir="val_result"

    """
    Run the inference on train, validation and test data
    """
    with torch.no_grad():
        model.eval()
        # train_errors = run_inference(model,  train_dataset, reference_paths, reverse_ld, load_name="best", split="train", save_dir=save_dir)
        # val_errors = run_inference(model,  val_dataset, reference_paths, reverse_ld, load_name="best", split="val", save_dir=save_dir)
        test_errors = run_inference(model,  test_dataset, reference_paths, reverse_ld, load_name=load_name, split="test", save_dir=save_dir)


            

    print("--------- Done ---------")
                
                                
            
    
    
    
    

    