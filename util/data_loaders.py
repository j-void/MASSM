import numpy as np
import torch
import os
import glob
import nrrd
import json

from torch.utils.data import Dataset
import util.net_utils as util
from torch.nn import functional as F
import torchvision.transforms as transforms
from skimage import segmentation as skimage_seg
from networks.extras import GaussianSmoothing
import torchio as tio
import shapeworks as sw
from pathlib import Path
from tqdm import tqdm
import random
import h5py
import pickle
from collections import defaultdict




def get_particles(model_path):
    """
    Load particles from a file and convert them into a list of float numpy arrays.

    Args:
        model_path (str): The path to the particles file.

    Returns:
        list: A list of float numpy arrays, where each array represents a particle.
    """
    f = open(model_path, "r")
    data = []
    for line in f.readlines():
        points = line.split()
        points = [float(i) for i in points]
        data.append(points)
    return data
    
                    
class MultiClassDataset_FA(Dataset):
    def __init__(self, cfg, split, unique_labels=None, labels_dict=None, reverse_ld=None):
        """
        Initialize the dataset.

        Args:
            cfg (config): The configuration object.
            split (str): The dataset split (train, val, test).
            unique_labels (list): The list of unique labels.
            labels_dict (dict): The dictionary mapping label names to indices.
            reverse_ld (dict): The dictionary mapping indices to label names.
        """
        super(MultiClassDataset_FA, self).__init__()
        # image dimensions
        self.img_dims = cfg.img_dims
        # configuration object
        self.cfg = cfg
        # dataset split (train, val, test)
        self.split = split
        # number of particles
        self.num_corr = 1024
        # config file path
        self.json_file = cfg.config_file
        # load the data and reference paths
        self.data_dict_list, self.reference_paths = self.load_paths()
        
        # list of unique labels
        self.unique_labels = unique_labels
        if self.unique_labels == None:
            self.unique_labels = list(self.reference_paths.keys())
            self.unique_labels.sort()
        
        self.labels_dict = labels_dict
        if self.labels_dict == None:
            self.labels_dict = {}
            for idx in range(len(self.unique_labels)):
                self.labels_dict[self.unique_labels[idx]] = idx
        
        # dictionary mapping indices to label names
        self.reverse_ld = reverse_ld
        if self.reverse_ld == None: 
            self.reverse_ld = dict(zip(self.labels_dict.values(), self.labels_dict.keys()))
            
        # Gaussian smoothing
        self.gs = GaussianSmoothing(len(self.unique_labels), 5, 1)
        # model scale
        self.model_scale = cfg.model_scale

        # resize shape of object detection outputs
        self.resize_shape = (np.array(self.img_dims)//cfg.model_scale).astype(int)

        self.transform = tio.Compose([
            tio.transforms.RandomGhosting(p=0.3),
            tio.transforms.RandomBiasField(p=0.3),
            tio.transforms.RandomBlur(p=0.3),
            tio.transforms.RandomSpike(p=0.3),
        ])
        
        self.normalize = tio.transforms.ZNormalization(p=1)
        
        self.aug_save_path = os.path.join(cfg.aug_save_path, self.split)
        if not os.path.exists(self.aug_save_path):
            os.makedirs(self.aug_save_path)
        
        # If we are in train mode and augmentation is enabled, 
        # then do data augmentation
        self.augment = False
        if cfg.augment and self.split == "train":
            self.augment = True

            # For each label, construct a sampler for the 
            # rigid transformations 
            self.RtsSamplers = {}
            for k in self.unique_labels:
                self.RtsSamplers[k] = da.contruct_Rts_sampler(self.data_dict_list, k)

    def __len__(self):
        return len(self.data_dict_list)

    def load_paths(self):
        """
        Load the paths to the data and reference paths from the config file.

        Returns:
            data_dict (list): A list of dictionaries containing the paths to the data.
            reference_paths (dict): A dictionary containing the reference paths.
        """
        reference_paths = {}
        data_dict = {}
        with open(self.json_file, 'r') as f:
            data = json.load(f)
        for fp in data['files']:
            if fp['type'] != self.split:
                continue
            data_dict[fp["name"]] = fp
        
        if self.split == "train":
            for key, value in data['references'].items():
                trd = {}
                v = value
                k = key
                fp = data_dict[v]
                trd["mesh_path"] = fp["meshes"][k]
                # local particles path
                trd["lp_path"] = fp["local_particles"][k]
                # world particles path
                trd["wp_path"] = fp["world_particles"][k]
                trd["local_particles"] = torch.tensor(np.array(get_particles(trd["lp_path"]))).unsqueeze(0).float().to(self.cfg.DEVICE)
                trd["world_particles"] = torch.tensor(np.array(get_particles(trd["wp_path"]))).unsqueeze(0).float().to(self.cfg.DEVICE)
                trd["segmentation"] = sw.Image(fp["segmentations"][k])
                # get center and radius of the particles
                cx, cy, cz, rx ,ry, rz = self.get_center_particles(trd["local_particles"][0])
                trd["radius"] = torch.tensor([rx, ry, rz]).float().to(self.cfg.DEVICE)
                trd["center"] = torch.tensor([cx, cy, cz]).float().to(self.cfg.DEVICE)
                image_sw = sw.Image(fp['image'])
                trd["image"] = torch.tensor(image_sw.toArray().transpose()).float().unsqueeze(0).unsqueeze(0).float().to(self.cfg.DEVICE)
                
                reference_paths[key] = trd

        return list(data_dict.values()), reference_paths

    
    def get_center(self, seg):
        """
        Get the center coordinates and radius based on the segmentation.

        Parameters
        ----------
        seg : torch.Tensor
            The segmentation tensor of shape (batch_size, channels, height, width, depth).

        Returns
        -------
        center : tuple of int
            The center coordinates of shape (3,).
        radius : tuple of float
            The radius of shape (3,).
        """
        # Get the indices of the segmentation
        crop_idx = torch.where(seg > 0.9)

        # Initialize the center and radius
        cx = cy = cz = max_dt = 0

        # If the segmentation is not empty
        if crop_idx[3].nelement() != 0:
            # Get the bounding box of the segmentation
            x_min = crop_idx[2].min()
            x_max = crop_idx[2].max()
            y_min = crop_idx[3].min()
            y_max = crop_idx[3].max()
            z_min = crop_idx[4].min()
            z_max = crop_idx[4].max()

            # Calculate the center coordinates
            cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

            # Calculate the radius
            max_dt = max(x_max - x_min, y_max - y_min, z_max - z_min)

        # Return the center coordinates and radius
        return  int(cx / self.model_scale), int(cy / self.model_scale), int(cz / self.model_scale), abs(x_max - x_min) / self.model_scale, abs(y_max - y_min) / self.model_scale, abs(z_max - z_min) / self.model_scale

    def get_center_particles(self, local_particles):
        """
        Get the center coordinates and radius of the particles.

        Parameters
        ----------
        local_particles : torch.Tensor
            The local particles tensor of shape (3,).

        Returns
        -------
        center : tuple of float
            The center coordinates of shape (3,).
        radius : tuple of float
            The radius of shape (3,).
        """
        # Get the bounding box of the particles
        x_min = torch.min(local_particles[:,0])
        x_max = torch.max(local_particles[:,0])
        y_min = torch.min(local_particles[:,1])
        y_max = torch.max(local_particles[:,1])
        z_min = torch.min(local_particles[:,2])
        z_max = torch.max(local_particles[:,2])

        # Calculate the center coordinates
        cx, cy, cz = (x_min+x_max)/2, (y_min+y_max)/2, (z_min+z_max)/2

        # Calculate the radius
        rx, ry, rz = abs(x_max-x_min), abs(y_max-y_min), abs(z_max-z_min)

        # Return the center coordinates and radius
        return  cx, cy, cz, rx, ry, rz

    def __getitem__(self, idx):

        current_data = self.data_dict_list[idx]
        center_heatmap_full = torch.zeros((len(self.unique_labels), self.resize_shape[0], self.resize_shape[1], self.resize_shape[2]))
        radius_map = torch.zeros((3, self.resize_shape[0], self.resize_shape[1], self.resize_shape[2]))
        segs_full = torch.zeros((len(self.unique_labels), self.img_dims[0], self.img_dims[1], self.img_dims[2]))
        detections = torch.zeros(len(self.unique_labels))
        points_world = torch.zeros(len(self.unique_labels), 1024, 3)
        points_local = torch.zeros(len(self.unique_labels), 1024, 3)
        center_true = torch.zeros(len(self.unique_labels), 3)
        # center_disp_map = torch.zeros((3, self.resize_shape[0], self.resize_shape[1], self.resize_shape[2]))

        # Get the image
        image_sw = sw.Image(current_data['image'])

        any_valid_aug = False
        
        if self.augment and random.uniform(0, 1) < 0.9:    

            # Get the available labels
            available_labels = [lb for lb in current_data["segmentations"].keys() if current_data["segmentations"][lb]]
            # Get the cumulative weights for the available labels
            # cum_weights = [self.weights[k] for k in available_labels]
            # Sample a random label from the available labels based on the weights
            label = random.choices(available_labels, k=1)[0]
            # Sample a random transform for the chosen label
            rts, _ = self.RtsSamplers[label].sample()
            R, t, c = rts[:3], rts[3:6], rts[-1]
            # Get the homogeneous similarity transform
            aug_transfrom = da.get_homogeneous_similar_transform(R=Rotation.from_rotvec(R).as_matrix(),t=t.reshape((-1, 1)), c=c)
            # Get the rigid transform for the chosen label
            rigid_transform = np.load(current_data["segmentations"][label].replace("segmentations", "properties").replace(".nrrd", ".npz"))["rigid_transform"]
            # Calculate the final transform by multiplying the rigid transform with the inverse of the similarity transform
            final_transform = rigid_transform @ np.linalg.inv(aug_transfrom)
        
            # Get the reference segmentation for the chosen label
            ref_seg = self.reference_paths[label]["segmentation"]

            # Iterate over the segmentations
            for lb, seg_path in current_data["segmentations"].items():
                if len(seg_path) > 0:
                    # Load the segmentation
                    # seg_sw = sw.Image(seg_path)
                    # Get the world particles
                    points_local_ = torch.tensor(np.array(get_particles(current_data["local_particles"][lb])))
                    
                    # Apply the transformation
                    final_transform_t = torch.tensor(final_transform).unsqueeze(0)
                    points_local_ = torch.cat([points_local_, torch.ones(points_local_.shape[0],1)], dim=1).unsqueeze(0)
                    points_local_ = torch.einsum('bij,bkj->bki', final_transform_t, points_local_)[0,:,:3]
                    
                    # Get the center coordinates and radius of the particles
                    cx, cy, cz, rx ,ry, rz = self.get_center_particles(points_local_)
                    radius = torch.tensor([rx, ry, rz])
                    center = torch.tensor([cx, cy, cz])/self.model_scale
                    center = center.long()
                    
                    # Check if the augmentation is valid
                    valid_augmentation = cx + rx <= self.img_dims[0] and cx - rx >= 0 and  cy + ry <= self.img_dims[1] and cy - ry >= 0 and cz + rz <= self.img_dims[2] and cz - rz >= 0
                    if  valid_augmentation:
                        # print('valid_augmentation')
                        # seg_sw.applyTransform(final_transform,
                        #         ref_seg.origin(),  ref_seg.dims(),
                        #         ref_seg.spacing(), ref_seg.coordsys(),
                        #         sw.InterpolationType.Linear, meshTransform=True)
                        
                        # Update the center heatmap, radius map, detections, points_world, points_local, and center_true
                        center_heatmap_full[self.labels_dict[lb], center[0], center[1], center[2]] = 1
                        radius_map[0, center[0], center[1], center[2]] = radius[0]
                        radius_map[1, center[0], center[1], center[2]] = radius[1]
                        radius_map[2, center[0], center[1], center[2]] = radius[2]
                        # segs_full[self.labels_dict[lb],:,:,:] = torch.tensor(seg_sw.toArray().transpose())
                        detections[self.labels_dict[lb]] = 1
                        points_world[self.labels_dict[lb]] = torch.tensor(np.array(get_particles(current_data["world_particles"][lb])))
                        points_local[self.labels_dict[lb]] = points_local_
                        center_true[self.labels_dict[lb]] = torch.tensor([cx, cy, cz])
                        # center_disp_map[:,center[0], center[1], center[2]] = center_true[self.labels_dict[lb]] - center*self.model_scale
                        any_valid_aug = True

        # If atleast one valid augmentation is found
        if any_valid_aug:
            # Apply the transformation to the image
            # print("any_valid_aug")
            image_sw.applyTransform(final_transform,
                ref_seg.origin(),  ref_seg.dims(),
                ref_seg.spacing(), ref_seg.coordsys(),
                sw.InterpolationType.Linear, meshTransform=True)            

            # Calculate the heatmap
            heatmap = self.gs(center_heatmap_full)
            heatmap = heatmap/torch.max(heatmap)
            
            image = torch.tensor(image_sw.toArray().transpose()).float().unsqueeze(0)
        
            points_world = points_world.float() 
            points_local = points_local.float()
            detections = detections.long()
            
        else: ## if no valid augmentation is found the do the same process without transformations
            center_heatmap_full = torch.zeros((len(self.unique_labels), self.resize_shape[0], self.resize_shape[1], self.resize_shape[2]))
            radius_map = torch.zeros((3, self.resize_shape[0], self.resize_shape[1], self.resize_shape[2]))
            segs_full = torch.zeros((len(self.unique_labels), self.img_dims[0], self.img_dims[1], self.img_dims[2]))
            detections = torch.zeros(len(self.unique_labels))
            points_world = torch.zeros(len(self.unique_labels), 1024, 3)
            points_local = torch.zeros(len(self.unique_labels), 1024, 3)
            center_true = torch.zeros(len(self.unique_labels), 3)
    
            image_sw = sw.Image(current_data['image'])
            for lb, seg_path in current_data["segmentations"].items():
                if len(seg_path) > 0:
                    seg_sw = sw.Image(seg_path)
                    points_local_ = torch.tensor(np.array(get_particles(current_data["local_particles"][lb])))
                    cx, cy, cz, rx ,ry, rz = self.get_center_particles(points_local_)
                    radius = torch.tensor([rx, ry, rz])
                    center = torch.tensor([cx, cy, cz])/self.model_scale
                    center = center.long()
                    center_heatmap_full[self.labels_dict[lb], center[0], center[1], center[2]] = 1
                    radius_map[0, center[0], center[1], center[2]] = radius[0]
                    radius_map[1, center[0], center[1], center[2]] = radius[1]
                    radius_map[2, center[0], center[1], center[2]] = radius[2]
                    # segs_full[self.labels_dict[lb],:,:,:] = torch.tensor(seg_sw.toArray().transpose())
                    detections[self.labels_dict[lb]] = 1
                    points_world[self.labels_dict[lb]] = torch.tensor(np.array(get_particles(current_data["world_particles"][lb])))
                    points_local[self.labels_dict[lb]] = points_local_
                    center_true[self.labels_dict[lb]] = torch.tensor([cx, cy, cz])
                    # center_disp_map[:,center[0], center[1], center[2]] = center_true[self.labels_dict[lb]] - center*self.model_scale
                    
                    heatmap = self.gs(center_heatmap_full)
                    heatmap = heatmap/torch.max(heatmap)
                    image = torch.tensor(image_sw.toArray().transpose()).float().unsqueeze(0)     
                    
                    points_world = points_world.float() 
                    points_local = points_local.float()
                    detections = detections.long()
        
        if self.augment:
            image = self.transform(image)
            
        image = self.normalize(image)
        
        return image, heatmap, radius_map, points_world, points_local, detections, center_true, segs_full

    
