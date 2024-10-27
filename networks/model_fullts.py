from .centernet import CenterNet, Flatten
import torch
import math
import torch.nn.functional as F
from torch import nn as nn
import numpy as np
import os
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
from tqdm import tqdm
import nrrd
from .vae import LocalModule
import random

def weight_init(module, initf):
	def foo(m):
		classname = m.__class__.__name__.lower()
		if isinstance(m, module):
			initf(m.weight)
	return foo

class MultiImage2Shape(nn.Module):
    def __init__(self, in_channels, cfg, train_data):
        super(MultiImage2Shape, self).__init__()
        """
        Initialize the networks
        
        Parameters
        ----------
        in_channels : int
            The number of input channels.
        cfg : config
            The configuration object.
        train_data : dataset
            The training dataset.
        """
        self.device = cfg.DEVICE
        self.save_dir = os.path.join(cfg.checkpoint_dir, "save")
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
        ## Initialize the networks
        self.num_corr = train_data.num_corr
        self.train_data = train_data
        
        # positional encoding for the class embedding
        self.class_embedding = self.positional_encoding(len(train_data.unique_labels)+1, 32).squeeze().to(self.device)
        
        # Anatomy detection network
        self.object_net = CenterNet(len(train_data.unique_labels))
        self.object_net.to(self.device)
        
        # Initialize the weights of the object network
        self.object_net.apply(weight_init(module=nn.Conv3d, initf=nn.init.kaiming_normal_))
        self.object_net.apply(weight_init(module=nn.ConvTranspose3d, initf=nn.init.kaiming_normal_))	
        self.object_net.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_normal_))
        
        # Local displacement prediction network
        self.lnet = LocalModule(num_latent=cfg.wp_latent, num_corr=1024)
        self.lnet.to(self.device)
        	
        # Initialize the weights of the local network
        self.lnet.apply(weight_init(module=nn.Linear, initf=nn.init.kaiming_normal_))
        
        # Define the parameters for the optimizer
        train_params = [
            {'params': list(self.object_net.parameters()) + list(self.lnet.parameters()), 'lr': 5e-5}
        ]
        
        # Initialize the optimizer and scheduler
        self.optimizer = torch.optim.Adam(train_params, betas=(0.9, 0.999))
        self.scheduler =  StepLR(self.optimizer, step_size=20, gamma=0.9) #CosineAnnealingLR(self.optimizer, T_max=cfg.num_epochs, eta_min=1e-6) #ReduceLROnPlateau(self.optimizer, factor=0.5, min_lr=1e-8, patience=3, verbose=True)
        self.optimizer.zero_grad()
        
        
        # define loss functions
        self.criterion_radius = nn.MSELoss(reduction="sum")    
        
    def positional_encoding(self, n, d_model):
        position = torch.arange(0, n).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        encoding = torch.zeros(n, d_model)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        return encoding
    
    def MSE(self, predicted, ground_truth):
        """
        Computes the mean squared error between the predicted and ground truth values.
        """
        return torch.mean((predicted - ground_truth)**2)
        
    def focal_loss(self, pred, ground_truth, a = 0, c = 0):
        '''
        Emplioys general formulation of focal loss, the pred and groundtruth can either be PCA/latent space or the correspondences directly.
        a = 0 will results in standard MSE loss
        '''
        l = torch.abs(pred - ground_truth)
        out = l**2 / (1 + torch.exp(a*(c - l)))
        return torch.mean(out)

    def focal_rel_loss(self, pred, ground_truth, mean, a = 0, c = 0):
        return self.focal_loss(pred, ground_truth, a, c) / self.focal_loss(mean, ground_truth, a, c)


    def neg_loss(self, pred, gt, detections):
        """
        Focal loss for heatmap.
        Modified from https://github.com/sidml/Understanding-Centernet/blob/master/loss.py
        """
        # Get the positive and negative indices
        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        # Compute the loss for the positive and negative indices
        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, 3) * pos_inds
        neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, 3) * neg_weights * neg_inds

        # Compute the total loss
        num_pos  = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # If there are no positive indices, set the loss to the negative loss
        if num_pos == 0:
            loss = loss - neg_loss
        else:
            # Otherwise, compute the average of the positive and negative losses
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss
        
    def masked_l1_loss(self, pred, gt):
        """
        Computes the L1 loss between the predicted correspondences and the ground truth
        correspondences, but only for the points where the ground truth is not zero.
        """
        # Create a mask for the points where the ground truth is not zero
        mask_ = gt.detach().clone()
        mask_[mask_<1] = 0
        mask_[mask_>=1] = 4
        # Compute the L1 loss between the predicted and ground truth correspondences
        # only for the points where the ground truth is not zero
        return self.criterion_radius(pred*mask_, gt*mask_)
    
    def rigid_registration(self, source_points, target_points):
        """
        Computes the rigid transformation between two sets of points.

        Args:
            source_points: The source points to be registered.
            target_points: The target points to be registered to.

        Returns:
            The transformed source points.
        """

        # Center the points around their centroids
        centroid_source = torch.mean(source_points, dim=0, keepdim=True)
        centroid_target = torch.mean(target_points, dim=0, keepdim=True)
        centered_source = source_points - centroid_source
        centered_target = target_points - centroid_target

        # Compute the covariance matrix and its SVD
        covariance_matrix = torch.mm(centered_source.T, centered_target)
        U, S, Vt = torch.linalg.svd(covariance_matrix, full_matrices=False)

        # Compute the optimal rotation matrix
        rotation_matrix = torch.mm(Vt.T, U.T)

        # Ensure a proper rotation by adjusting for reflection case and making sure gradient computation is consistent
        # This is a hacky fix, but it works
        if torch.det(rotation_matrix) < 0:
            Vt_adjusted = Vt.clone()
            Vt_adjusted[-1, :] = Vt_adjusted[-1, :] * -1
            rotation_matrix = torch.mm(Vt_adjusted.T, U.T)

        # Compute the optimal translation vector
        translation_vector = centroid_target - torch.mm(rotation_matrix, centroid_source.T).T

        # Apply the transformation to the source points
        transformed_source = torch.mm(source_points, rotation_matrix.T) + translation_vector

        return transformed_source

    def forward(self, image, heatmap, radius_map, points_world, points_local, detections, center_true, schedule_prob, segs_full):
        """
        Forward pass of the model.

        Args:
            image: The input image.
            heatmap: The ground truth heatmap.
            radius_map: The ground truth radius map.
            points_world: The ground truth world particle coordinates.
            points_local: The ground truth local particle coordinates.
            detections: The ground truth detection masks.
            center_true: The ground truth center coordinates.
            schedule_prob: The probability of sampling from the ground truth or the predicted heatmap.
            segs_full: The full segmentation masks.

        Returns:
            The loss and the outputs of the model.
        """
        
        # get the predicted heatmaps, radius maps, and center displacement maps
        pred_heatmap, pred_radius_map, features, center_disp_map  = self.object_net(image)
        
        # Compute the loss for the heatmap
        loss_heatmap = self.neg_loss(pred_heatmap, heatmap, detections)
        
        # Compute the loss for the radius map
        loss_radius = self.masked_l1_loss(pred_radius_map, radius_map)
        
        # Initialize the tensors for storing the outputs
        centers = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
        pred_centers = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3).to(self.device)
        radius_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
        pred_radius_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
        crop_list = []
        
        loss_local = 0
        loss_world = 0
        
        pred_points_world_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 1024, 3).to(self.device)
        pred_points_local_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 1024, 3).to(self.device)
        
        # Iterate over the detections
        b = 0
        for d in range(detections.shape[1]):
            dt = int(detections[b, d].item())
            if dt==1:
                # Extract the center from the ground truth heatmap
                center, pred_center = self.extract_centers(heatmap[b,d,:,:,:], pred_heatmap[b,d,:,:,:])
                # Extract the radius from the ground truth and predicted radius maps
                radius = self.extract_radius(center, radius_map)
                pred_radius = self.extract_radius(pred_center, pred_radius_map)
                
                # Compute the center displacement
                cdisp = self.extract_center_disp(pred_center, center_disp_map)
                # Predicted center = center form predicted heatmap * scale factor + center displacement
                pred_centers[b:b+1,d, :] = pred_center*4 + cdisp  
                
                radius_full[b:b+1,d, :] = radius.detach().cpu().squeeze()
                pred_radius_full[b:b+1,d, :] = pred_radius.detach().cpu().squeeze()
                
                # crop_list.append(self.extract_crop(segs_full[b, d, :,:,:], pred_center, pred_radius).detach())
                
                # Rereference the center and local particles
                ref_lp = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["local_particles"]
                ref_center = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["center"] 
                
                # Sample from the ground truth or the predicted heatmap based on the schedule probability
                if random.uniform(0, 1) < schedule_prob[0]:
                    # Do RoI alignment
                    roi = self.extract_fpn_features(features, center_true[b,d,:], radius)
                    roi = F.dropout(roi, p=0.2, training=True)
                    with torch.no_grad():
                        ref_lp_t = (center_true[b,d,:] - ref_center) + ref_lp
                    # one hot for the anatomy label
                    one_hot = torch.nn.functional.one_hot(torch.tensor(d), detections.shape[1]).unsqueeze(0).to(self.device) 
                    # predict the local particles displacement
                    l_disp_pred = self.lnet(roi, one_hot, torch.max(radius).item())
                else:
                    roi = self.extract_fpn_features(features, pred_centers[b,d, :], pred_radius)
                    roi = F.dropout(roi, p=0.2, training=True)
                    with torch.no_grad():
                        ref_lp_t =  (pred_centers[b,d, :] - ref_center) + ref_lp 
                    one_hot = torch.nn.functional.one_hot(torch.tensor(d), detections.shape[1]).unsqueeze(0).to(self.device)
                    l_disp_pred = self.lnet(roi, one_hot, torch.max(pred_radius).item())
    
                
                # get the local particles
                pred_points_local = l_disp_pred + ref_lp_t
                pred_points_local_full[b:b+1,d, :,:] = pred_points_local

                # Template world particles
                ref_wp = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["world_particles"].to(self.device)
                    
                # Sample from the ground truth or the predicted local particles based on the schedule probability
                if random.uniform(0, 1) < schedule_prob[1]:
                    pred_points_world = self.rigid_registration(source_points=points_local[b,d, :,:], target_points=ref_wp[0])
                else:
                    pred_points_world = self.rigid_registration(source_points=pred_points_local_full[b,d, :,:], target_points=ref_wp[0])

                pred_points_world_full[b:b+1,d, :,:] = pred_points_world


        # Compute the loss for local particles, center displacement and  world particles
        loss_local = self.focal_loss(points_local, pred_points_local_full, 10, 0.2)
        loss_cdisp = self.focal_loss(center_true, pred_centers, 10, 0.2)
        loss_world = self.focal_loss(points_world, pred_points_world_full, 10, 0.2)
             

        return [loss_heatmap, loss_radius, loss_local, loss_cdisp, loss_world], [centers, pred_centers, radius_full, pred_radius_full, \
            pred_heatmap, crop_list, pred_points_local_full, pred_points_world_full]

    

    def val(self, image, heatmap, radius_map, detections):
        """
        Inference
        """
        with torch.no_grad():

            pred_heatmap, pred_radius_map, features, center_disp  = self.object_net(image)
            
            centers = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
            pred_centers = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3).to(self.device)
            radius_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
            pred_radius_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 3)
            crop_list = []

            pred_points_world_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 1024, 3).to(self.device)
            pred_points_local_full = torch.zeros(image.shape[0], len(self.train_data.unique_labels), 1024, 3).to(self.device)

            b = 0
            for d in range(detections.shape[1]):
                dt = int(detections[b, d].item())
                if dt==1:
                    center, pred_center = self.extract_centers(heatmap[b,d,:,:,:], pred_heatmap[b,d,:,:,:])

                    radius = self.extract_radius(center, radius_map)
                    pred_radius = self.extract_radius(pred_center, pred_radius_map)
                    
                    
                    
                    cdisp = self.extract_center_disp(pred_center, center_disp)
                    
                    pred_centers[b:b+1,d, :] = pred_center*4 + cdisp   #pred_center.detach().cpu().squeeze()
                    radius_full[b:b+1,d, :] = radius.detach().cpu().squeeze()
                    pred_radius_full[b:b+1,d, :] = pred_radius.detach().cpu().squeeze()

                    
                    ref_lp = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["local_particles"]
                    ref_center = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["center"] #torch.mean(ref_lp, dim=1)

                    roi = self.extract_fpn_features(features, pred_centers[b,d, :], pred_radius)
                    roi = F.dropout(roi, p=0.2, training=False)
                    with torch.no_grad():
                        ref_lp_t =  (pred_centers[b,d, :] - ref_center) + ref_lp 
                    one_hot = torch.nn.functional.one_hot(torch.tensor(d), detections.shape[1]).unsqueeze(0).to(self.device)
                    l_disp_pred = self.lnet(roi, one_hot, torch.max(pred_radius).item())
                    pred_points_local = l_disp_pred + ref_lp_t
                    pred_points_local_full[b:b+1,d, :,:] = pred_points_local
                    
                    ref_wp = self.train_data.reference_paths[self.train_data.reverse_ld[d]]["world_particles"].to(self.device)                     
                    pred_points_world = self.rigid_registration(source_points=pred_points_local_full[b,d,:,:], target_points=ref_wp[0])
                    pred_points_world_full[b:b+1,d, :,:] = pred_points_world

        return [centers, pred_centers, radius_full, pred_radius_full, pred_heatmap, pred_points_local_full, pred_points_world_full]

    
    def extract_centers(self, heatmap, pred_heatmap):
        """
        Extract the centers from the predicted and ground truth heatmaps.

        Parameters
        ----------
        heatmap : torch.tensor
            The ground truth heatmap of shape (batch_size, num_classes, height, width, depth).
        pred_heatmap : torch.tensor
            The predicted heatmap of shape (batch_size, num_classes, height, width, depth).

        Returns
        -------
        centers : torch.tensor
        pred_centers : torch.tensor
        """
        hm = heatmap
        pred_hm = pred_heatmap
        centers = (hm==torch.max(hm)).nonzero().float()[0].to(self.device)
        pred_centers = (pred_hm==torch.max(pred_hm)).nonzero().float()[0].to(self.device)

        return centers.to(self.device), pred_centers.to(self.device)


    def extract_feature_region(self, feature, center, radius, scale):
        """
        Extract the feature region from the feature map based on the center and radius of the feature.

        Parameters
        ----------
        feature : torch.tensor
            The feature map of shape (batch_size, channels, height, width, depth).
        center : torch.tensor
            The center of the feature of shape (3,).
        radius : torch.tensor
            The radius of the feature of shape (3,).
        scale : torch.tensor
            The scale of the feature of shape ().

        Returns
        -------
        feature_region : torch.tensor
            The feature region of shape (batch_size, channels, height, width, depth).
        """
        center = center / scale
        radius = radius / scale
        x_min = (center[0] - radius[0]).long()
        x_max = (center[0] + radius[0]).long() + 1
        y_min = (center[1] - radius[1]).long()
        y_max = (center[1] + radius[1]).long() + 1
        z_min = (center[2] - radius[2]).long()
        z_max = (center[2] + radius[2]).long() + 1
        x_min = x_min if x_min > 0 else 0
        y_min = y_min if y_min > 0 else 0
        z_min = z_min if z_min > 0 else 0
        x_max = x_max if x_max < feature.shape[2] else (feature.shape[2] - 1)
        y_max = y_max if y_max < feature.shape[3] else (feature.shape[3] - 1)
        z_max = z_max if z_max < feature.shape[4] else (feature.shape[4] - 1)
        return feature[:,:,x_min:x_max,y_min:y_max,z_min:z_max]

    def pad_input(self, im: torch.Tensor, output_size: List[int]) -> torch.Tensor:
        """
        Pad the input tensor to the specified output size.

        Args:
            im (torch.Tensor): The input tensor of shape (batch_size, channels, height, width, depth).
            output_size (List[int]): The desired output size of the tensor as a list of 3 integers.

        Returns:
            torch.Tensor: The padded tensor of shape (batch_size, channels, output_size[0], output_size[1], output_size[2]).
        """
        pooling = torch.nn.AdaptiveMaxPool3d(output_size)
        bs, ch, w, h, l = im.shape
        if w == 0 or h == 0 or l == 0:
            # If the input tensor has a size of 0 in any dimension, return a tensor of zeros with the desired output size
            return torch.zeros(bs, ch, output_size[0], output_size[1], output_size[2]).to("cuda")
        if w >= output_size[0] and h >= output_size[1] and l >= output_size[2]:
            # If the input tensor is larger than the desired output size, use adaptive max pooling to downsample it
            return pooling(im)
        padding_tuple = (0, output_size[2] - l, 0, output_size[1] - h, 0, output_size[0] - w)
        im = nn.functional.pad(im, padding_tuple, mode="replicate")
        # Use adaptive max pooling to downsample the padded tensor to the desired output size
        return pooling(im)

    def extract_fpn_features(self, fpn_x: List[torch.Tensor], center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Extract the feature regions from the feature pyramid based on the center and radius of the feature.

        Parameters
        ----------
        fpn_x : List[torch.Tensor]
            The feature pyramid of shape (batch_size, channels, height, width, depth).
        center : torch.tensor
            The center of the feature of shape (3,).
        radius : torch.tensor
            The radius of the feature of shape (3,).

        Returns
        -------
        feature_regions : torch.tensor
            The feature regions of shape (batch_size, channels, height, width, depth).
        """
        ## return center and radius to original size
        # center = center * 4
        # radius = radius * 4
        roi_full = []
        start_size = 5
        size_list = [8, 4, 2, 1]
        roi_final = 0
        for i in range(len(fpn_x)):
            start_size = size_list[i]
            roi = self.extract_feature_region(fpn_x[i], center, radius, 2**(i+2))
            roi = Flatten()(self.pad_input(roi, [start_size,start_size,start_size]))
            # roi_final += roi
            roi_full.append(roi)
            # start_size -= 1
        return torch.cat(roi_full, dim=1)
    
    
    def get_center(self, seg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get the center coordinates and radius of the segmentation.

        Parameters
        ----------
        seg : torch.Tensor
            The segmentation tensor of shape (batch_size, channels, height, width, depth).

        Returns
        -------
        center : torch.Tensor
            The center coordinates of shape (3,).
        radius : torch.Tensor
            The radius of shape (3,).
        seg_crop : torch.Tensor
            The cropped segmentation tensor of shape (batch_size, channels, height, width, depth).
        """
        # Get the indices of the segmentation
        crop_idx = torch.where(seg > 0.9)

        # Initialize the center and radius
        cx = cy = cz = max_dt = 0

        # If the segmentation is not empty
        if crop_idx[2].nelement() != 0:
            # Get the bounding box of the segmentation
            x_min = crop_idx[2].min().long()
            x_max = crop_idx[2].max().long()
            y_min = crop_idx[3].min().long()
            y_max = crop_idx[3].max().long()
            z_min = crop_idx[4].min().long()
            z_max = crop_idx[4].max().long()

            # Calculate the center coordinates
            cx, cy, cz = (x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2

        # Return the center coordinates, radius, and cropped segmentation
        return torch.tensor([cx, cy, cz]).long(), torch.tensor([abs(x_max - x_min), abs(y_max - y_min), abs(z_max - z_min)]), seg[:, :, x_min:x_max, y_min:y_max, z_min:z_max]
    

    def extract_radius(self, center: torch.Tensor, radius_map: torch.Tensor) -> torch.Tensor:
        """
        Extract the radius of the segmentation from the radius map based on the center coordinates.

        Parameters
        ----------
        center : torch.Tensor
            The center coordinates of shape (3,).
        radius_map : torch.Tensor
            The radius map of shape (1, 3, height, width, depth).

        Returns
        -------
        radius : torch.Tensor
            The radius of shape (3,).
        """
        # Get the center coordinates as long integers
        center_ = center.long()
        
        # Extract the radius from the radius map
        rx = radius_map[0, 0, center_[0], center_[1], center_[2]]
        ry = radius_map[0, 1, center_[0], center_[1], center_[2]]
        rz = radius_map[0, 2, center_[0], center_[1], center_[2]]
        
        # Return the radius as a tensor
        return torch.tensor([rx, ry, rz])

    def extract_center_disp(self, center: torch.Tensor, center_disp: torch.Tensor) -> torch.Tensor:
        """
        Extract the center displacement from the center displacement map based on the center coordinates.

        Parameters
        ----------
        center : torch.Tensor
            The center coordinates of shape (3,).
        center_disp : torch.Tensor
            The center displacement map of shape (batch_size, 3, height, width, depth).

        Returns
        -------
        center_disp : torch.Tensor
            The center displacement of shape (3,).
        """
        # Get the center coordinates as long integers
        center_ = center.long()
        
        # Extract the center displacement from the center displacement map
        return center_disp[:,:,center_[0], center_[1], center_[2]]

    def extract_crop(self, region: torch.Tensor, center: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
        """
        Extract the crop from the region based on the center coordinates and radius.

        Parameters
        ----------
        region : torch.Tensor
            The 3D tensor of shape (height, width, depth).
        center : torch.Tensor
            The center coordinates of shape (3,).
        radius : torch.Tensor
            The radius of shape (3,).

        Returns
        -------
        crop : torch.Tensor
            The crop of shape (height, width, depth).
        """
        # Calculate the crop indices
        rx = radius[0].item()
        ry = radius[1].item()
        rz = radius[2].item()
        x_min = (center[0]*4 - rx).long()
        x_max = (center[0]*4 + rx).long()
        y_min = (center[1]*4 - ry).long()
        y_max = (center[1]*4 + ry).long()
        z_min = (center[2]*4 - rz).long()
        z_max = (center[2]*4 + rz).long()
        
        # Make sure the crop indices are within the bounds of the region
        x_min = x_min if x_min > 0 else 0
        y_min = y_min if y_min > 0 else 0
        z_min = z_min if z_min > 0 else 0
        
        # Extract the crop from the region
        return region[x_min:x_max, y_min:y_max, z_min:z_max]


    
    ### create different for inference
    
    def save_network(self, name):
        """
        Save the network weights.
        """
        torch.save(self.object_net.state_dict(), os.path.join(self.save_dir, "object_net_"+name+".torch"))
        torch.save(self.lnet.state_dict(), os.path.join(self.save_dir, "lnet_"+name+".torch"))
        torch.save(self.wnet.state_dict(), os.path.join(self.save_dir, "wnet_"+name+".torch"))
    
    # def load_pretrained(self, name):
    #     self.object_net.load_state_dict(torch.load(os.path.join(self.save_dir, "object_net_"+name+".torch")))
    #     self.lnet.load_state_dict(torch.load(os.path.join(self.save_dir, "lnet_"+name+".torch")))
    #     self.wnet.load_state_dict(torch.load(os.path.join(self.save_dir, "wnet_"+name+".torch")))
    
    def load_pretrained(self, name):
        """
        Load the network weights.
        """
        self.object_net.load_state_dict(torch.load(os.path.join(self.save_dir, "object_net_"+"val_best_lp"+".torch")))
        self.lnet.load_state_dict(torch.load(os.path.join(self.save_dir, "lnet_"+"val_best_lp"+".torch")))
        self.wnet.load_state_dict(torch.load(os.path.join(self.save_dir, "wnet_"+"val_best_wp"+".torch")))
    