import numpy as np
import torch
from util.data_loaders import *
import config as cfg
from networks.model_fullts import MultiImage2Shape
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import random
import nrrd
from collections import defaultdict
import math
from torch.utils.tensorboard import SummaryWriter
import pickle
from util.net_utils import plot_particles

def RMSE(predicted, ground_truth):
    """
    Compute the root mean squared error between the predicted and ground truth values.
    """
    a = torch.sqrt(torch.mean((predicted - ground_truth)**2, dim=1))
    return torch.mean(a)



if __name__ == "__main__":
    
    save_imdt = os.path.join(cfg.checkpoint_dir, "output")
    if not os.path.exists(save_imdt):
        os.makedirs(save_imdt)
    
    writer = SummaryWriter(os.path.join(cfg.checkpoint_dir, "logs"))
    print("--------- Loading training data ---------")
    train_dataset = MultiClassDataset_FA(cfg, split="train")
    val_dataset =  MultiClassDataset_FA(cfg, split="val", unique_labels=train_dataset.unique_labels, \
        labels_dict=train_dataset.labels_dict, reverse_ld=train_dataset.reverse_ld)
    
    train_loader = DataLoader(
			train_dataset,
			batch_size=cfg.batch_size,
			shuffle=True,
			num_workers=12,
		)

    val_loader = DataLoader(
			val_dataset,
			batch_size=cfg.batch_size,
			shuffle=False,
			num_workers=4,
		)
    start_epoch = 0
    
    print("--------- Initializing Networks ---------")
    model = MultiImage2Shape(in_channels=1, cfg=cfg, train_data=train_dataset)
    lambda_ = {"heatmap":1, "points_w":0, "points_l":0, "radius":0.02, "cdisp":0}
    schedule_prob = [1.0, 1.0]
    
    # if True:
    #     start_epoch = 0
    #     model.load_pretrained("val_best_lp")
    #     lambda_ = {"heatmap":40, "points_w":2, "points_l":3, "radius":0.02, "cdisp":1}
    #     schedule_prob = [0.0, 0.0]


    min_loss = math.inf
    val_loss = math.inf
    val_loss_wp = math.inf
    print("--------- Start Training ---------")
    train_w_loss = 0

    

    for epoch in tqdm(range(start_epoch ,cfg.num_epochs), desc="Epochs: "):
        torch.cuda.empty_cache()
        model.train()
        loss_dict = defaultdict(list)
        loss_dict_train = defaultdict(list)
        
        # Gradually increase the weight of the center displacement loss
        if lambda_["cdisp"] < 1:
            lambda_["cdisp"] += 0.1

        # Gradually increase the weight of the heatmap loss
        if epoch >= 20 and lambda_["heatmap"] < 40:
            lambda_["heatmap"] += 2.0

        # Gradually increase the weight of the local points loss
        if epoch >= 20 and lambda_["points_l"] < 3:
            lambda_["points_l"] += 0.3
        
        # Gradually decrease the probability of using the original center
        if epoch >= 25 and schedule_prob[0] > 0:
            schedule_prob[0] -= 0.1
            
        # Gradually increase the weight of the world points loss
        if epoch >= 40 and lambda_["points_w"] < 2:
            lambda_["points_w"] += 0.2
            
        # Gradually decrease the probability of using the original local particles
        if epoch >= 45 and schedule_prob[1] > 0:
            schedule_prob[1] -= 0.1


        
        for idx, (image, heatmap, radius_map, points_world, points_local, detections, center_true, segs_full) in enumerate((tqdm(train_loader, desc="Batch: "))):
            rnd_idx = random.sample(range(0, len(train_loader)), 2)
            model.optimizer.zero_grad()
            image = image.to(cfg.DEVICE)
            heatmap = heatmap.to(cfg.DEVICE)
            points_world = points_world.to(cfg.DEVICE)
            points_local = points_local.to(cfg.DEVICE)
            detections = detections.to(cfg.DEVICE)
            radius_map = radius_map.to(cfg.DEVICE)
            center_true = center_true.to(cfg.DEVICE)

            losses, outputs = model(image, heatmap, radius_map, points_world, points_local, detections, center_true, schedule_prob, segs_full)
            loss_heatmap, loss_radius, loss_local, loss_cdisp, loss_world = losses
            centers, pred_centers, radius_full, pred_radius_full, pred_heatmap, crop_list, pred_points_local, pred_points_world = outputs
            final_loss = loss_heatmap*lambda_["heatmap"] + loss_radius*lambda_["radius"] + loss_cdisp*lambda_["cdisp"] + loss_local*lambda_["points_l"] + loss_world*lambda_["points_w"] 
            
            final_loss.backward()
            model.optimizer.step()
            
            loss_dict["loss_heatmap"].append(loss_heatmap.item()*lambda_["heatmap"])
            loss_dict["loss_radius"].append(loss_radius.item()*lambda_["radius"])
            loss_dict["loss_cdisp"].append(loss_cdisp.item()*lambda_["cdisp"])
            loss_dict["loss_points_w"].append(loss_world.item()*lambda_["points_w"])
            loss_dict["loss_points_l"].append(loss_local.item()*lambda_["points_l"])
            loss_dict["final_loss"].append(final_loss.item())

            with torch.no_grad():
                loss_dict_train["loss_heatmap"].append(model.neg_loss(pred_heatmap.detach(), heatmap.detach(), detections).cpu().squeeze())
                for i in range(radius_full.shape[1]):
                    if detections[:,i].item() == 0:
                        continue
                    loss_dict_train["radius_"+train_dataset.reverse_ld[i]].append(np.sqrt(model.MSE(radius_full[:,i,:].detach().cpu(), \
                        pred_radius_full[:,i,:].detach().cpu()).squeeze().item()))
                    loss_dict_train["center_"+train_dataset.reverse_ld[i]].append(np.sqrt(model.MSE(center_true[:,i,:].detach().cpu(), \
                        pred_centers[:,i,:].detach().cpu()).squeeze().item()))
                    loss_dict_train["local_"+train_dataset.reverse_ld[i]].append(RMSE(points_local[:,i,:,:], pred_points_local[:,i,:,:]).item())
                    loss_dict_train["world_"+train_dataset.reverse_ld[i]].append(RMSE(points_world[:,i,:,:], pred_points_world[:,i,:,:]).item())
            
                
                
        loss_out = ""
        
        # Debug log the losses
        for key, val in loss_dict.items():
            mean_ = np.mean(np.array(val))
            loss_out += key + " = " + str(mean_) + ", "
            writer.add_scalar("Train/"+key, mean_, epoch)
        tqdm.write("Epoch - "+str(epoch+1)+" : "+loss_out)
        
        ## Validation every 5 epochs
        if epoch % 5 == 0:
            model.eval()
            loss_dict_val = defaultdict(list)
            rnd_idx = random.sample(range(0, len(val_loader)), 2)
            for idx, (image, heatmap, radius_map, points_world, points_local, detections, center_true, segs_full) in enumerate(val_loader):
                image = image.to(cfg.DEVICE)
                heatmap = heatmap.to(cfg.DEVICE)
                points_world = points_world.to(cfg.DEVICE)
                points_local = points_local.to(cfg.DEVICE)
                detections = detections.to(cfg.DEVICE)
                radius_map = radius_map.to(cfg.DEVICE)
                center_true = center_true.to(cfg.DEVICE)

                centers, pred_centers, radius_full, pred_radius_full, pred_heatmap, pred_points_local, pred_points_world = model.val(image, heatmap, radius_map, detections)
                

                with torch.no_grad():
                    loss_dict_val["loss_heatmap"].append(model.neg_loss(pred_heatmap.detach(), heatmap.detach(), detections).cpu().squeeze())
                    for i in range(radius_full.shape[1]):
                        if detections[:,i].item() == 0:
                            continue
                        loss_dict_val["radius_"+train_dataset.reverse_ld[i]].append(np.sqrt(model.MSE(radius_full[:,i,:].detach().cpu(), \
                            pred_radius_full[:,i,:].detach().cpu()).squeeze().item()))
                        loss_dict_val["center_"+train_dataset.reverse_ld[i]].append(np.sqrt(model.MSE(center_true[:,i,:].detach().cpu(), \
                            pred_centers[:,i,:].detach().cpu()).squeeze().item()))
                        loss_dict_val["local_"+train_dataset.reverse_ld[i]].append(RMSE(points_local[:,i,:,:], pred_points_local[:,i,:,:]).item())
                        loss_dict_val["world_"+train_dataset.reverse_ld[i]].append(RMSE(points_world[:,i,:,:], pred_points_world[:,i,:,:]).item())

                    

            for key in loss_dict_val.keys():
                writer.add_scalars("Val/"+key, {'val':np.mean(np.array(loss_dict_val[key])),
                                                'train': np.mean(np.array(loss_dict_train[key]))}, epoch)                              


            # Debug output the loss values
            loss_out = ""
            val_local_loss = []
            val_world_loss = []
            for key, val in loss_dict_val.items():
                mean_ = np.mean(np.array(val))
                loss_out = (key + " = " + str(mean_))
                if "local" in key:
                    val_local_loss.append(mean_)
                elif "world" in key:
                    val_world_loss.append(mean_)
                tqdm.write("Val Loss - : "+loss_out)
            loss_out = ""
            for key, val in loss_dict_train.items():
                mean_ = np.mean(np.array(val))
                loss_out = (key + " = " + str(mean_))
                tqdm.write("Train Loss - : "+loss_out)


            # save best model for local and world
            if epoch >= 40:
                cvl = np.mean(np.array(val_local_loss))
                if val_loss >= cvl:
                    model.save_network("val_best_lp")
                    tqdm.write("Val best for local found at epoch="+str(epoch+1))
                    val_loss = cvl
                cvw = np.mean(np.array(val_world_loss))
                if val_loss_wp >= cvw:
                    model.save_network("val_best_wp")
                    tqdm.write("Val best for world found at epoch="+str(epoch+1))
                    val_loss_wp = cvw

            
        if epoch > 20:
            model.scheduler.step()
        
        

    print("--------- Done Training ---------")
    
    writer.flush()
    writer.close()      

    
                                
            
    
    
    
    

    