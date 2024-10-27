import numpy as np
import os

DEVICE = "cuda"
config_file = "datasets/total_segmentator/final_shapes3.json"
checkpoint_dir = "checkpoints/run43-sae"
img_dims = [256, 224, 448]
batch_size = 1
num_epochs = 1000
learning_rate = 1e-4
seg_vae_save_path = "checkpoints/run35-seg/save/model_best.torch"
wp_vae_save_path = "checkpoints/run43-vae/save/model_best.torch"
wp_latent = 16
seg_latent = 2048
is_dt = False
normalize_images = False
model_scale = 4
preprocess_augmentation = False
aug_save_path = "/home/sci/janmesh/DeepSSM/Janmesh_Data/processed_lzf"
augment = True


