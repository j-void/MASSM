import torch
from torch import nn
import sys
import json
import numpy as np
from collections import OrderedDict
import os
import torch.nn.functional as F
from .resnet import FPN

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class DoubleConv(nn.Module):
	"""
    Double convolution layer

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        encoder (bool, optional): Whether to use this layer as an encoder. Defaults to True.
        conv_kernel_size (int, optional): The size of the convolution kernel. Defaults to 3.
    """
	def __init__(self, in_channels, out_channels, encoder=True, conv_kernel_size=3, conv_padding=1):

		super(DoubleConv, self).__init__()
		
		conv1_in_channels = in_channels
		conv1_out_channels = out_channels
		conv2_in_channels = out_channels
		conv2_out_channels = out_channels
		
		self.double_conv = nn.Sequential(
			nn.Conv3d(conv1_in_channels, conv1_out_channels, conv_kernel_size, padding=conv_padding),
			nn.InstanceNorm3d(conv1_out_channels),
			nn.ReLU(inplace=True),
			nn.Conv3d(conv2_in_channels, conv2_out_channels, conv_kernel_size, padding=conv_padding),
			nn.InstanceNorm3d(conv2_out_channels),
			nn.ReLU(inplace=True),
			# nn.Dropout3d(0.1)
		)
		
	def forward(self, x):
		"""
		Args:
			x (torch.Tensor): The input tensor.

		Returns:
		"""
		return self.double_conv(x)


class Upscale(nn.Module):
	"""
	Upscales without skip connections

	Parameters:
	num_levels (int): The number of levels of the feature pyramid.
	f_maps (int): The number of feature maps.

	"""
	def __init__(self, num_levels, f_maps):
		super(Upscale, self).__init__()

		self.decoder_steps = nn.ModuleList()
		self.upconv = nn.ModuleList()
		features = f_maps
		for k in reversed(range(num_levels-1)):
			self.upconv.append(nn.ConvTranspose3d(features, features, kernel_size=2, stride=2))
			self.decoder_steps.append(DoubleConv(in_channels=features, out_channels=features))


	def forward(self, x):
		for i in range(len(self.decoder_steps)):
			x = self.upconv[i](x)
			x = self.decoder_steps[i](x)
		return x
	


	
class Upscale_Skip(nn.Module):
	"""
	Upscales with skip connections

	Parameters:
	num_levels (int): The number of levels of the feature pyramid.
	f_maps (int): The number of feature maps.

	"""
	def __init__(self, num_levels, f_maps):
		super(Upscale_Skip, self).__init__()

		self.decoder_steps = nn.ModuleList()
		self.upconv = nn.ModuleList()
		features = f_maps
		for i in range(num_levels-1):
			# print(i, features)
			self.upconv.append(nn.ConvTranspose3d(features, features, kernel_size=2, stride=2))
			features += f_maps # Add the number of feature maps of the current level
			self.decoder_steps.append(DoubleConv(in_channels=features, out_channels=features))


	def forward(self, fpn_x):
		fpn_x.reverse()
		x = fpn_x[0]
		for i in range(len(self.decoder_steps)):
			x = self.upconv[i](x)
			concat_x = torch.cat((fpn_x[i+1], x), dim=1)
			x = self.decoder_steps[i](concat_x)
			# print("c", concat_x.shape, x.shape)
		return x

class CenterNet(nn.Module):
	"""
	Implementation of the CenterNet model.

	Parameters:
	num_classes (int): The number of object classes.

	"""
	def __init__(self, num_classes):
		super(CenterNet, self).__init__()
		# The feature pyramid network
		self.FPN = FPN(32, 20, fpn_output=True, relu_enc='leaky_relu', relu_dec='leaky_relu', norm='instance_norm')
		# The upscale module with skip connections
		self.upscale = Upscale_Skip(4, 20)
		# The feature maps of the last layer of the feature pyramid
		features = 80 #//(2**(3-1))
		# The output layers - heatmap, radius, center displacement
		self.heatmap = nn.Sequential(
			nn.Conv3d(in_channels=features, out_channels=num_classes, kernel_size=1),
			nn.Sigmoid()
		)
		self.center_disp = nn.Conv3d(in_channels=features, out_channels=3, kernel_size=1)
		self.radius = nn.Conv3d(in_channels=features, out_channels=3, kernel_size=1)
		
		

	def forward(self, x):
		"""
		Forward pass of the CenterNet model.

		Parameters:
		x (torch.tensor): Input data of shape (batch_size, channels, height, width, depth).

		Returns:
		heatmap (torch.tensor): Output heatmap of shape (batch_size, num_classes, height, width, depth).
		radius (torch.tensor): Output radius of shape (batch_size, 3, height, width, depth).
		fpn_x (list): List of feature maps of the feature pyramid of shape (batch_size, channels, height, width, depth).
		center_disp (torch.tensor): Output center displacement of shape (batch_size, 3, height, width, depth).
		"""
		fpn_x = self.FPN(x) # Feature pyramid network
		h_x = self.upscale(fpn_x) # Upscale module with skip connections
		hm = self.heatmap(h_x) # Heatmap 
		r = self.radius(h_x) # Radiusmap
		cd = self.center_disp(h_x) # Center displacement map
		return hm, r, fpn_x, cd
	




## For debug purpose only   
if __name__ == "__main__":
	DEVICE = "cuda"
	encoder = CenterNet(6).to(DEVICE)
	float_tensor = torch.tensor(np.zeros((1, 1, 256, 224, 448))).float().to(DEVICE)
	output = encoder(float_tensor)
	
