import torch
from torch import nn
import sys
import json
import numpy as np
from collections import OrderedDict
import os
import torch.nn.functional as F

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

# ResNet from https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/models/backbone.py

class ConvGenerator(nn.Module):

	def __init__(self, c_in, c_out, ks, pad=0, stride=1, norm='instance_norm', relu='relu'):
		super(ConvGenerator, self).__init__()
		self.module = nn.Conv3d(c_in, c_out, kernel_size=ks, padding=pad, stride=stride)
		if norm is not None:
			if norm == 'instance_norm':
				norm_layer = nn.InstanceNorm3d(c_out)
			elif norm == 'batch_norm':
				norm_layer = nn.BatchNorm3d(c_out)
			else:
				raise ValueError('norm type as specified in configs is not implemented... {}'.format(norm))
			self.module = nn.Sequential(self.module, norm_layer)


		if relu is not None:
			if relu == 'relu':
				relu_layer = nn.ReLU(inplace=True)
			elif relu == 'leaky_relu':
				relu_layer = nn.LeakyReLU(inplace=True)
			else:
				raise ValueError('relu type as specified in configs is not implemented...')
			self.module = nn.Sequential(self.module, relu_layer)

	def forward(self, x):
		x = self.module(x)
		return x


class Interpolate(nn.Module):
	def __init__(self, scale_factor, mode):
		super(Interpolate, self).__init__()
		self.interp = nn.functional.interpolate
		self.scale_factor = scale_factor
		self.mode = mode

	def forward(self, x):
		x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
		return x

class ResBlock(nn.Module):

	def __init__(self, in_features, planes, out_features, stride=1, identity_skip=True, norm='batch_norm', relu='relu'):

		super(ResBlock, self).__init__()

		self.conv1 = ConvGenerator(in_features, planes, ks=1, stride=stride, norm=norm, relu=relu)
		self.conv2 = ConvGenerator(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
		self.conv3 = ConvGenerator(planes, out_features, ks=1, norm=norm, relu=None)
		if relu == 'relu':
			self.relu = nn.ReLU(inplace=True)
		elif relu == 'leaky_relu':
			self.relu = nn.LeakyReLU(inplace=True)
		else:
			raise Exception("Chosen activation {} not implemented.".format(self.relu))

		if stride!=1 or in_features!=out_features or not identity_skip:
			self.scale_residual = ConvGenerator(in_features, out_features, ks=1, stride=stride, norm=norm, relu=None)
		else:
			self.scale_residual = None

	def forward(self, x):
		out = self.conv1(x)
		out = self.conv2(out)
		out = self.conv3(out)
		if self.scale_residual:
			residual = self.scale_residual(x)
		else:
			residual = x
		out += residual
		out = self.relu(out)
		return out


class FPN(nn.Module):
	"""
	Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
	by default is constructed with Pyramid levels P2, P3, P4, P5.
	"""
	def __init__(self, start_features, out_channels, relu_enc="relu", relu_dec=None, norm='batch_norm', fpn_output=True):
		"""
		:param relu_enc: string specifying type of nonlinearity in encoder. If None, no nonlinearity is applied.
		:param relu_dec: same as relu_enc but for decoder.
		:param start_features:  number of feature_maps in first layer. rest is scaled accordingly.
		:param out_channels: number of feature_maps for output_layers of all levels in decoder.
		:param norm: string specifying type of feature map normalization. If None, no normalization is applied.
		:param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
		"""
		super(FPN, self).__init__()

		self.start_filts, sf = start_features, start_features
		self.out_channels = out_channels
		self.n_blocks = [3, 4, 23, 3]#[2, 2, 2, 2]  ## [3, 4, 23, 3] - for resnet-101 | [3, 4, 6, 3] - for resnet-50
		self.block_exp = 2 #factor by which to increase nr of channels in first block layer.
		self.relu_enc = relu_enc
		self.relu_dec = relu_dec
		self.sixth_pooling = False
		self.fpn_out = fpn_output

		self.C1 = ConvGenerator(1, start_features, ks=5, stride=2, pad=2, norm=norm, relu=relu_enc)

		C2_layers = []
		C2_layers.append(nn.MaxPool3d(kernel_size=3, stride=2, padding=1))
		C2_layers.append(ResBlock(sf, sf, sf*self.block_exp, stride=1, norm=norm, relu=relu_enc))
		for i in range(1, self.n_blocks[0]):
			C2_layers.append(ResBlock(sf*self.block_exp, sf, sf*self.block_exp, stride=1, norm=norm, relu=relu_enc))
		self.C2 = nn.Sequential(*C2_layers)

		C3_layers = []
		C3_layers.append(ResBlock(sf*self.block_exp, sf*2, sf*self.block_exp*2, stride=2, norm=norm, relu=relu_enc))
		for i in range(1, self.n_blocks[1]):
			C3_layers.append(ResBlock(sf*self.block_exp*2, sf*2, sf*self.block_exp*2, norm=norm, relu=relu_enc))
		self.C3 = nn.Sequential(*C3_layers)

		C4_layers = []
		C4_layers.append(ResBlock(sf*self.block_exp*2, sf*4, sf*self.block_exp*4, stride=2, norm=norm, relu=relu_enc))
		for i in range(1, self.n_blocks[2]):
			C4_layers.append(ResBlock(sf*self.block_exp*4, sf*4, sf*self.block_exp*4, norm=norm, relu=relu_enc))
		self.C4 = nn.Sequential(*C4_layers)

		C5_layers = []
		C5_layers.append(ResBlock(sf*self.block_exp*4, sf*8, sf*self.block_exp*8, stride=2, norm=norm, relu=relu_enc))
		for i in range(1, self.n_blocks[3]):
			C5_layers.append(ResBlock(sf*self.block_exp*8, sf*8, sf*self.block_exp*8, norm=norm, relu=relu_enc))
		self.C5 = nn.Sequential(*C5_layers)

		if self.sixth_pooling:
			C6_layers = []
			C6_layers.append(ResBlock(sf*self.block_exp*8, sf*16, sf*self.block_exp*16, stride=2, norm=norm, relu=relu_enc))
			for i in range(1, self.n_blocks[3]):
				C6_layers.append(ResBlock(sf*self.block_exp*16, sf*16, sf*self.block_exp*16, norm=norm, relu=relu_enc))
			self.C6 = nn.Sequential(*C6_layers)

		self.final = nn.AdaptiveAvgPool3d((2,2,2))

		if self.fpn_out:
			self.P1_upsample = Interpolate(scale_factor=2, mode='trilinear')
			self.P2_upsample = Interpolate(scale_factor=2, mode='trilinear')

			if self.sixth_pooling:
				self.P6_conv1 = ConvGenerator(sf*self.block_exp*16, self.out_channels, ks=1, stride=1, relu=relu_dec)
			self.P5_conv1 = ConvGenerator(sf*self.block_exp*8, self.out_channels, ks=1, stride=1, relu=relu_dec)
			self.P4_conv1 = ConvGenerator(sf*self.block_exp*4, self.out_channels, ks=1, stride=1, relu=relu_dec)
			self.P3_conv1 = ConvGenerator(sf*self.block_exp*2, self.out_channels, ks=1, stride=1, relu=relu_dec)
			self.P2_conv1 = ConvGenerator(sf*self.block_exp, self.out_channels, ks=1, stride=1, relu=relu_dec)
			self.P1_conv1 = ConvGenerator(sf, self.out_channels, ks=1, stride=1, relu=relu_dec)


			self.P1_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
			self.P2_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
			self.P3_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
			self.P4_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)
			self.P5_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)

			if self.sixth_pooling:
				self.P6_conv2 = ConvGenerator(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=relu_dec)


	def forward(self, x):
		"""
		:param x: input image of shape (b, c, y, x, (z))
		:return: list of output feature maps per pyramid level, each with shape (b, c, y, x, (z)).
		"""
		c0_out = x

		c1_out = self.C1(c0_out)
		c2_out = self.C2(c1_out)
		c3_out = self.C3(c2_out)
		c4_out = self.C4(c3_out)
		c5_out = self.C5(c4_out)

		if self.fpn_out==False:
			return c5_out
		else:
			if self.sixth_pooling:
				c6_out = self.C6(c5_out)
				p6_pre_out = self.P6_conv1(c6_out)
				p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
			else:
				p5_pre_out = self.P5_conv1(c5_out)

			p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
			p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
			p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)


			p2_out = self.P2_conv2(p2_pre_out)
			p3_out = self.P3_conv2(p3_pre_out)
			p4_out = self.P4_conv2(p4_pre_out)
			p5_out = self.P5_conv2(p5_pre_out)
			out_list = [p2_out, p3_out, p4_out, p5_out]

			if self.sixth_pooling:
				p6_out = self.P6_conv2(p6_pre_out)
				out_list.append(p6_out)

			return out_list

## For debug purpose only   
if __name__ == "__main__":
	DEVICE = "cuda"
	encoder = FPN(32, 18, fpn_output=True)#.to(DEVICE)
	float_tensor = torch.tensor(np.zeros((1, 1, 256, 224, 448))).float()#.to(DEVICE)
	output = encoder(float_tensor)
	#print(output.shape)
	for out in output:
		print(out.shape)
