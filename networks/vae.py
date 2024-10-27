import torch
from torch import nn
import sys
import json
import numpy as np
from collections import OrderedDict
import os

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)

class VAEEncoder(nn.Module):
	"""
	VAE Encoder

	Parameters
	----------
	num_latent : int
		Number of latent variables
	num_corr : int
		Number of correspondence points
	"""
	def __init__(self, num_latent, num_corr):

		super(VAEEncoder, self).__init__()

		self.encoder_steps = nn.Sequential(
	   		nn.Linear(num_corr*3, 512*3),
			nn.LeakyReLU(),

			nn.Linear(512*3, 512*2),
			nn.LeakyReLU(),

			nn.Linear(512*2, 512),
			nn.LeakyReLU(),
		)

		# final layer to compute mean and variance
		self.mu = nn.Linear(512, num_latent)
		self.logvar = nn.Linear(512, num_latent)

	def forward(self, x):
		if x is None:
			raise ValueError("Input to VAE encoder cannot be None")
		x = x.view(x.size(0), -1)
		x = self.encoder_steps(x)

		return self.mu(x), self.logvar(x)

class VAEDecoder(nn.Module):
	"""
	Conditional Decoder for VAE

	Parameters
	----------
	num_latent : int
		Number of latent variables.
	num_corr : int
		Number of correspondence points.
	num_classes : int
		Number of classes.
	"""
	def __init__(self, num_latent, num_corr, num_classes):

		super(VAEDecoder, self).__init__()

		self.decoder_net = nn.Sequential(
			nn.Linear(num_latent+num_classes, 512),
   			nn.LeakyReLU(),
		
			nn.Linear(512, 512*2),
   			nn.LeakyReLU(),

			nn.Linear(512*2, 512*3),
   			nn.LeakyReLU(),
		
			nn.Linear(512*3, num_corr*3)
		)

	def forward(self, z):
		if z is None:
			raise ValueError("Input to VAE decoder cannot be None")
		return self.decoder_net(z)


class SimpleCVAEModule(nn.Module):
	"""
	Simple Conditional VAE Module
	
	Parameters
	----------
	num_latent : int
		Number of latent variables.
	num_corr : int
		Number of correspondence points.
	num_classes : int
		Number of classes.
	"""
	def __init__(self, num_latent, num_corr, num_classes):
		super(SimpleCVAEModule, self).__init__()
		self.numL = num_corr
		self.encoder = VAEEncoder(num_latent=num_latent, num_corr=num_corr)
		self.decoder = VAEDecoder(num_latent=num_latent, num_corr=num_corr, num_classes=num_classes)
	
	def reparameterize(self, mu, logvar):
		# z = torch.distributions.normal.Normal(mu, torch.nn.functional.softplus(logvar)).rsample()
		std = torch.exp(0.5*logvar)
		eps = torch.randn_like(std)
		z = mu + eps*std
		return z

	def decode(self, z):
		return self.decoder(z).reshape(-1, self.numL, 3)
	
	def forward(self, x, one_hot):
		if x is None or one_hot is None:
			raise ValueError("Input to SimpleCVAEModule cannot be None")
		x = torch.cat((x.view(x.shape[0], -1), one_hot), dim=1)
		mu, logvar = self.encoder(x)
		z = self.reparameterize(mu, logvar)
		z = torch.cat((z, one_hot), dim=1)
		corr_out = self.decode(z)
		return corr_out, mu, logvar




# Encoder Decoder type for local module
# class LocalModule(nn.Module):
# 	def __init__(self, num_latent, num_corr):
# 		super(LocalModule, self).__init__()

# 		self.numL = num_corr

# 		self.encoder = nn.Sequential(
# 			nn.Linear(11700, 6144),
# 			nn.InstanceNorm1d(6144),
# 			nn.LeakyReLU(),
# 			nn.Dropout(0.1),
#    			nn.Linear(6144, 2048),
# 			nn.InstanceNorm1d(2048),
# 			nn.LeakyReLU(),
# 			nn.Dropout(0.1),
#    			nn.Linear(2048, 256),
# 			nn.InstanceNorm1d(256),
# 			nn.LeakyReLU(),
# 			nn.Linear(256, 32),
# 			nn.InstanceNorm1d(32)
# 		)
  
# 		self.decoder = nn.Sequential(
# 			nn.Linear(32+32, 256),
#    			nn.InstanceNorm1d(256),
# 			nn.LeakyReLU(),
# 			nn.Linear(256, 512),
#    			nn.InstanceNorm1d(512),
# 			nn.LeakyReLU(),
# 			nn.Linear(512, 1024),
# 			nn.InstanceNorm1d(1024),
# 			nn.LeakyReLU(),
# 			nn.Linear(1024, num_corr*3),
# 		)

  
# 		self.tanh = nn.Tanh()
		
# 	def forward(self, roi, label, bbox_max):

# 		x = self.encoder(roi)
# 		x =  torch.cat((x, label), dim=1)
  
# 		lp_disp = bbox_max * self.tanh(self.decoder(x)) 

# 		return lp_disp.reshape(-1, self.numL, 3)

class LocalModule(nn.Module):
	"""
	Local Particles Prediction Module.
	"""
	def __init__(self, num_latent, num_corr):
		"""
		num_latent : int
			Number of latent variables.
		num_corr : int
			Number of correspondence points.
		"""
		super(LocalModule, self).__init__()

		self.numL = num_corr

		# +7 for class label provided as one hot
		self.layers = nn.Sequential(
			nn.Linear(11700+7, 6144),
			nn.InstanceNorm1d(6144),
			nn.LeakyReLU(),
			nn.Dropout(0.2),
   			nn.Linear(6144, 6144),
			nn.InstanceNorm1d(6144),
			nn.LeakyReLU(),
			nn.Dropout(0.2),
   			nn.Linear(6144, 3072),
			nn.InstanceNorm1d(3072),
			nn.LeakyReLU(),
			nn.Linear(3072, num_corr*3),
		)

  
		self.tanh = nn.Tanh()
		
	def forward(self, roi, label, bbox_max):
		"""
		Parameters
		----------
		roi : torch.tensor
			Region of interest features.
		label : torch.tensor
			Class label.
		bbox_max : torch.tensor
			Maximum bounding box size.

		Returns
		-------
		torch.tensor
			Local particles displacement.
		"""
		# concatenate roi features with class label
		x = torch.cat((roi, label), dim=1)

		if x is None:
			raise ValueError("Input to LocalModule cannot be None")

		# pass through the network scale the displacement by tanh (-1, 1) then to (-bbox_max, bbox_max)
		# predicting displacement in constrained space
		lp_disp = bbox_max * self.tanh(self.layers(x))

		return lp_disp.reshape(-1, self.numL, 3)




## For debug purpose only   
if __name__ == "__main__":
	pass

