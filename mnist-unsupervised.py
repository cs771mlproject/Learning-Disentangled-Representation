from __future__ import print_function
import numpy as np
from scipy.stats import norm
import torch
import torch.nn as nn
from torch.nn import Parameter
import sys
from torchvision import datasets, transforms
import os
import time
from random import random
sys.path.append('../')
import probtorch
from probtorch.util import expand_inputs
print('probtorch:', probtorch.__version__,
      'torch:', torch.__version__,
      'cuda:', torch.cuda.is_available())


# model parameters
NUM_PIXELS = 784
NUM_HIDDEN = 256
NUM_DIGITS = 10
NUM_LATENT = 10

# training parameters
NUM_SAMPLES = 8
NUM_BATCH = 128
NUM_EPOCHS = 200
LEARNING_RATE = 1e-3
BETA1 = 0.90
EPS = 1e-9
CUDA = torch.cuda.is_available()

# path parameters
MODEL_NAME = 'mnist-%02ddim' % NUM_LATENT
WEIGHTS_PATH = 'weights_unsupervised'
IMAGES_PATH = 'images_unsupervised'
RESTORE = False


class Encoder(nn.Module):
	def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden=NUM_HIDDEN,
                       num_latent=NUM_LATENT,
                       num_batch=NUM_BATCH):
		super(self.__class__, self).__init__()
		self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden),
                            nn.ReLU())
		self.z_mean = nn.Linear(num_hidden, num_latent)
		self.z_log_std = nn.Linear(num_hidden, num_latent)

	@expand_inputs
	def forward(self, images, labels=None, num_samples=None):
		q = probtorch.Trace()
		hiddens = self.enc_hidden(images)
		q.normal(self.z_mean(hiddens),
			self.z_log_std(hiddens).exp(),
			name='z')
		return q

class Decoder(nn.Module):
	def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden=NUM_HIDDEN,
                       num_latent=NUM_LATENT):
		super(self.__class__, self).__init__()
		self.z_mean = torch.zeros(num_latent)
		self.z_std = torch.ones(num_latent)
		self.dec_image = nn.Sequential(
					nn.Linear(num_latent, num_hidden),
					nn.ReLU(),
					nn.Linear(num_hidden, num_pixels),
					nn.Sigmoid())

	def forward(self, images, q=None, num_samples=None):
		p = probtorch.Trace()
		z = p.normal(self.z_mean,
                     self.z_std,
                     value=q['z'],
                     name='z')
		images_mean = self.dec_image(z)
		p.loss(lambda x_hat, x: -(torch.log(x_hat + EPS) * x +
				torch.log(1 - x_hat + EPS) * (1-x)).sum(-1),
				images_mean, images, name='x')
		return p

def elbo(q, p, alpha=0.1):
	if NUM_SAMPLES is None:
		return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=None, batch_dim=0, alpha=alpha)
	else:
		return probtorch.objectives.montecarlo.elbo(q, p, sample_dim=0, batch_dim=1, alpha=alpha)

if not os.path.isdir(DATA_PATH):
	os.makedirs(DATA_PATH)

train_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=True, download=False,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)
test_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=False, download=False,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)

def cuda_tensors(obj):
	for attr in dir(obj):
		value = getattr(obj, attr)
		if isinstance(value, torch.Tensor):
			setattr(obj, attr, value.cuda())

enc = Encoder()
dec = Decoder()
if CUDA:
	enc.cuda()
	dec.cuda()
	cuda_tensors(enc)
	cuda_tensors(dec)

optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                              lr=LEARNING_RATE,
                              betas=(BETA1, 0.999))


def train(data, enc, dec, optimizer):
	epoch_elbo = 0.0
	enc.train()
	dec.train()
	N = 0
	for b, (images, labels) in enumerate(data):
		if images.size()[0] == NUM_BATCH:
			N += NUM_BATCH
			images = images.view(-1, NUM_PIXELS)
			if CUDA:
				images = images.cuda()
			optimizer.zero_grad()
			q = enc(images, num_samples=NUM_SAMPLES)
			p = dec(images, q, num_samples=NUM_SAMPLES)
			loss = -elbo(q, p)
			loss.backward()
			optimizer.step()
			if CUDA:
				loss = loss.cpu()
			epoch_elbo -= float(loss.item())
	return epoch_elbo / N

def test(data, enc, dec):
	enc.eval()
	dec.eval()
	epoch_elbo = 0.0
	N = 0
	for b, (images, labels) in enumerate(data):
		if images.size()[0] == NUM_BATCH:
			N += NUM_BATCH
			images = images.view(-1, NUM_PIXELS)
			if CUDA:
				images = images.cuda()
			q = enc(images, num_samples=NUM_SAMPLES)
			p = dec(images, q, num_samples=NUM_SAMPLES)
			batch_elbo = elbo(q, p)
			if CUDA:
				batch_elbo = batch_elbo.cpu()
			epoch_elbo += float(batch_elbo.item())
	return epoch_elbo / N

if not RESTORE:
	print("Trainig Start")
	mask = {}
    	for e in range(NUM_EPOCHS):
        	train_start = time.time()
        	train_elbo = train(train_data, enc, dec, optimizer)
        	train_end = time.time()
		test_start = time.time()
		test_elbo = test(test_data, enc, dec)
		test_end = time.time()
		f = open("ouput_unsupervised.txt","a")
		f.write('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
			e, train_elbo, train_end - train_start,
			test_elbo, test_end - test_start))
		f.close()
		print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e (%ds)' % (
			e, train_elbo, train_end - train_start,
			test_elbo, test_end - test_start))
	if not os.path.isdir(WEIGHTS_PATH):
		os.mkdir(WEIGHTS_PATH)
	torch.save(enc.state_dict(),
               '%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
	torch.save(dec.state_dict(),
               '%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
