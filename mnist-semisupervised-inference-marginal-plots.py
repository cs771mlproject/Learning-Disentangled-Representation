from __future__ import print_function
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
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

# Model Paramters:
NUM_PIXELS = 784
NUM_HIDDEN1 = 400
NUM_HIDDEN2 = 200
NUM_STYLE = 10
NUM_DIGITS = 10

# Training Parameters:
NUM_SAMPLES = 1
NUM_BATCH = 128
NUM_EPOCHS = 200
LABEL_FRACTION = 0.1
LEARNING_RATE = 1e-3
EPS = 1e-9
BIAS_TRAIN = (60000 - 1) / (NUM_BATCH - 1)
BIAS_TEST = (10000 - 1) / (NUM_BATCH - 1)
CUDA = torch.cuda.is_available()

# LOSS parameters:
ALPHA = 0.1
BETA = (4.0, 1.0, 1.0, 0.0, 1.0)

# path parameters
MODEL_NAME = 'mnist-semisupervised-inference-marginal-%02ddim' % NUM_STYLE
DATA_PATH = 'data'
IMAGES_PATH = 'images_semisupervised_inference_marginal'
WEIGHTS_PATH = 'weights_semisupervised_inference_marginal'
RESTORE = True

class Encoder(nn.Module):

    def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden1=NUM_HIDDEN1,
                       num_hidden2=NUM_HIDDEN2,
                       num_style=NUM_STYLE,
                       num_digits=NUM_DIGITS):

        super(self.__class__, self).__init__()
        self.enc_hidden = nn.Sequential(
                            nn.Linear(num_pixels, num_hidden1),
                            nn.ReLU())
        self.digit_log_weights = nn.Linear(num_hidden1, num_digits)
        self.digit_temp = 0.66
        self.style_mean = nn.Sequential(
                            nn.Linear(num_hidden1 + num_digits, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_style))
        self.style_log_std = nn.Sequential(
                                nn.Linear(num_hidden1 + num_digits, num_hidden2),
                                nn.ReLU(),
                                nn.Linear(num_hidden2, num_style))

    @expand_inputs
    def forward(self, images, labels=None, num_samples=NUM_SAMPLES):
        q = probtorch.Trace()
        hidden = self.enc_hidden(images)
        digits = q.concrete(logits=self.digit_log_weights(hidden),
                            temperature=self.digit_temp,
                            value=labels,
                            name='y')
        hidden2 = torch.cat([digits, hidden] , -1)
        styles_mean = self.style_mean(hidden2)
        styles_std = self.style_log_std(hidden2).exp()
        q.normal(loc=styles_mean,
                 scale=styles_std,
                 name='z')
        return q

def binary_cross_entropy(x_mean, x, EPS=1e-9):
    return - (torch.log(x_mean + EPS) * x +
              torch.log(1 - x_mean + EPS) * (1 - x)).sum(-1)

class Decoder(nn.Module):
    def __init__(self, num_pixels=NUM_PIXELS,
                       num_hidden1=NUM_HIDDEN1,
                       num_hidden2=NUM_HIDDEN2,
                       num_style=NUM_STYLE,
                       num_digits=NUM_DIGITS):

        super(self.__class__, self).__init__()
        self.dec_hidden = nn.Sequential(
                            nn.Linear(num_style + num_digits, num_hidden2),
                            nn.ReLU(),
                            nn.Linear(num_hidden2, num_hidden1),
                            nn.ReLU())

        self.num_style = num_style
        self.num_digits = num_digits
        self.digit_temp = 0.66
        self.dec_images = nn.Sequential(
                            nn.Linear(num_hidden1, num_pixels),
                            nn.Sigmoid())

    def forward(self, images, q=None, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH):
        p = probtorch.Trace()
        digit_log_weights = torch.zeros(num_samples, batch_size, self.num_digits)
        style_mean = torch.zeros(num_samples, batch_size, self.num_style)
        style_std = torch.ones(num_samples, batch_size, self.num_style)

        if CUDA:
            digit_log_weights = digit_log_weights.cuda()
            style_mean = style_mean.cuda()
            style_std = style_std.cuda()

        digits = digits = p.concrete(logits=digit_log_weights,
                                     temperature=self.digit_temp,
                                     value=q['y'],
                                     name='y')

        styles = p.normal(loc=style_mean,
                          scale=style_std,
                          value=q['z'],
                          name='z')

        hiddens = self.dec_hidden(torch.cat([digits, styles], -1))
        images_mean = self.dec_images(hiddens)
        p.loss(binary_cross_entropy, images_mean, images, name='images')
        return p

def elbo(q, p, alpha=ALPHA, beta=BETA, bias=1.0):
    return probtorch.objectives.marginal.elbo(q, p, sample_dim=0, batch_dim=1,
                                              alpha=alpha, beta=beta, bias=bias)

if not os.path.isdir(DATA_PATH):
    os.makedirs(DATA_PATH)

train_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=True, download=True,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)
test_data = torch.utils.data.DataLoader(
                datasets.MNIST(DATA_PATH, train=False, download=True,
                               transform=transforms.ToTensor()),
                batch_size=NUM_BATCH, shuffle=True)

enc = Encoder()
dec = Decoder()
if CUDA:
    enc.cuda()
    dec.cuda()
optimizer =  torch.optim.Adam(list(enc.parameters())+list(dec.parameters()),
                              lr=LEARNING_RATE)

def train(data, enc, dec, optimizer,
          label_mask={}, label_fraction=LABEL_FRACTION):
    epoch_elbo = 0.0
    enc.train()
    dec.train()
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size(0) == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            labels_onehot = torch.zeros(NUM_BATCH, NUM_DIGITS)
            labels_onehot.scatter_(1, labels.unsqueeze(1), 1)
            labels_onehot = torch.clamp(labels_onehot, EPS, 1-EPS)
            if CUDA:
                images = images.cuda()
                labels_onehot = labels_onehot.cuda()
            optimizer.zero_grad()
            if b not in label_mask:
                label_mask[b] = (random() < label_fraction)
            if label_mask[b]:
                q = enc(images, labels_onehot, num_samples=NUM_SAMPLES)
            else:
                q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH)
            loss = -elbo(q, p, bias=BIAS_TRAIN)
            loss.backward()
            optimizer.step()
            if CUDA:
                loss = loss.cpu()
            epoch_elbo -= loss.item()
    return epoch_elbo / N, label_mask


def test(data, enc, dec, infer=True):
    enc.eval()
    dec.eval()
    epoch_elbo = 0.0
    epoch_correct = 0
    N = 0
    for b, (images, labels) in enumerate(data):
        if images.size()[0] == NUM_BATCH:
            N += NUM_BATCH
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            p = dec(images, q, num_samples=NUM_SAMPLES, batch_size=NUM_BATCH)
            batch_elbo = elbo(q, p, bias=BIAS_TEST)
            if CUDA:
                batch_elbo = batch_elbo.cpu()
            epoch_elbo += batch_elbo.data.numpy()

            log_p = p.log_joint(0, 1)
            log_q = q.log_joint(0, 1)
            log_w = log_p - log_q
            w = torch.nn.functional.softmax(log_w, 0)
            y_samples = q['y'].value
            y_expect = (w.unsqueeze(-1) * y_samples).sum(0)
            _ , y_pred = y_expect.data.max(-1)
            if CUDA:
                    y_pred = y_pred.cpu()
            epoch_correct += (labels == y_pred).float().sum()
    return epoch_elbo / N, epoch_correct / N

if not os.path.isdir(IMAGES_PATH):
	os.mkdir(IMAGES_PATH)

if RESTORE:
    enc.load_state_dict(torch.load('%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__)))
    dec.load_state_dict(torch.load('%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__)))


# Get all the embeddings
Zs = np.zeros((len(train_data),NUM_BATCH, NUM_STYLE))
for b, (images, labels) in enumerate(train_data):
        if images.size()[0] == NUM_BATCH:
            images = images.view(-1, NUM_PIXELS)
            if CUDA:
                images = images.cuda()
            q = enc(images, num_samples=NUM_SAMPLES)
            z = q['z'].value.cpu().data.squeeze().numpy()
            Zs[b] = z
Zs = Zs.reshape(-1,NUM_STYLE)

f, axarr = plt.subplots(NUM_STYLE, NUM_STYLE, figsize=(30, 30), sharex=True)
f.suptitle(r'$Z \ Embeddings$' , fontsize=30)

for i in range(NUM_STYLE):
    axarr[NUM_STYLE-1,i].set_xlabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    axarr[i,0].set_ylabel(r'$\mathbf{z_{%d}}$' % i, fontsize=10)
    for j in range(NUM_STYLE):
        if i==j:
            axarr[j,i].hist(Zs[:,i], bins=40)
        else:
            axarr[j,i].scatter(Zs[:,i],Zs[:,j],alpha=0.5)

plt.savefig('%s/%s-%02d-embedding.png' % (IMAGES_PATH, MODEL_NAME, NUM_STYLE), dpi=300)

def vary_z2(index, zmin, zmax):
    f, axarr = plt.subplots(10,10,figsize=(10,10),sharey=True)
    f.suptitle(r'$\mathbf{z_{%d}} \ \  varying$' % index , fontsize=30)
    z_range = np.linspace(zmin,zmax,num=10)

    for i in range(10):
        for j in range(10):
            null_image = torch.zeros((1,784))
            z = torch.zeros((1,10))
            y_hot = torch.zeros((1,10))
            z[0,index] = z_range[i]
            y_hot[0,j] = 1
            if CUDA:
                z = z.cuda()
                y_hot = y_hot.cuda()
                null_image = null_image.cuda()
            q_null = {'z': z, 'y':y_hot}
            p = dec(null_image, q_null, num_samples=NUM_SAMPLES, batch_size=1)
            image = p['images']
            image = image.value.cpu().data.numpy().reshape(28,28)
            axarr[i,j].imshow(image)
            axarr[i,j].axis('off')

    return None

for style in range(NUM_STYLE):
    vary_z2(style, -3, 3)
    plt.savefig('%s/z%d-varying-%s-%02d-traversal.png' % (IMAGES_PATH, style, MODEL_NAME, NUM_STYLE), dpi=300)

def zi_vs_zj(z_index1, z_index2, zmin=3, zmax=3, num_z=10, digit=0):
    f, axarr = plt.subplots(num_z, num_z, figsize=(num_z, num_z), sharey=True)
    f.suptitle(r'$Digit: %s$' % digit , fontsize=30)
    z_range = np.linspace(zmin,zmax,num=num_z)

    for i in range(num_z):
        for j in range(num_z):
            null_image = torch.zeros((1, NUM_PIXELS))
            z = torch.zeros((1,NUM_STYLE))
            y_hot = torch.zeros((1,NUM_DIGITS))
            z[0,z_index1] = z_range[i]
            z[0,z_index2] = z_range[j]
            y_hot[0,digit] = 1
            if CUDA:
                null_image = null_image.cuda()
                z = z.cuda()
                y_hot = y_hot.cuda()
            q_null = {'z':z, 'y':y_hot}
            p = dec(null_image, q_null, num_samples=NUM_SAMPLES)
            image = p['images']
            pixels = int(np.sqrt(NUM_PIXELS))
            image = image.value.cpu().data.numpy().reshape(pixels,pixels)
            axarr[i,j].imshow(image)
            axarr[i,j].axis('off')
    f.text(0.52, 0.08, r'$\mathbf{z_{%d}}$' % z_index2, ha='center', fontsize=20)
    f.text(0.09, 0.5, r'$\mathbf{z_{%d}}$' % z_index1, va='center', rotation='vertical', fontsize=20)


z_index1 = 9
z_index2 = 2
for digit in range(NUM_DIGITS):
    zi_vs_zj(z_index1, z_index2, zmin=-3, zmax=3, num_z=10, digit=digit)
    plt.savefig('%s/z%d-z%d-digit-%d-varying-%s-%02d-traversal.png' % (IMAGES_PATH, z_index1, z_index2, digit, MODEL_NAME, NUM_STYLE), dpi=300)
