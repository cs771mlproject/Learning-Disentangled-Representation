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
RESTORE = False

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

if not RESTORE:
    mask = {}
    for e in range(NUM_EPOCHS):
        train_start = time.time()
        train_elbo, mask = train(train_data, enc, dec,
                                 optimizer, mask, LABEL_FRACTION)
        train_end = time.time()
        test_start = time.time()
        test_elbo, test_accuracy = test(test_data, enc, dec)
        test_end = time.time()

        print('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)' % (
                e, train_elbo, train_end - train_start,
                test_elbo, test_accuracy, test_end - test_start))
        f = open("ouput_semisupervised_iwae.txt","a")
        f.write('[Epoch %d] Train: ELBO %.4e (%ds) Test: ELBO %.4e, Accuracy %0.3f (%ds)\n' % (
                e, train_elbo, train_end - train_start,
                test_elbo, test_accuracy, test_end - test_start))
        f.close()
    if not os.path.isdir(WEIGHTS_PATH):
        os.mkdir(WEIGHTS_PATH)
    torch.save(enc.state_dict(),
               '%s/%s-%s-%s-enc.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
    torch.save(dec.state_dict(),
               '%s/%s-%s-%s-dec.rar' % (WEIGHTS_PATH, MODEL_NAME, probtorch.__version__, torch.__version__))
