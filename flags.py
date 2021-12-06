# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 10:34:37 2021

@author: padel
"""
import os
# os.chdir(r'C:\Users\padel\OneDrive\Desktop\improved_contrastive_divergence-master')

import tensorflow as tf
import numpy as np
import timeit
from tensorflow.python.platform import flags
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm
import time
from multiprocessing import Process
import pickle as pkl

# from datasets import Cifar10, CelebAHQ, Mnist, ImageNet, LSUNBed, STLDataset
#from models_real import ResNetModel, CelebAModel, MNISTModel, ImagenetModel #Uncomment for real use
import os.path as osp
from logger import TensorBoardOutputFormat
from utils import ReplayBuffer, ReservoirBuffer
from tqdm import tqdm
import random
from torch.utils.data import DataLoader
import time as time
from io import StringIO
from tensorflow.core.util import event_pb2
import torch
import numpy as np
from scipy.misc import imsave
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


from easydict import EasyDict

from utils import ReplayBuffer
from torch.optim import Adam, SGD
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP

def get_flags():
    FLAGS = flags.FLAGS
    
    # Distributed training hyperparameters
    flags.DEFINE_integer('nodes', 1,
        'number of nodes for training')
    flags.DEFINE_integer('gpus', 1,
        'number of gpus per nodes')
    flags.DEFINE_integer('node_rank', 0,
        'rank of node')
    
    # Configurations for distributed training
    flags.DEFINE_string('master_addr', '8.8.8.8',
        'address of communicating server')
    flags.DEFINE_string('port', '10002',
        'port of training')
    flags.DEFINE_bool('slurm', False,
        'whether we are on slurm')
    flags.DEFINE_bool('repel_im', True,
        'maximize entropy by repeling images from each other')
    flags.DEFINE_bool('hmc', False,
        'use the hamiltonian monte carlo sampler')
    flags.DEFINE_bool('square_energy', True,
        'make the energy square')
    flags.DEFINE_bool('alias', True,
        'make the energy square')
    
    flags.DEFINE_string('dataset','mnist',
        'cifar10 or celeba')
    flags.DEFINE_integer('batch_size', 128, 'batch size during training')
    flags.DEFINE_bool('multiscale', True, 'A multiscale EBM')
    flags.DEFINE_bool('self_attn', True, 'Use self attention in models')
    flags.DEFINE_bool('sigmoid', True, 'Apply sigmoid on energy (can improve the stability)')
    flags.DEFINE_bool('anneal', True, 'Decrease noise over Langevin steps')
    flags.DEFINE_integer('data_workers', 4,
        'Number of different data workers to load data in parallel')
    flags.DEFINE_integer('buffer_size', 10000, 'Size of inputs')
    
    # General Experiment Settings
    flags.DEFINE_string('logdir', 'cachedir',
        'location where log of experiments will be stored')
    flags.DEFINE_string('exp', 'default', 'name of experiments')
    flags.DEFINE_integer('log_interval', 10, 'log outputs every so many batches')
    flags.DEFINE_integer('save_interval', 1000,'save outputs every so many batches')
    flags.DEFINE_integer('test_interval', 1000,'evaluate outputs every so many batches')
    flags.DEFINE_integer('resume_iter', 0, 'iteration to resume training from')
    flags.DEFINE_bool('train', True, 'whether to train or test')
    flags.DEFINE_bool('transform', True, 'apply data augmentation when sampling from the replay buffer')
    flags.DEFINE_bool('kl', True, 'apply a KL term to loss')
    flags.DEFINE_bool('cuda', True, 'move device on cuda')
    flags.DEFINE_integer('epk', 0, 'epoch to resume training from')
    flags.DEFINE_integer('epoch_num', 50, 'Number of Epochs to train on')
    flags.DEFINE_integer('ensembles', 1, 'Number of ensembles to train models with')
    flags.DEFINE_float('lr', 2e-4, 'Learning for training')
    flags.DEFINE_float('kl_coeff', 1.0, 'coefficient for kl')
    flags.DEFINE_bool('noise', True, 'Decide if add noise to the input')
    
    # EBM Specific Experiments Settings
    flags.DEFINE_string('objective', 'cd', 'use the cd objective')
    
    # Setting for MCMC sampling
    flags.DEFINE_integer('num_steps', 40, 'Steps of gradient descent for training')
    flags.DEFINE_float('step_lr', 87.5, 'Size of steps for gradient descent')
    flags.DEFINE_bool('replay_batch', True, 'Use MCMC chains initialized from a replay buffer.')
    flags.DEFINE_bool('reservoir', True, 'Use a reservoir of past entires')
    flags.DEFINE_float('noise_scale', 1.,'Relative amount of noise for MCMC')
    
    # Architecture Settings
    flags.DEFINE_integer('filter_dim', 8, 'number of filters for conv nets')
    flags.DEFINE_integer('im_size', 32, 'size of images')
    flags.DEFINE_bool('spec_norm', True, 'Whether to use spectral normalization on weights')
    flags.DEFINE_bool('norm', True, 'Use group norm in models norm in models')
    
    # Conditional settings
    flags.DEFINE_bool('cond', True, 'conditional generation with the model')
    flags.DEFINE_bool('all_step', False, 'backprop through all langevin steps')
    flags.DEFINE_bool('log_grad', False, 'log the gradient norm of the kl term')
    flags.DEFINE_integer('cond_idx', 1, 'conditioned index')
    
    return FLAGS

def gen_image(label, FLAGS, model, im_neg, num_steps, sample=False):
    #gen_image(label, FLAGS, model, data_corrupt, num_steps)
    #Non allena il modello va solo su e giù e aggiusta l'immagine che fa merda
    im_noise = torch.randn_like(im_neg).detach()

    im_negs_samples = []

    for i in range(num_steps):
        im_noise.normal_()

        if FLAGS.anneal:
            im_neg = im_neg + 0.001 * (num_steps - i - 1) / num_steps * im_noise
        else:
            im_neg = im_neg + 0.001 * im_noise
        
        #Evaluate the energy of noised images
        im_neg.requires_grad_(requires_grad=True)
        energy = model.forward(im_neg, label, read_out=False)

        if FLAGS.all_step:
            im_grad = torch.autograd.grad([energy.sum()], [im_neg], create_graph=True)[0]
        else:
            #Compute the sum of the gradients of the outputs respect to the inputs
            im_grad = torch.autograd.grad([energy.sum()], [im_neg])[0]

        if i == num_steps - 1:
            im_neg_orig = im_neg  
            
            #Here there's the update
            im_neg = im_neg - FLAGS.step_lr * im_grad

            if FLAGS.dataset == "cifar10":
                n = 128
            elif FLAGS.dataset == "celeba":
                # Save space
                n = 128
            elif FLAGS.dataset == "lsun":
                # Save space
                n = 32
            elif FLAGS.dataset == "object":
                # Save space
                n = 32
            elif FLAGS.dataset == "mnist":
                n = 100000
            elif FLAGS.dataset == "imagenet":
                n = 32
            elif FLAGS.dataset == "stl":
                n = 32

            im_neg_kl = im_neg_orig[:n]
            if sample:
                pass
            else:
                energy = model.forward(im_neg_kl, label, read_out=False)
                im_grad = torch.autograd.grad([energy.sum()], [im_neg_kl], create_graph=True)[0]

            im_neg_kl = im_neg_kl - FLAGS.step_lr * im_grad[:n]
            im_neg_kl = torch.clamp(im_neg_kl, 0, 1)
        else:
            im_neg = im_neg - FLAGS.step_lr * im_grad

        im_neg = im_neg.detach()

        if sample:
            im_negs_samples.append(im_neg)

        im_neg = torch.clamp(im_neg, 0, 1)

    if sample:
        return im_neg, im_neg_kl, im_negs_samples, im_grad
    else:
        return im_neg, im_neg_kl, im_grad
