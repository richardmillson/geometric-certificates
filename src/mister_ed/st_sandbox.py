# Universal import block
# Block to get the relative imports working
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.autograd import Variable

import mister_ed.adversarial_attacks as aa
import mister_ed.adversarial_evaluation as adveval
import mister_ed.adversarial_training as advtrain
import mister_ed.cifar10.cifar_loader as cifar_loader
import mister_ed.cifar10.cifar_resnets as cifar_resnets
import mister_ed.loss_functions as lf
import mister_ed.prebuilt_loss_functions as plf
import mister_ed.utils.checkpoints as checkpoints
import mister_ed.utils.image_utils as img_utils
import mister_ed.utils.pytorch_utils as utils
from mister_ed import config

# Load up dataLoader, classifier, normer
use_gpu = torch.cuda.is_available()
classifier_net = cifar_loader.load_pretrained_cifar_resnet(flavor=32,
                                                           use_gpu=use_gpu)
classifier_net.eval()

val_loader = cifar_loader.load_cifar_data('val',
                                          normalize=False,
                                          use_gpu=use_gpu)

cifar_normer = utils.DifferentiableNormalize(mean=config.CIFAR10_MEANS,
                                             std=config.CIFAR10_STDS)

examples, labels = next(iter(val_loader))

# build loss fxn and attack object
loss_fxn = plf.VanillaXentropy(classifier_net, normalizer=cifar_normer)

spatial_attack = aa.SpatialPGDLp(classifier_net, cifar_normer, loss_fxn, 'inf')

outputs = spatial_attack.attack(examples, labels, 0.1, 20, verbose=True)
