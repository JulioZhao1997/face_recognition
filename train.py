import argparse
import logging
import os
import sys
import random
import threading
import time

import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torchvision import models
from tensorboardX import SummaryWriter

from data.config import *
from data.make_loader import make_data_loader
from model.model_maker import *
from preprocess import *
from loss.make_loss import make_creterion
from engine.trainer import do_train
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Face Recognition with Multiple Models and LossFunctions')

    parser.add_argument('--dataset', default = 'CASIA', choices = ['CASIA'])
    parser.add_argument('--datapath', default = '', help = 'path of data', type = str)
    parser.add_argument('--loss', default = 'arc', choices = ['arc', 'sphere', 'normal'])
    parser.add_argument('--cuda', default = False, help = 'whether to use gpu', type = bool)
    parser.add_argument('--batch_size', default = 512, help = 'bacth size', type = int)
    parser.add_argument('--resume', default = None, help = 'path of model to resume training', type = str)
    parser.add_argument('--num_workers', default = 4, help = 'num_workers', type = int)
    parser.add_argument('--save_step', default = 5000, help = 'step to save model', type = int)
    parser.add_argument('--eval_step', default = 5000, help = 'step to eval model', type = int)
    parser.add_argument('--output_dir', default = 'output/', help = 'path to save models and logging files', type = str)
    parser.add_argument('--use_tensorboard', default = False, help = 'whether to user use_tensorboard', type = bool)
    parser.add_argument('--tensorboard_logs_path', default = 'tb_logs', help = 'path to tensorboard logs', type = str)
    parser.add_argument('--lr', '--learning-rate', default = 0.1, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default = 0.9, type=float, help='Momentum value for optim')
    parser.add_argument('--weight_decay', default = 5e-4, type=float, help='Weight decay for SGD')
    args = parser.parse_args()

    if args.loss == 'arc':
        cfg = arcface
    if args.loss == 'sphere':
        cfg = sphereface
    elif args.loss == 'normal':
        cfg = normal

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    if args.use_tensorboard:
        if not os.path.exists(args.tensorboard_logs_path):
            os.mkdir(args.tensorboard_logs_path)
		#else:
			#clean_dir(args.tensorboard_logs_path)
    #dataloader
    train_loader, test_loader = make_data_loader(args.dataset, args.datapath, preprocess_all[cfg['preprocess']], args.num_workers, args.batch_size,cfg['ratio'])

    logger = setup_logger('face_recognition', args.output_dir)
    logger.info('pid {}, {}'.format(threading.currentThread().ident, args))
    logger.info('pid {} running with config {}'.format(threading.currentThread().ident, cfg))

    #model
    creterion = make_creterion(cfg, args)
    model = make_model(cfg['model']['name'], cfg['model']['layers'], cfg['num_embeddings'], cfg['num_class'])
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.resume is not None:
        model, optimizer, cfg['start_iter'] = load_model(model, optimizer, args)
#    else:
#        model.backbone.apply(weights_init)

    do_train(model, creterion, optimizer, train_loader, test_loader, args, cfg)
