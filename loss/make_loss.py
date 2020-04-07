from .arc_loss import *
from .sphere_loss import *
import torch.nn as nn

def make_creterion(cfg, args):
    if args.loss == 'arc':
        return arc_loss(cfg['num_class'], cfg['margin'], cfg['scale'], args.cuda)
    elif args.loss == 'sphere':
        return sphere_loss(cfg['m'], cfg['scale'], cfg['num_class'], args.cuda)
    elif args.loss == 'normal':
        return nn.CrossEntropyLoss()
