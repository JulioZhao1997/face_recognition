import random
import logging
import torch
import os
import sys

import torch.nn as nn

def get_pic_and_label(datapath):
    '''
        return the list of pics and correspoding label
    '''

    pic_list, label_list, class_to_idx = [], [], {}
    count = 0
    persons = os.listdir(datapath)
    for person in persons:
        if '.' in person:
            continue
        class_to_idx[person] = count
        pics = os.listdir('{}/{}'.format(datapath, person))
        for pic in pics:
            if 'DS' in pic:
                continue
            pic_list.append('{}/{}/{}'.format(datapath, person, pic))
            label_list.append(count)
        count += 1
    #shuffle
    z = list(zip(pic_list, label_list))
    random.shuffle(z)
    pic_list[:], label_list[:] = zip(*z)

    return pic_list, label_list

def get_train_test_index(pic_list, ratio):
    '''
        get index of train pics and test pics
    '''
    len_train = int(len(pic_list) * (1 - ratio))
    train_index = random.sample(list(range(len(pic_list))), len_train)
    test_index = list(set(list(range(len(pic_list)))).difference(set(train_index)))

    return train_index, test_index

def save_model(model, optimizer, args, epoch):
    '''
        save model
    '''
    logger = setup_logger('save model', args.output_dir)
    model_path = os.path.join(args.output_dir, 'model_epoch{}.pth'.format(epoch))
    state = {'model' : model.state_dict(), 'optimizer' : optimizer, 'epoch' : epoch}
    torch.save(state, model_path)
    logger.info('save model {}'.format(model_path))

def load_model(model, optimizer, args):
    '''
        load model, if resume, update epoch
    '''
    logger = setup_logger('load model', args.output_dir)
    logger.info('load model {}'.format(args.resume))
    checkpoint = torch.load(model_dir)
    model.load_state_dict(checkpoint['mode'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    if args.resume is not None:
        epoch = checkpoint['epoch'] + 1
    else:
        epoch = 0
    return model, optimizer, epoch

def adjust_learning_rate(optimizer, epoch, cfg, args):
    '''
        adjust learning rate by *0.1
    '''
    logger = setup_logger('adjust learning rate', args.output_dir)
    lr = optimizer.state_dict()['param_groups'][0]['lr']
    if epoch in cfg['lr_steps']:
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr * 0.1
        logger.info('adjust learning rate to {}'.format(optimizer.state_dict()['param_groups'][0]['lr']))


def setup_logger(name, save_dir=None):
    '''
        logger
    '''
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    stream_handler = logging.StreamHandler(stream=sys.stdout)
    stream_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, 'log.txt'))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
    return logger

def clean_dir(dir):
    files = os.listdir(dir)
    for file in files:
        os.remove(os.path.abspath(file))

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
