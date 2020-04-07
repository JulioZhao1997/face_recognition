from .evaluator import do_evaluation
from utils import save_model, adjust_learning_rate, setup_logger
from tqdm import tqdm
from tensorboardX import SummaryWriter

import os
import torch
import torch.nn as nn
import threading
import sys

def do_train(model, creterion, optimizer, train_loader, test_loader, args, cfg):
    logger = setup_logger('trainer', args.output_dir)
    if args.cuda:
        model = model.cuda()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
		
    if args.use_tensorboard:
        logger.info('path to tensorboard data {}'.format(os.path.abspath(args.tensorboard_logs_path)))
        writer = SummaryWriter(log_dir = args.tensorboard_logs_path)

    for epoch in range(cfg['start_epoch'], cfg['max_epoch']):
        #model.train()
        t = tqdm(train_loader, desc = 'epoch {}'.format(epoch))
        adjust_learning_rate(optimizer, epoch, cfg, args)
        loss_count = 0
        acc_sum, n = 0, 0
        for (i, (X, y)) in enumerate(t):
            if args.cuda:
                X, y = X.cuda(), y.cuda()
            y_hat = model(X)
            loss = creterion(y_hat, y)
            loss_count += float(loss)
            #print(loss_count, loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #train acc
            acc_sum += (y_hat.argmax(dim=1) == y).float().sum().cpu().item()
            n += X.shape[0]
        logger.info('pid {} epoch {}, loss {}, train acc {}'.format(threading.currentThread().ident, epoch, loss_count, acc_sum / n))
        if args.use_tensorboard:
            writer.add_scalar('data/loss', loss_count, epoch)
            writer.add_scalar('data/train_acc', acc_sum / n, epoch)
        if epoch % args.eval_step == 0 and epoch != 0:
            test_acc = do_evaluation(test_loader, model, args)
            if args.use_tensorboard:
                writer.add_scalar('data/test_acc', test_acc, epoch)
        if epoch % args.save_step == 0 and epoch != 0:
            save_model(model, optimizer, args, epoch)
