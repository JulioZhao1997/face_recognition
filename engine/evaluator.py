import torch
from utils import *

def do_evaluation(test_loader, model, args):
    logger = setup_logger('evaluation', args.output_dir)
    acc_sum, n = 0.0, 0
    model.eval()
    with torch.no_grad():
        for X, y in test_loader:
            if args.cuda:
                X, y = X.cuda(), y.cuda()
            acc_sum += (model(X).argmax(dim=1) == y).float().sum().cpu().item()
            n += y.shape[0]
    model.train()
    logger.info('evaluate accuracy {}'.format(acc_sum / n))
    return acc_sum / n