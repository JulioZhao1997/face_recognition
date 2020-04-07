from .CASIA import *
#import CASIA

#sys.path.append('..')
from utils import * 
from preprocess import *

from torch.utils.data import DataLoader

def make_data_loader(dataset, datapath, preprocess, num_workers, batch_size, ratio = 0.1):
    '''
    return dataloader accroding to dataset type and preprocess
    if needed, using kfold
    '''

    if dataset == 'CASIA':
        if ratio != 0:
            pic_list, label_list = get_pic_and_label(datapath)
            train_index, test_index = get_train_test_index(pic_list, ratio)

            train_dataset = CASIADataset(pic_list = [pic_list[i] for i in train_index], label_list = [label_list[i] for i in train_index], preprocess = preprocess)
            test_dataset = CASIADataset(pic_list = [pic_list[i] for i in test_index], label_list = [label_list[i] for i in test_index], preprocess = preprocess)
            train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)
            test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

            return train_loader, test_loader

        else:
            index = list(range(len(pic_list)))
            dataset = CASIADataset(pic_list = pic_list, label_list = label_list, index = index, preprocess = preprocess)
            loader = DataLoader(dataset = dataset, batch_size = batch_size, shuffle = True, num_workers = num_workers)

            return loader
