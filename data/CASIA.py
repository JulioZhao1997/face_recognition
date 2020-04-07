from PIL import Image
from torch.utils.data import Dataset

class CASIADataset(Dataset):
    def __init__(self, pic_list, label_list, preprocess):
        self.pic_list = pic_list
        self.label_list = label_list
        self.len = len(pic_list)
        self.preprocess = preprocess

    def __getitem__(self, i):
        img_path, y = self.pic_list[i], self.label_list[i]
        return self.preprocess(Image.open(self.pic_list[i])), y

    def __len__(self):
        return self.len
