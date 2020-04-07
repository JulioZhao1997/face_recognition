import torch
import torch.nn as nn

class l2norm(nn.Module):
    def __init__(self, num_embeddings, num_class):
        super(l2norm, self).__init__()
        self.fc = nn.Linear(num_embeddings, num_class, bias = False)
        self.num_embeddings = num_embeddings
        self.num_class = num_class

    def normalize(self, x):
        '''
            normalize the weight of W and x
        '''
        n_w = torch.norm(self.fc.weight, p = 2, dim = 1)
        n_x = torch.norm(x, p = 2, dim = 1)
        n_w = n_w.reshape(n_w.shape[0], -1).expand(self.num_class, self.num_embeddings)
        n_x = n_x.reshape(n_x.shape[0], -1).expand(x.shape[0], self.num_embeddings)
        self.fc.weight = torch.nn.Parameter(torch.div(self.fc.weight, n_w))
        x = torch.div(x, n_x)
        return x

    def forward(self, x):
        x = self.normalize(x)
        x = self.fc(x)
        return x

