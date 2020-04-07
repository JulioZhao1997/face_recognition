from .resnet import *

def make_model(model, num_layers, num_embeddings, num_class):
    if model == 'resnet':
        model = make_resnet(num_layers, num_embeddings, num_class)
        return model
