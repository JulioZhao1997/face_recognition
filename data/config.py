arcface = {
    'num_embeddings' : 512,
    'num_class' : 10575,
    'margin' : 0.5,
    'scale' : 64,
    'ratio' : 0.1,
    'preprocess' : 'normal',
    'model' : {
        'name' : 'resnet',
        'layers' : 50
    },
    'lr_steps' : [20000, 28000],
    'max_epoch' : 32000,
    'start_epoch' : 0
}

sphereface = {
    'num_embeddings' : 512,
    'num_class' : 10575,
    'm' : 2,
    'scale' : 32,
    'ratio' : 0.1,
    'preprocess' : 'normal',
    'model' : {
        'name' : 'resnet',
        'layers' : 50
    },
    'lr_steps' : [20000, 28000],
    'max_epoch' : 32000,
    'start_epoch' : 0
}

normal = {
    'num_embeddings' : 512,
    'num_class' : 10575,
    'ratio' : 0.1,
    'preprocess' : 'normal',
    'model' : {
        'name' : 'resnet',
        'layers' : 50
    },
    'lr_steps' : [20000, 28000],
    'max_epoch' : 32000,
    'start_epoch' : 0
}
