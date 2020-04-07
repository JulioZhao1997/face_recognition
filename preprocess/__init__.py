import torchvision.transforms as transforms

preprocess_normal = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean = [127.5, 127.5, 127.5], std = [128, 128, 128])
])

preprocess_all = {
    'normal' : preprocess_normal
}
