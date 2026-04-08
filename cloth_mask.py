import os
from PIL import Image
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from networks.u2net import U2NET

def load_checkpoint_mgpu(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print("----No checkpoints at given path----")
        return
    model_state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    print("----checkpoints loaded from path: {}----".format(checkpoint_path))
    return model

class Normalize_image(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        self.normalize_1 = transforms.Normalize(self.mean, self.std)
        self.normalize_3 = transforms.Normalize([self.mean] * 3, [self.std] * 3)
        self.normalize_18 = transforms.Normalize([self.mean] * 18, [self.std] * 18)

    def __call__(self, image_tensor):
        if image_tensor.shape[0] == 1:
            return self.normalize_1(image_tensor)
        elif image_tensor.shape[0] == 3:
            return self.normalize_3(image_tensor)
        elif image_tensor.shape[0] == 18:
            return self.normalize_18(image_tensor)

def get_palette(num_cls):
    n = num_cls
    palette = [0] * (n * 3)
    for j in range(0, n):
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        lab = j
        while lab:
            palette[j * 3 + 0] = 255
            palette[j * 3 + 1] = 255
            palette[j * 3 + 2] = 255
            i += 1
            lab >>= 3
    return palette

def generate_mask(image_dir, result_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(result_dir, exist_ok=True)
    checkpoint_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'cloth_segm_u2net_latest.pth')

    transforms_list = [transforms.ToTensor(), Normalize_image(0.5, 0.5)]
    transform_rgb = transforms.Compose(transforms_list)

    net = U2NET(in_ch=3, out_ch=4)
    net = load_checkpoint_mgpu(net, checkpoint_path)
    net = net.to(device)
    net = net.eval()

    palette = get_palette(4)

    images_list = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])
    
    for image_name in images_list:
        img_path = os.path.join(image_dir, image_name)
        img = Image.open(img_path).convert('RGB')
        img_size = img.size
        img = img.resize((768, 768), Image.BICUBIC)
        image_tensor = transform_rgb(img)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        
        with torch.no_grad():
            output_tensor = net(image_tensor.to(device))
            output_tensor = F.log_softmax(output_tensor[0], dim=1)
            output_tensor = torch.max(output_tensor, dim=1, keepdim=True)[1]
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_tensor = torch.squeeze(output_tensor, dim=0)
            output_arr = output_tensor.cpu().numpy()

        output_img = Image.fromarray(output_arr.astype('uint8'), mode='L')
        output_img = output_img.resize(img_size, Image.BICUBIC)
        
        output_img.putpalette(palette)
        output_img = output_img.convert('L')
        
        base_name = os.path.splitext(image_name)[0]
        output_img.save(os.path.join(result_dir, base_name + '.jpg'))
