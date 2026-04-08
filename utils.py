import os

import cv2
import numpy as np
from PIL import Image
import torch


def gen_noise(shape):
    noise = np.zeros(shape, dtype=np.uint8)
    ### noise
    noise = cv2.randn(noise, 0, 255)
    noise = np.asarray(noise / 255, dtype=np.uint8)
    noise = torch.tensor(noise, dtype=torch.float32)
    return noise


def save_images(img_tensors, img_names, save_dir, scores=None):
    os.makedirs(save_dir, exist_ok=True)
    for img_tensor, img_name in zip(img_tensors, img_names):
        tensor = (img_tensor.clone()+1)*0.5 * 255
        tensor = tensor.cpu().clamp(0,255)

        try:
            array = tensor.numpy().astype('uint8')
        except:
            array = tensor.detach().numpy().astype('uint8')

        if array.shape[0] == 1:
            array = array.squeeze(0)
            im = Image.fromarray(array)
        elif array.shape[0] == 3:
            array = array.swapaxes(0, 1).swapaxes(1, 2)
            im = Image.fromarray(array)

        if scores is not None and img_name in scores:
            from PIL import ImageDraw, ImageFont
            score = scores[img_name]
            im = im.convert('RGBA')
            draw = ImageDraw.Draw(im)
            draw.rounded_rectangle([10, 10, 170, 54], fill=(20, 20, 20, 200), radius=5)
            try:
                font = ImageFont.truetype("arial.ttf", 18)
            except:
                font = ImageFont.load_default()
            
            if score > 75:
                color = (72, 199, 142)
            elif score >= 50:
                color = (255, 190, 0)
            else:
                color = (231, 76, 60)
            
            draw.text((22, 24), f"Fit score: {score}/100", fill=color, font=font)
            im = im.convert('RGB')

        im.save(os.path.join(save_dir, img_name), format='JPEG')


def load_checkpoint(model, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise ValueError("'{}' is not a valid checkpoint path".format(checkpoint_path))
    model.load_state_dict(torch.load(checkpoint_path))


def apply_transfer(warped_c, warped_cm, c_orig, opt, img_name="preview.jpg"):
    if not hasattr(opt, 'transfer_mode'):
        return warped_c
        
    result = warped_c
    if opt.transfer_mode == "color":
        # Pure-PyTorch RGB to LAB conversion
        wc = (warped_c + 1) * 0.5
        wc_gamma = wc ** 2.2
        
        xyz_mat = torch.tensor([[0.4124, 0.3576, 0.1805],
                                [0.2126, 0.7152, 0.0722],
                                [0.0193, 0.1192, 0.9505]]).to(warped_c.device)
                                
        xyz = torch.einsum('ij,bjk->bik', xyz_mat, wc_gamma.view(wc_gamma.size(0), 3, -1))
        xyz = xyz.view(wc_gamma.shape)
        
        white_point = torch.tensor([0.9505, 1.0000, 1.0890]).to(warped_c.device).view(1, 3, 1, 1)
        xyz_norm = xyz / white_point
        f_xyz = torch.where(xyz_norm > 0.008856, xyz_norm ** (1/3), 7.787 * xyz_norm + 16/116)
        
        L = 116 * f_xyz[:, 1:2, :, :] - 16
        a = 500 * (f_xyz[:, 0:1, :, :] - f_xyz[:, 1:2, :, :])
        b = 200 * (f_xyz[:, 1:2, :, :] - f_xyz[:, 2:3, :, :])
        
        if getattr(opt, 'target_color', None):
            try:
                r, g, b_val = [float(x)/255.0 for x in opt.target_color.split(',')]
            except:
                r, g, b_val = 0.0, 0.0, 0.0
            tc = torch.tensor([[[[r]], [[g]], [[b_val]]]]).to(warped_c.device)
            tc_gamma = tc ** 2.2
            xyz_tc = torch.einsum('ij,bjk->bik', xyz_mat, tc_gamma.view(1, 3, -1)).view(tc_gamma.shape)
            xyz_norm_tc = xyz_tc / white_point
            f_xyz_tc = torch.where(xyz_norm_tc > 0.008856, xyz_norm_tc ** (1/3), 7.787 * xyz_norm_tc + 16/116)
            
            a_t = 500 * (f_xyz_tc[:, 0:1, :, :] - f_xyz_tc[:, 1:2, :, :])
            b_t = 200 * (f_xyz_tc[:, 1:2, :, :] - f_xyz_tc[:, 2:3, :, :])
            a = a_t.expand_as(a)
            b = b_t.expand_as(b)
            
        fy = (L + 16) / 116
        fx = a / 500 + fy
        fz = fy - b / 200
        
        xyz_new = torch.cat([
            torch.where(fx ** 3 > 0.008856, fx ** 3, (fx - 16/116) / 7.787),
            torch.where(fy ** 3 > 0.008856, fy ** 3, (fy - 16/116) / 7.787),
            torch.where(fz ** 3 > 0.008856, fz ** 3, (fz - 16/116) / 7.787)
        ], dim=1) * white_point
        
        xyz_mat_inv = torch.inverse(xyz_mat)
        rgb_new = torch.einsum('ij,bjk->bik', xyz_mat_inv, xyz_new.view(xyz_new.size(0), 3, -1)).view(xyz_new.shape)
        rgb_new = torch.clamp(rgb_new, 0.0, 1.0)
        rgb_new = rgb_new ** (1/2.2)
        
        result = rgb_new * 2 - 1
        
    elif opt.transfer_mode == "texture":
        if getattr(opt, 'texture_path', None) and os.path.exists(opt.texture_path):
            import torchvision.transforms as transforms
            tex = Image.open(opt.texture_path).convert('RGB')
            tex = transforms.Resize((getattr(opt, 'load_height', 1024), getattr(opt, 'load_width', 768)))(tex)
            texture = transforms.ToTensor()(tex).unsqueeze(0).to(warped_c.device) * 2 - 1
            
            cloth_mean = (warped_c * warped_cm).sum(dim=[2,3], keepdim=True) / (warped_cm.sum(dim=[2,3], keepdim=True) + 1e-6)
            cloth_var = (((warped_c - cloth_mean) ** 2) * warped_cm).sum(dim=[2,3], keepdim=True) / (warped_cm.sum(dim=[2,3], keepdim=True) + 1e-6)
            cloth_std = torch.sqrt(cloth_var + 1e-5)
            
            tex_mean = texture.mean(dim=[2,3], keepdim=True)
            tex_std = texture.std(dim=[2,3], keepdim=True)
            
            tex_t = (texture - tex_mean) / (tex_std + 1e-5) * cloth_std + cloth_mean
            blend = getattr(opt, 'texture_blend', 0.6)
            result = tex_t * blend + warped_c * (1 - blend)

    result = result * warped_cm + c_orig * (1 - warped_cm)
    save_images([result[0]], [img_name.split('.')[0] + "_cloth_preview.jpg"], os.path.join(opt.save_dir, "warped_cloth"))
    return result
