'''
@File: utils.py
@Author: Luyufan
@Date: 2020/3/31
@Desc: Utils Module
'''

import torchvision.transforms as transforms
import config,torch,numpy as np
import scipy.ndimage as nd
import model

origin_img_input_transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize(config.pretrained_mean, config.pretrained_std)
                                                 ])

def deprocess(image_np):
    image_np = image_np.squeeze().transpose(1, 2, 0)
    image_np = image_np * np.array(config.pretrained_std).reshape((1, 1, 3)) + np.array(config.pretrained_mean).reshape((1, 1, 3))
    image_np = np.clip(image_np * 255, 0.0, 255.0)
    image_np = np.array(image_np, dtype=np.uint8)
    return image_np

def clip(image_np):
    with torch.no_grad():
        image_np = torch.as_tensor(image_np)
        for c in range(3):
            m, s = config.pretrained_mean[c], config.pretrained_std[c]
            image_np[0, c] = torch.clamp(image_np[0, c], - m / s, (1 - m) / s)
        return image_np.numpy()

def zooming_img(image_arr,zoom):
    """zooming image"""
    return nd.zoom(image_arr,zoom=zoom,order=1)

def prepare_small_imgs(image_tensor,small_num,small_ratio):
    """get multiscale images"""
    assert small_num >= 1
    if image_tensor is None:
        return [None ] * small_num

    image_tensor = image_tensor.cpu().numpy()
    small_imgs = [image_tensor]
    zoom_img = image_tensor
    for i in range(small_num-1):
        zoom_img = zooming_img(zoom_img,zoom=(1,1,1.0 / small_ratio,1.0 / small_ratio))
        small_imgs.append(zoom_img)
    return small_imgs

def split_grad_lapnorm(grad_tensor,deep,device):
    """split gradient"""
    gaussian_dsampler = model.GaussianSampler().to(device)
    gaussian_usampler = model.GaussianSampler(False).to(device)
    hf_levels_tensor_list = []
    for i in range(deep):
        down_grad_tensor = gaussian_dsampler(grad_tensor)
        up_grad_tensor = gaussian_usampler(down_grad_tensor)
        # avoid the different shape
        zoom = np.array(grad_tensor.cpu().numpy().shape) / np.array(up_grad_tensor.cpu().numpy().shape)
        up_grad_tensor = nd.zoom(up_grad_tensor.cpu().numpy(),zoom)
        up_grad_tensor = torch.from_numpy(up_grad_tensor).to(device)
        hf_tensor = grad_tensor - up_grad_tensor
        hf_levels_tensor_list.append(hf_tensor)
        grad_tensor = down_grad_tensor
    hf_levels_tensor_list.append(grad_tensor)
    return hf_levels_tensor_list

def normalize_each_level(level_tensor,device):
    """normalize each level"""
    mean = torch.tensor(0).to(device)
    std = torch.sqrt(torch.mean(torch.mul(level_tensor,level_tensor) - mean)).to(device)
    level_tensor = (level_tensor - mean) / (std + torch.tensor(1e-8).to(device))
    return level_tensor

def lap_merge(hp_levels,device):
    """rebuild the gradient."""
    gaussian_usampler = model.GaussianSampler(False).to(device)
    base_grad_tensor = hp_levels[-1]
    hp_levels_tensor_list = hp_levels[:-1]

    for hp_level_tensor in hp_levels_tensor_list[::-1]:
        base_grad_tensor = gaussian_usampler(base_grad_tensor)
        # avoid the different shape
        zoom = np.array(hp_level_tensor.cpu().numpy().shape) / np.array(base_grad_tensor.cpu().numpy().shape)
        base_grad_tensor = nd.zoom(base_grad_tensor.cpu().numpy(),zoom)
        base_grad_tensor = torch.from_numpy(base_grad_tensor).to(device)
        base_grad_tensor = base_grad_tensor+hp_level_tensor

    return base_grad_tensor

def normalize_grad(grad, deep,device):
    """normalize gradient"""
    hp_levels_tensor_list = split_grad_lapnorm(grad,deep,device)
    for i in range(len(hp_levels_tensor_list)):
        hp_levels_tensor_list[i] = normalize_each_level(hp_levels_tensor_list[i],device)
    grad = lap_merge(hp_levels_tensor_list,device)
    return grad
