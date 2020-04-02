'''
@File: simple_multiscale_deepdream.py
@Author: Luyufan
@Date: 2020/4/1
@Desc: Main Module
'''
import torch,os
from PIL import Image
import utils,config
from model import DeepDreamNet
import matplotlib.pyplot as plt
import tqdm
import numpy as np

os.environ["TORCH_HOME"] = config.pretrained_dir

torch.autograd.set_detect_anomaly(True)

def match_features_product_loss(output_feature,dst_feature,device):
    """Compute matched loss"""
    if dst_feature is None:
        return output_feature.data

    output_fea = output_feature.squeeze(0)
    dst_fea = dst_feature.squeeze(0)
    c,h,w = output_fea.size()
    output_fea = output_fea.view(c,-1)
    output_fea = output_fea.cpu().detach().numpy().copy()
    dst_fea = dst_fea.view(c,-1)
    dst_fea = dst_fea.cpu().detach().numpy().copy()
    dot_matrix = np.matmul(output_fea.T,dst_fea)
    return torch.from_numpy(
        dst_fea[:, dot_matrix.argmax(1)].reshape(c, h, w))\
        .unsqueeze(0).to(device)

def dream_process(input_img_np,target_img_np,model,lr,mode,iteration,epoch, device):
    """Dreaming Iteration Process"""
    _, _, h, w = input_img_np.shape
    if h > 400 and w > 400:
        deep = 8
    elif h > 300 and w > 300:
        deep = 6
    elif h > 200 and w > 200:
        deep = 4
    else:
        deep = 2

    for _ in tqdm.tqdm(range(iteration),desc='Epoch ' + str(epoch)+ ' :'):
        input_tensor = torch.from_numpy(input_img_np).type(dtype=torch.float32)
        input_tensor = input_tensor.to(device)
        input_tensor.requires_grad = True

        model.zero_grad()

        out_feature = model(input_tensor)
        dst_feature = None

        if target_img_np is not None:
            guide_tensor = torch.from_numpy(target_img_np).type(dtype=torch.float32)
            guide_tensor = guide_tensor.to(device)
            guide_tensor.requires_grad = False
            dst_feature = model(guide_tensor)

        matched_data = match_features_product_loss(out_feature, dst_feature,device)
        out_feature.backward(matched_data)

        # Update input tensor with different mode.
        if mode == "lapnorm":
            normed_grad = utils.normalize_grad(input_tensor.grad.data, deep=deep, device=device)
            input_tensor.data.add_(lr * normed_grad)
        else:
            avg_grad = np.abs(input_tensor.grad.data.cpu().numpy()).mean()
            norm_lr = lr / avg_grad
            input_tensor.data.add_(input_tensor.grad.data * norm_lr)

        input_tensor.grad.data.zero_()

        # Convert to numpy for clipping, It is not differentiable.
        input_img_np = input_tensor.cpu().detach().numpy()
        input_img_np = utils.clip(input_img_np)

    return input_img_np

if __name__ == "__main__":

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Preprocess
    input_img = Image.open(config.input_img_path)
    input_img_tensor = utils.origin_img_input_transform(input_img).unsqueeze(0)
    guide_img_tensor = None
    if config.guide:
        guide_img = Image.open(config.guide_img_path)
        guide_img_tensor = utils.origin_img_input_transform(guide_img).unsqueeze(0)

    # Model Definition
    deep_dreamer_net = DeepDreamNet()
    deep_dreamer_net = deep_dreamer_net.to(device)
    deep_dreamer_net.eval()
    deep_dreamer_net.fixParams()

    # Small Images
    input_small_imgs_np_list = utils.prepare_small_imgs(input_img_tensor,config.small_img_num,config.small_zoom_ratio)
    guide_small_imgs_np_list = utils.prepare_small_imgs(guide_img_tensor,config.small_img_num,config.small_zoom_ratio)

    # Training
    dreaming_detail_np = np.zeros_like(input_small_imgs_np_list[-1])
    output_img_np = dreaming_detail_np
    for index,base_img_np in enumerate(tqdm.tqdm(input_small_imgs_np_list[::-1],desc="Processing")):

        if index > 0:
            zoom = np.array(base_img_np.shape) / np.array(dreaming_detail_np.shape)
            dreaming_detail_np = utils.zooming_img(dreaming_detail_np,zoom)

        input_img_np = base_img_np + dreaming_detail_np
        iteration = (config.each_iteration + 10) if (index == len(input_small_imgs_np_list) -1) else config.each_iteration
        guide_img_np = guide_small_imgs_np_list[len(input_small_imgs_np_list)-1-index]
        output_img_np = dream_process(input_img_np,guide_img_np,deep_dreamer_net,config.lr,config.mode,iteration,index,device)
        dreaming_detail_np = output_img_np - base_img_np

    # Display
    dreamed_img = utils.deprocess(output_img_np)
    plt.figure()
    plt.imshow(dreamed_img)
    plt.show()
    plt.imsave(config.output_img_path,dreamed_img)
