'''
@File: config.py
@Author: Luyufan
@Date: 2020/3/31
@Desc: Common Configuration
'''

pretrained_model = 'resnet50'
pretrained_dir = "./pretrained_model/"

pretrained_mean =(0.485, 0.456, 0.406)
pretrained_std =(0.229, 0.224, 0.225)

small_img_num = 16
small_zoom_ratio = 1.3
each_iteration = 30
lr = 0.04
guide = True

# lapnorm or multiscale
mode = "lapnorm"

input_img_path = "./input/test.jpg"
guide_img_path = "./input/guide.jpg"
output_img_path = "./output/output-guide-lapnorm.jpg"


