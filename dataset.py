# import torchvision.transforms.functional as F
import torch.nn.functional as F
import numpy as np
import random
import os
import PIL.ImageEnhance as ImageEnhance
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch


# class ToTensor(object):
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#         return {'image': F.to_tensor(image), 'label': F.to_tensor(label)}
#
# class Resize(object):#标签不resize
# #
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         return {'image': F.resize(image, (self.size[1], self.size[0]),interpolation=F.InterpolationMode.BILINEAR), 'label': label}

# class Resize(object):
#     def __init__(self, size):
#         self.size = size
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         # 使用双线性插值调整图像尺寸
#         image_resized = F.interpolate(image.unsqueeze(0), size=(self.size[1], self.size[0]), mode='bilinear', align_corners=False).squeeze(0)
#
#         # 使用最邻近插值调整标签尺寸
#         label_resized = F.interpolate(label.unsqueeze(0), size=(self.size[1], self.size[0]), mode='nearest').squeeze(0)
#
#         return {'image': image_resized, 'label': label_resized}

class Resize(object):
    def __init__(self, size):
        self.size = size  # [W, H]

    def __call__(self, data):
        image = data['image']
        label = data['label']

        image = image.resize((self.size[0], self.size[1]), Image.BILINEAR)
        label = label.resize((self.size[0], self.size[1]), Image.NEAREST)

        return {'image': image, 'label': label}

# class Normalize(object):#标签不归一化
#     def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
#         self.mean = mean
#         self.std = std
#
#     def __call__(self, sample):
#         image, label = sample['image'], sample['label']
#         image = F.normalize(image, self.mean, self.std)
#         return {'image': image, 'label': label}

# class RandomHorizontalFlip(object):
#     def __init__(self, p=0.5):
#         self.p = p
#
#     def __call__(self, data):
#         image, label = data['image'], data['label']
#
#         if random.random() < self.p:
#             return {'image': F.hflip(image), 'label': F.hflip(label)}
#
#         return {'image': image, 'label': label}



class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, data):
        im = data['image']
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        im = ImageEnhance.Brightness(im).enhance(r_brightness)
        im = ImageEnhance.Contrast(im).enhance(r_contrast)
        im = ImageEnhance.Color(im).enhance(r_saturation)
        data['image'] = im
        return data

class HorizontalFlip(object):
    def __init__(self, p=0.5, *args, **kwargs):
        self.p = p

    def __call__(self, data):
        if random.random() > self.p:
            return data
        else:
            im = data['image']
            lb = data['label']

            return {'image': im.transpose(Image.FLIP_LEFT_RIGHT), 'label': lb.transpose(Image.FLIP_LEFT_RIGHT)}


class RandomScale(object):
    def __init__(self, scales=(1, ), *args, **kwargs):
        self.scales = scales

    def __call__(self, data):
        im = data['image']
        lb = data['label']
        W, H = im.size
        scale = random.choice(self.scales)
        w, h = int(W * scale), int(H * scale)
        return {'image': im.resize((w, h), Image.BILINEAR), 'label': lb.resize((w, h), Image.NEAREST)}

class RandomCrop(object):
    def __init__(self, size, *args, **kwargs):
        self.size = size

    def __call__(self, data):
        im = data['image']
        lb = data['label']
        assert im.size == lb.size
        W, H = self.size
        w, h = im.size

        # if (W, H) == (w, h): return dict(im=im, lb=lb)
        if (W, H) == (w, h):
            return {'image': im, 'label': lb}

        if w < W or h < H:
            scale = float(W) / w if w < h else float(H) / h
            w, h = int(scale * w + 1), int(scale * h + 1)
            im = im.resize((w, h), Image.BILINEAR)
            lb = lb.resize((w, h), Image.NEAREST)
        sw, sh = random.random() * (w - W), random.random() * (h - H)
        crop = int(sw), int(sh), int(sw) + W, int(sh) + H
        return {'image': im.crop(crop), 'label': lb.crop(crop)}

class Compose(object):
    def __init__(self, do_list):
        self.do_list = do_list

    def __call__(self, data):
        for comp in self.do_list:
            data = comp(data)
        return data

class FullDataset(Dataset):
    def __init__(self, image_root, gt_root, cropsize, mode):

        # self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        # self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if
                       f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.mode = mode


        ## pre-processing
        self.img_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        self.gt_to_tensor = transforms.Compose([transforms.ToTensor()])
        self.trans_train = Compose([
            ColorJitter(
                brightness = 0.5,
                contrast = 0.5,
                saturation = 0.5),
            HorizontalFlip(),
            RandomScale((0.75, 1.0, 1.25, 1.5, 1.75, 2.0)),
            RandomCrop(cropsize)
            ])
        self.trans_val = Compose([])




    # def __getitem__(self, idx):
    #     image = self.rgb_loader(self.images[idx])
    #     name = self.images[idx].split('/')[-1]
    #     label = self.binary_loader(self.gts[idx])
    #
    #     data = {'image': image, 'label': label}
    #     if self.mode == 'train':
    #         data = self.trans_train(data)
    #
    #     data['image'] = self.img_to_tensor(data['image'])
    #     data['label'] = self.gt_to_tensor(data['label'])
    #     data['label'] = data['label']*255
    #     data['label'] = data['label'].to(torch.uint8)  # 将张量数据类型转换为 uint8
    #
    #     if self.mode == 'test':
    #        data['image']=data['image'][:, 22:, :]
    #        data['label']= data['label'][:, 22:, :]
    #        return data['image'], data['label'], name
    #     if self.mode == 'private_val':
    #        data['image']=data['image'][:, 22:, :]
    #        data['label']= data['label'][:, 22:, :]
    #     return data

    # def __getitem__(self, idx):
    #     image = self.rgb_loader(self.images[idx])
    #     name = os.path.basename(self.images[idx])
    #     label = self.binary_loader(self.gts[idx])
    #
    #     data = {'image': image, 'label': label}
    #
    #     if self.mode == 'train':
    #         data = self.trans_train(data)
    #     elif self.mode in ['private_val', 'test']:
    #         data = self.trans_val(data)  # 确保验证/测试也 resize
    #
    #     # 图像转为 Tensor，并归一化
    #     data['image'] = self.img_to_tensor(data['image'])  # (3,H,W), float
    #
    #     # 标签转换为 tensor（ADE20K 是 0~149 的整数像素值）
    #     data['label'] = torch.from_numpy(np.array(data['label'], dtype=np.int64))  # (H,W), long
    #
    #     if self.mode == 'test':
    #         return data['image'], data['label'], name
    #     else:
    #         return data

    def __getitem__(self, idx):
        image = self.rgb_loader(self.images[idx])
        label = self.binary_loader(self.gts[idx])
        name = os.path.basename(self.images[idx])


        data = {'image': image, 'label': label}

        if self.mode == 'train':
            data = self.trans_train(data)
        elif self.mode in ['private_val', 'test']:
            data = self.trans_val(data)

        data['image'] = self.img_to_tensor(data['image'])
        data['label'] = torch.from_numpy(np.array(data['label'], dtype=np.int64))



        if self.mode == 'test':
            return data['image'], data['label'], name
        else:
            return data


    def __len__(self):
        return len(self.images)

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


