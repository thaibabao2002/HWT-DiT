import os
import cv2
import numpy as np
import torch
import random
import pickle
from torch.utils.data import Dataset
from data_loader.builder import DATASETS
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F

ASPECT_RATIO_64 = {
    '2.00': [64.0, 32.0], '1.00': [64.0, 64.0], '0.666667': [64, 96], '0.5': [64, 128],
    '0.4': [64.0, 160.0], '0.33333': [64.0, 192.0], '0.2857': [64, 224], '0.25': [64.0, 256.0],
    '0.22222': [64.0, 288.0], '0.2': [64, 320]
}

text_path = {'train': 'data/IAM64_train_add.txt',
             'test': 'data/IAM64_test_add.txt'}
letters = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
style_len = 352


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


def get_transform(img_size, pad=True, augment=False):
    if not pad:
        resize = transforms.Resize((img_size[0], img_size[1]), interpolation=transforms.InterpolationMode.BICUBIC)
    else:
        resize = Resize_with_pad(w=img_size[1], h=img_size[0])
    trans = [resize]
    if augment:
        trans.append(transforms.RandomCrop(img_size))
    trans.append(transforms.ToTensor())
    return transforms.Compose(trans)


class Resize_with_pad:
    def __init__(self, w=320, h=32):
        self.w = w
        self.h = h

    def __call__(self, image):
        w_1, h_1 = image.size
        # check if the original and final aspect ratios are the same within a margin
        wp = int(h_1 * self.w / self.h - w_1)
        if wp > 0:
            # image = F.pad(image, [0, 0, wp, 0], [i[1] for i in image.getextrema()], padding_mode="constant")
            image = F.pad(image, [0, 0, wp, 0], 255, padding_mode="constant")
        return F.resize(image, [self.h, self.w])


@DATASETS.register_module()
class HWTDataset(Dataset):
    def __init__(self,
                 image_path,
                 style_path,
                 laplace_path,
                 type,
                 content_type='unifont',
                 max_len=10,
                 transform=None,
                 load_vae_feat=False,
                 vae_path=None,
                 **kwargs):
        self.load_vae_feat = load_vae_feat
        if self.load_vae_feat:
            self.vae_path = os.path.join(vae_path, type)
        self.transform = transform
        self.aspect_ratio = ASPECT_RATIO_64
        self.laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float
                                    ).to(torch.float32).view(1, 1, 3, 3).contiguous()
        self.ratio_nums = {}
        self.ratio_index = {}
        self.max_len = max_len
        self.style_len = style_len
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.image_path = os.path.join(image_path, type)
        self.style_path = os.path.join(style_path, type)
        self.laplace_path = os.path.join(laplace_path, type)
        for k, v in self.aspect_ratio.items():
            self.ratio_index[float(k)] = []  # used for self.getitem
            self.ratio_nums[float(k)] = 0  # used for batch-sampler
        self.data_dict = self.load_data(text_path[type])
        print(self.ratio_index)
        print(self.ratio_nums)
        self.indices = list(self.data_dict.keys())
        self.con_symbols = self.get_symbols(content_type)
        # self.transform = get_transform((64, 320))
    def load_data(self, data_path):
        with open(data_path, 'r') as f:
            train_data = f.readlines()
            train_data = [i.strip().split(' ') for i in train_data]
            full_dict = {}
            idx = 0
            for idxx, i in enumerate(train_data):
                s_id = i[0].split(',')[0]
                image = i[0].split(',')[1] + '.png'
                transcription = i[1]
                ori_h = int(i[2])
                ori_w = int(i[3])
                closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
                self.ratio_nums[closest_ratio] += 1
                if len(self.ratio_index[closest_ratio]) == 0:
                    self.ratio_index[closest_ratio].append(idxx)
                if len(transcription) > self.max_len:
                    continue
                full_dict[idx] = {'image': image, 's_id': s_id, 'label': transcription, 'h': ori_h, 'w': ori_w}
                idx += 1
        return full_dict

    def __len__(self):
        return len(self.indices)

    def get_style_ref(self, wr_id):
        style_list = os.listdir(os.path.join(self.style_path, wr_id))
        style_index = random.sample(range(len(style_list)), 2)  # anchor and positive
        style_images = [cv2.imread(os.path.join(self.style_path, wr_id, style_list[index]), flags=0)
                        for index in style_index]
        laplace_images = [cv2.imread(os.path.join(self.laplace_path, wr_id, style_list[index]), flags=0)
                          for index in style_index]

        height = style_images[0].shape[0]
        assert height == style_images[1].shape[0], 'the heights of style images are not consistent'
        max_w = max([style_image.shape[1] for style_image in style_images])

        '''style images'''
        style_images = [style_image / 255.0 for style_image in style_images]
        new_style_images = np.ones([2, height, max_w], dtype=np.float32)
        new_style_images[0, :, :style_images[0].shape[1]] = style_images[0]
        new_style_images[1, :, :style_images[1].shape[1]] = style_images[1]

        '''laplace images'''
        laplace_images = [laplace_image / 255.0 for laplace_image in laplace_images]
        new_laplace_images = np.zeros([2, height, max_w], dtype=np.float32)
        new_laplace_images[0, :, :laplace_images[0].shape[1]] = laplace_images[0]
        new_laplace_images[1, :, :laplace_images[1].shape[1]] = laplace_images[1]
        return new_style_images, new_laplace_images

    def get_symbols(self, input_type):
        with open(f"data/{input_type}.pickle", "rb") as f:
            symbols = pickle.load(f)

        symbols = {sym['idx'][0]: sym['mat'].astype(np.float32) for sym in symbols}
        contents = []
        for char in self.letters:
            symbol = torch.from_numpy(symbols[ord(char)]).float()
            contents.append(symbol)
        contents.append(torch.zeros_like(contents[0]))  # blank image as PAD_TOKEN
        contents = torch.stack(contents)
        return contents

    def get_data_info(self, index):
        ori_h, ori_w = self.data_dict[self.indices[index]]['h'], self.data_dict[self.indices[index]]['w']
        return {'height': ori_h, 'width': ori_w}

    def getdata(self, index):
        image_name = self.data_dict[self.indices[index]]['image']
        label = self.data_dict[self.indices[index]]['label']
        wr_id = self.data_dict[self.indices[index]]['s_id']
        ori_h, ori_w = self.data_dict[self.indices[index]]['h'], self.data_dict[self.indices[index]]['w']

        closest_size, closest_ratio = get_closest_ratio(ori_h, ori_w, self.aspect_ratio)
        self.closest_ratio = closest_ratio
        closest_size = list(map(lambda x: int(x), closest_size))
        if closest_size[0] / ori_h > closest_size[1] / ori_w:
            resize_size = closest_size[0], int(ori_w * closest_size[0] / ori_h)
        else:
            resize_size = int(ori_h * closest_size[1] / ori_w), closest_size[1]
        if self.load_vae_feat:
            feature_name = image_name.replace(".png", ".npy")
            vae_path_idx = os.path.join(self.vae_path, wr_id, feature_name)
            image = np.load(vae_path_idx)
            image = torch.from_numpy(image)
        else:
            self.transform = T.Compose([
                T.Resize(resize_size, interpolation=InterpolationMode.BICUBIC),  # Image.BICUBIC
                T.CenterCrop(closest_size),
                T.ToTensor(),
                T.Normalize([.5], [.5]),
            ])
            img_path = os.path.join(self.image_path, wr_id, image_name)
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)

        style_ref, laplace_ref = self.get_style_ref(wr_id)
        style_ref = torch.from_numpy(style_ref).to(torch.float32)
        laplace_ref = torch.from_numpy(laplace_ref).to(torch.float32)
        return {'img': image,
                'content': label,
                'style': style_ref,
                "laplace": laplace_ref,
                'wid': int(wr_id),
                'transcr': label,
                'img_hw': torch.tensor([ori_h, ori_w], dtype=torch.float32),
                'aspect_ratio': closest_ratio,
                'image_name': image_name}

    def __getitem__(self, idx):
        # return self.getdata(idx)
        for _ in range(20):
            try:
                return self.getdata(idx)
            except Exception as e:
                print(f"Error details: {str(e)}")
                idx = random.choice(self.ratio_index[self.closest_ratio])
        raise RuntimeError('Too many bad data_loader.')

    def collate_fn_(self, batch):
        width = [item['img'].shape[2] for item in batch]
        c_width = [len(item['content']) for item in batch]
        s_width = [item['style'].shape[2] for item in batch]

        transcr = [item['transcr'] for item in batch]
        target_lengths = torch.IntTensor([len(t) for t in transcr])
        image_name = [item['image_name'] for item in batch]

        if max(s_width) < self.style_len:
            max_s_width = max(s_width)
        else:
            max_s_width = self.style_len
        imgs = torch.ones([len(batch), batch[0]['img'].shape[0], batch[0]['img'].shape[1], max(width)],
                          dtype=torch.float32)
        content_ref = torch.zeros([len(batch), max(c_width), 16, 16], dtype=torch.float32)

        style_ref = torch.ones([len(batch), batch[0]['style'].shape[0], batch[0]['style'].shape[1], max_s_width],
                               dtype=torch.float32)
        laplace_ref = torch.zeros([len(batch), batch[0]['laplace'].shape[0], batch[0]['laplace'].shape[1], max_s_width],
                                  dtype=torch.float32)
        target = torch.zeros([len(batch), max(target_lengths)], dtype=torch.int32)
        # imgs = [item['img'] for item in batch]
        # imgs = torch.stack(imgs)
        img_hws = [item['img_hw'] for item in batch]
        img_hws = torch.stack(img_hws)

        aspect_ratios = torch.tensor([item['aspect_ratio'] for item in batch], dtype=torch.float32)

        for idx, item in enumerate(batch):
            try:
                imgs[idx, :, :, 0:item['img'].shape[2]] = item['img']
            except:
                print('img', item['img'].shape)
            try:
                content = [self.letter2index[i] for i in item['content']]
                content = self.con_symbols[content]
                content_ref[idx, :len(content)] = content
            except:
                print('content', item['content'])

            target[idx, :len(transcr[idx])] = torch.Tensor([self.letter2index[t] for t in transcr[idx]])

            try:
                if max_s_width < self.style_len:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style']
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace']
                else:
                    style_ref[idx, :, :, 0:item['style'].shape[2]] = item['style'][:, :, :self.style_len]
                    laplace_ref[idx, :, :, 0:item['laplace'].shape[2]] = item['laplace'][:, :, :self.style_len]
            except:
                print('style', item['style'].shape)

        wid = torch.tensor([item['wid'] for item in batch])
        content_ref = 1.0 - content_ref  # invert the image
        return {'img': imgs, 'style': style_ref, 'content': content_ref, 'wid': wid, 'laplace': laplace_ref,
                'target': target, 'target_lengths': target_lengths, 'img_hw': img_hws, 'aspect_ratio': aspect_ratios,
                'image_name': image_name}


"""prepare the content image during inference"""

class ContentData(HWTDataset):
    def __init__(self, content_type='unifont') -> None:
        self.letters = letters
        self.letter2index = {label: n for n, label in enumerate(self.letters)}
        self.con_symbols = self.get_symbols(content_type)

    def get_content(self, label):
        word_arch = [self.letter2index[i] for i in label]
        content_ref = self.con_symbols[word_arch]
        content_ref = 1.0 - content_ref
        return content_ref.unsqueeze(0)