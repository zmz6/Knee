import itertools
import os
import pickle
from functools import partial

import cv2

from torch.utils.data import Dataset
import solt.core as slc
import solt.data as sld
import solt.transforms as slt
import torchvision.transforms as transform
import numpy as np
import torch
from torch.utils.data.sampler import Sampler


def generate_mask(img_height, img_width, radius, center_x, center_y):
    y, x = np.ogrid[0:img_height, 0:img_width]
    # circle mask
    mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
    return mask


def generate_masks(img_height, img_width, radius, landmarks):
    masks = []
    for landmark in landmarks:
        mask = generate_mask(img_height, img_width, radius, landmark[0], landmark[1])
        masks.append(mask)
    return np.array(masks).astype(int)


class RandomGenerator(object):
    def __init__(self, output_size, downsample=1, sigma=3, radius=20, state='train'):
        self.output_size = output_size
        self.state = state
        self.sigma = sigma
        self.downsample = downsample

    def __call__(self, dc):
        if self.state == 'train':
            transforms = transform.Compose([
                # slt.ImageColorTransform(mode='rgb2gs'),
                # w, h
                # 在训练脊柱的时候进行了更改，之前的实验结果中可能是老版本的代码
                slt.ImageRandomBrightness(),
                slt.ImageRandomContrast(),
                slc.SelectiveStream([
                    slt.RandomScale(range_x=(0.8, 1.2), p=0.5),
                ]),
                slc.SelectiveStream([
                    slt.RandomRotate(rotation_range=(-20, 20), p=0.5),
                ]),
                slc.SelectiveStream([
                    slt.RandomFlip(p=0.5),
                ]),
                # slt.CropTransform((self.output_size[0], self.output_size[1]), crop_mode='c'),
                slt.ResizeTransform((self.output_size[0], self.output_size[1])),
                partial(self.solt2torchhm, downsample=self.downsample, sigma=self.sigma)
            ])
        else:
            transforms = transform.Compose([
                # slt.ImageColorTransform(mode='rgb2gs'),
                # w, h
                slt.ResizeTransform((self.output_size[0], self.output_size[1])),
                partial(self.solt2torchhm, downsample=self.downsample, sigma=self.sigma)
            ])
        img, target_hm, target_kp = transforms(dc)
        return img, target_hm, target_kp

    def convert_img(self, img):
        img = torch.from_numpy(img).float()
        if len(img.shape) == 2:
            img = img.unsqueeze(0)
        elif len(img.shape) == 3:
            img = img.transpose(0, 2).transpose(1, 2)
        # H  W  C ---> C  H  W ----> C W H
        return img

    def l2m(self, lm, shape, sigma):
        m = np.zeros(shape, dtype=np.uint8)

        if np.all(lm > 0) and lm[0] < shape[1] and lm[1] < shape[0]:
            x, y = np.meshgrid(np.linspace(-0.5, 0.5, m.shape[1]), np.linspace(-0.5, 0.5, m.shape[0]))
            mux = (lm[0] - m.shape[1] // 2) / 1. / m.shape[1]
            muy = (lm[1] - m.shape[0] // 2) / 1. / m.shape[0]
            s = sigma / 1. / m.shape[0]
            m = (x - mux) ** 2 / 2. / s ** 2 + (y - muy) ** 2 / 2. / s ** 2
            m = np.exp(-m)
            m -= m.min()
            m /= m.max()

        return m

    def numpy2tens(self, x: np.ndarray, dtype='f') -> torch.Tensor:
        x = x.squeeze()
        x = torch.from_numpy(x)
        if x.dim() == 2:  # CxHxW format
            x = x.unsqueeze(0)

        if dtype == 'f':
            return x.float()
        elif dtype == 'l':
            return x.long()
        else:
            raise NotImplementedError

    def generate_heatmap(self, landmarks, shape, sigma, scale=1):
        # shape = (w, h)
        num_landmarks = len(landmarks)
        sigmas = np.array([sigma] * num_landmarks)
        # 1.缩放标签的尺度
        landmarks_reshaped = landmarks.reshape((1, 1, num_landmarks, 2))
        aranges = [np.arange(s) for s in shape]
        grid = np.meshgrid(*aranges, indexing='xy')
        grid = np.stack(grid, axis=2)
        grids = np.stack([grid] * num_landmarks, axis=2).astype('float32')
        squared_distance = np.sum(np.power(grids - landmarks_reshaped, 2), axis=-1)
        sigmas_reshaped = sigmas.reshape((1, 1, num_landmarks))
        # scale 正则化系数
        # scale = scale / np.power(np.sqrt(2 * np.pi) * sigmas_reshaped, 2)
        heatmap = scale * np.exp(-squared_distance / (2 * np.power(sigmas_reshaped, 2)))
        channel_first_heatmap = np.transpose(heatmap, (2, 0, 1))
        return channel_first_heatmap

    def solt2torchhm(self, dc: sld.DataContainer, downsample=1, sigma=1.5, convert=True, scale_ld=False):
        global new_size
        img, landmarks = dc.data
        img = img.squeeze()
        h, w = img.shape[0], img.shape[1]
        target = None
        if sigma != 0:
            new_size = (w // downsample, h // downsample)

            # target = []
            # for i in range(landmarks.data.shape[0]):
            #     res = self.l2m(landmarks.data[i] // downsample, new_size, sigma)
            #     target.append(self.numpy2tens(res))
            #
            # target = torch.cat(target, 0)
            target = self.generate_heatmap(landmarks.data // downsample, new_size, sigma)
            target = self.numpy2tens(target)
            target = target / target.max()
            # c, h, w
            assert target.size(0) == landmarks.data.shape[0]
            assert target.size(1) == img.shape[0] // downsample
            assert target.size(2) == img.shape[1] // downsample

        if convert:
            img = self.convert_img(img)
        # the ground truth should stay in the image coordinate system.
        landmarks = torch.from_numpy(landmarks.data).float()
        if scale_ld:
            landmarks[:, 0] /= w
            landmarks[:, 1] /= h

        return img, target, landmarks


class BaseDataSets(Dataset):
    def __init__(self, base_dir=None, split='train', num=None, num_class=7, transform=None, cv=1):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.annotation = self.get_annotation()
        self.space = self.get_space()
        self.num_class = num_class

        if self.split == 'train':
            with open(self._base_dir + '/group/cv/{}/train.txt'.format(cv), 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'val':
            with open(self._base_dir + '/group/cv/{}/val.txt'.format(cv), 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]
        elif self.split == 'test':
            with open(self._base_dir + '/group/cv/{}/test.txt'.format(cv), 'r') as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace('\n', '')
                                for item in self.sample_list]

        if num is not None and self.split == "train":
            self.sample_list = self.sample_list[:num]

        all_sample_list = self.sample_list.copy()
        for index in self.sample_list:
            all_sample_list.append(index + '3cm')
        self.sample_list = all_sample_list

        print("The {} dataset total {} samples".format(self.split, len(self.sample_list)))

    def get_annotation(self):
        with open(os.path.join(self._base_dir, 'annotation'), 'rb') as f:
            anno_dict = pickle.load(f)
        return anno_dict

    def get_space(self):
        with open(os.path.join(self._base_dir, 'space'), 'rb') as f:
            space_dict = pickle.load(f)
        return space_dict

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        if "knee" in self._base_dir:
            image = cv2.imread(self._base_dir + "/{}.jpg".format(case))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(7, 7))
            image = clahe.apply(image)
        else:
            raise NotImplemented

        label, _ = self.annotation[case]
        label = label[:self.num_class]
        space = self.space[case]
        h, w = image.shape
        kpts_wrapped = sld.KeyPoints(label, h, w)
        dc = sld.DataContainer((image, kpts_wrapped), 'IP')
        img, target_hm, target_kp = self.transform(dc)
        img /= img.max()

        sample = {
            'img_index': case,
            'ori_h': h,
            'ori_w': w,
            'ori_kp': label,
            'image': img,
            'space': space,
            'target_hm': target_hm,
            'target_kp': target_kp,
            "idx": idx,
        }

        return sample


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                   grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
