import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from dataloaders.data_io import get_transform, read_all_lines, pfm_imread


class SceneflowDrivingDataset(Dataset):
    def __init__(self, t, list_filenames, training):
        self.training = training
        self.t = t
        self.left_filenames, self.right_filenames, self.disp_filenames, self.disp_right_filenames = self.load_synthetic_path(list_filenames)

        train_realdata_list = ['./filenames/drivingstereo/drivingstereo_cloudy_train.txt', './filenames/drivingstereo/drivingstereo_foggy_train.txt', 
                           './filenames/drivingstereo/drivingstereo_rainy_train.txt', './filenames/drivingstereo/drivingstereo_sunny_train.txt']

        self.real_left_filenames = self.load_real_path(train_realdata_list[self.t])


    def load_synthetic_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        disp_images = [x[2] for x in splits]
        if len(splits[0]) == 3:  # right ground truth not available
            return left_images, right_images, disp_images, None
        else:
            disp_right_images = [x[3] for x in splits]
            return left_images, right_images, disp_images, disp_right_images

    def load_real_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        return left_images


    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_pfm_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def load_img_disp(self, filename):
        data = Image.open(filename)
        data = np.array(data, dtype=np.float32) / 256.
        return data

    def transfer_color(self, target, source):
        target = target.astype(float) / 255
        source = source.astype(float) / 255

        target_means = target.mean(0).mean(0)
        target_stds = target.std(0).std(0)

        source_means = source.mean(0).mean(0)
        source_stds = source.std(0).std(0)

        target -= target_means
        target /= target_stds / source_stds
        target += source_means

        target = np.clip(target, 0, 1)
        target = (target * 255).astype(np.uint8)

        return target

    def __len__(self):
        
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        left_img = self.load_image(os.path.join(self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.right_filenames[index]))
        disparity = self.load_pfm_disp(os.path.join(self.disp_filenames[index]))
        if self.disp_right_filenames:
            disparity_right = self.load_pfm_disp(os.path.join(self.disp_right_filenames[index]))
        else:
            disparity_right = None

        real_left_img = self.load_image(os.path.join(self.real_left_filenames[index%len(self.real_left_filenames)]))


        left_img = self.transfer_color(np.array(left_img), np.array(real_left_img))
        right_img = self.transfer_color(np.array(right_img), np.array(real_left_img))
        left_img = Image.fromarray(left_img)
        right_img = Image.fromarray(right_img)


        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 384, 192
            #crop_w, crop_h = 768, 384

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            if self.disp_right_filenames:
                disparity_right = disparity_right[h - crop_h:h, w - crop_w: w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            if self.disp_right_filenames:
                return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_right":disparity_right}
            else:
                return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size

            # normalize
            processed = get_transform()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()

            top_pad = 540 - h
            right_pad = 960 - w
            assert top_pad >= 0 and right_pad >= 0

            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}

