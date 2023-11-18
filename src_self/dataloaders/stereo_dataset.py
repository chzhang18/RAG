import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from dataloaders.data_io import get_transform, read_all_lines, pfm_imread


class StereoDataset(Dataset):
    def __init__(self, t, list_filenames, training):
        self.training = training
        self.t = t
        self.left_filenames, self.right_filenames, self.disp_filenames, self.disp_right_filenames = self.load_path(list_filenames[self.t])

    def load_path(self, list_filename):
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

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        data, scale = pfm_imread(filename)
        data = np.ascontiguousarray(data, dtype=np.float32)
        return data

    def load_img_disp(self, filename):
        data = Image.open(filename)
        #data = np.array(data, dtype=np.float32) / 256.
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        
        left_img = self.load_image(os.path.join(self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.right_filenames[index]))
        disparity = self.load_img_disp(os.path.join(self.disp_filenames[index]))

        
        if self.disp_right_filenames:
            disparity_right = self.load_img_disp(os.path.join(self.disp_right_filenames[index]))
        else:
            disparity_right = None


        # cityscapes resize half
        flag = 1
        if left_img.width > 1800:
            left_img = left_img.resize( (1024, 512), Image.ANTIALIAS)
            right_img = right_img.resize( (1024, 512), Image.ANTIALIAS)
            disparity = disparity.resize( (1024, 512), Image.ANTIALIAS)
            flag = 0

        if flag == 1:
            disparity = np.array(disparity, dtype=np.float32) / 256.
            if self.disp_right_filenames:
                disparity_right = np.array(disparity_right, dtype=np.float32) / 256.
        else:
            disparity = np.array(disparity, dtype=np.float32) / 256. / 2



        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 384, 192
            
            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            if self.disp_right_filenames:
                disparity_right = disparity_right[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform()
            left_img = processed(left_img)
            right_img = processed(right_img)

            if self.disp_right_filenames:
                return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "disparity_right":disparity_right,
                    "left_filename": self.left_filenames[index]}
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

            
            top_pad = 576-h #480 - h
            right_pad = 1248-w #960 - w
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
