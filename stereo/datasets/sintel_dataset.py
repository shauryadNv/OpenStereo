
import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate
import glob


class SintelDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        # If no list file provided, auto-scan dataset structure to build data_list
        if len(self.data_list) == 0:
            self.data_list = self._scan_all_samples()
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def _scan_all_samples(self):
        samples = []
        # support both 'final' and 'clean' passes
        patterns = [
            os.path.join(self.root, 'training', 'final_left', '*', '*.png'),
            os.path.join(self.root, 'training', 'clean_left', '*', '*.png'),
        ]
        left_imgs = []
        for p in patterns:
            left_imgs.extend(glob.glob(p))
        for left_abs in left_imgs:
            rel_left = os.path.relpath(left_abs, self.root)
            if rel_left.startswith('training' + os.sep + 'final_left' + os.sep):
                rel_right = rel_left.replace('final_left', 'final_right')
            else:
                rel_right = rel_left.replace('clean_left', 'clean_right')
            # disparities location is common for both passes
            scene_and_frame = rel_left.split(os.sep)[2:]  # [scene, frame]
            rel_disp = os.path.join('training', 'disparities', *scene_and_frame)
            right_abs = os.path.join(self.root, rel_right)
            disp_abs = os.path.join(self.root, rel_disp)
            if os.path.exists(right_abs) and os.path.exists(disp_abs):
                samples.append([rel_left, rel_right, rel_disp])
        return samples

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        disp_img = self.disparity_read(disp_path)

        occ_path = disp_path.replace('disparities','occlusions')
        occ = Image.open(occ_path)
        occ = np.array(occ, dtype=np.float32)
        occ_mask = occ == 255.0

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': disp_img,
            'occ_mask': occ
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/Sintel', item[0])
            super_pixel_label = str(super_pixel_label)[:-len('.png')] + "_lsc_lbl.png"
            if not os.path.exists(os.path.dirname(super_pixel_label)):
                os.makedirs(os.path.dirname(super_pixel_label), exist_ok=True)
            if not os.path.exists(super_pixel_label):
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                cv2.imwrite(super_pixel_label, label.astype(np.uint16))
            super_pixel_label = cv2.imread(super_pixel_label, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            if super_pixel_label is None:
                img = cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR)
                lsc = cv2.ximgproc.createSuperpixelLSC(img, region_size=10, ratio=0.075)
                lsc.iterate(20)
                label = lsc.getLabels()
                super_pixel_label = label.astype(np.int32)
            else:
                super_pixel_label = super_pixel_label.astype(np.int32)
            sample['super_pixel_label'] = super_pixel_label

        sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
    
    def disparity_read(self,filename):
        f_in = np.array(Image.open(filename))
        d_r = f_in[:,:,0].astype('float64')
        d_g = f_in[:,:,1].astype('float64')
        d_b = f_in[:,:,2].astype('float64')

        disp = d_r * 4 + d_g / (2**6) + d_b / (2**14)
        return disp