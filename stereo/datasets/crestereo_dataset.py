
import os
import numpy as np
from PIL import Image
from pathlib import Path
from .dataset_template import DatasetTemplate
import cv2
import glob


class CREStereoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        # If no list file provided, auto-scan dataset structure to build data_list
        if len(self.data_list) == 0:
            self.data_list = self._scan_all_samples()
        self.return_right_disp = self.data_info.RETURN_RIGHT_DISP
        if hasattr(self.data_info, 'RETURN_SUPER_PIXEL'):
            self.retrun_super_pixel = self.data_info.RETURN_SUPER_PIXEL
        else:
            self.retrun_super_pixel = False

    def _scan_all_samples(self):
        samples = []
        # Search for files like *_left.jpg and corresponding *_left.disp.png
        left_imgs = glob.glob(os.path.join(self.root, '**', '*_left.jpg'), recursive=True)
        for left_abs in left_imgs:
            rel_left = os.path.relpath(left_abs, self.root)
            rel_right = rel_left.replace('_left.jpg', '_right.jpg')
            rel_disp = rel_left.replace('_left.jpg', '_left.disp.png')
            right_abs = os.path.join(self.root, rel_right)
            disp_abs = os.path.join(self.root, rel_disp)
            if os.path.exists(right_abs) and os.path.exists(disp_abs):
                samples.append([rel_left, rel_right, rel_disp])
        return samples

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, left_disp_path = full_paths

        left_img = Image.open(left_path).convert('RGB')
        left_img = np.array(left_img, dtype=np.float32)

        right_img = Image.open(right_path).convert('RGB')
        right_img = np.array(right_img, dtype=np.float32)

        left_disp = cv2.imread(left_disp_path, cv2.IMREAD_UNCHANGED)
        left_disp = left_disp.astype(np.float32) / 32
        occ_mask = np.zeros_like(left_disp, dtype=bool)

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp,
            'occ_mask': occ_mask
        }

        if self.retrun_super_pixel:
            super_pixel_label = Path(self.root).parent.joinpath('SuperPixelLabel/CREStereo', item[0])
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

        if self.return_right_disp:
            right_disp_path = left_disp_path.replace('left','right')
            right_disp = cv2.imread(right_disp_path, cv2.IMREAD_UNCHANGED)
            right_disp = right_disp.astype(np.float32) / 32
            sample['disp_right'] = right_disp     

        sample = self.transform(sample)

        sample['valid'] = sample['disp'] < 512
        sample['index'] = idx
        sample['name'] = left_path

        return sample
    