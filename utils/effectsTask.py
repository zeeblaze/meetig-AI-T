import cv2
import numpy as np
from .yolo_seg import YOLOSeg
from .animeGAN import AnimeGAN, convert_3channel_add_alpha


class effects():
    def __init__(self):
        pass

    def DrawMask(self, img, model, provider_config):
        self.yolo_seg =  YOLOSeg(model, provider_config, conf_thres=0.3, iou_thres=0.3)
        boxes, scores, class_ids, masks = self.yolo_seg(img)

        combined_img = self.yolo_seg.draw_masks(img)

        return combined_img
    
    def Cartoonize(self, img, model_path, provider_config):
        self.img = convert_3channel_add_alpha(img, alpha=255)
        self.cartoonizer = AnimeGAN()
        anime =self.cartoonizer.animize(self.img, model_path, provider_config)

        return anime

