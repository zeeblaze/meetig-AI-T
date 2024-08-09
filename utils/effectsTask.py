import cv2
import numpy as np
from .yolo_seg import YOLOSeg
from PIL import Image, ImageSequence
from .animeGAN import AnimeGAN, convert_3channel_add_alpha

class ImageCapture() :
    def __init__(self, filename) -> None:
        self.img  = cv2.imread(filename)
        self.type = "image"
        if self.img is None :
            gif = Image.open(filename)
            self.img = []                                      
            self.gif_index = 0
            for frame in ImageSequence.Iterator(gif):
                red, green, blue, alpha = np.array(frame.convert('RGBA')  ) .T
                data = np.array([blue, green, red])        
                self.img.append(data.transpose())         
            self.type = "gif"

    def read(self):
        if (self.type == "gif") :
            current_index = self.gif_index
            self.gif_index += 1
            if (self.gif_index == len(self.img)) :
                self.gif_index = 0
            return True, np.array(self.img[current_index])
        else :
            if self.img is None :
                return False, self.img
            else :
                return True, self.img


class effects():
    def __init__(self):
        pass

    def DrawMask(self, img, model, provider_config):
        self.yolo_seg =  YOLOSeg(model, provider_config)
        self.yolo_seg.DetectFrame(img)
        img_BGRA = convert_3channel_add_alpha(img, alpha=255)
        combined_img = self.yolo_seg.DrawIdentifyOnFrame(img_BGRA, mask_alpha=0.3, detect=False, seg=True)

        return combined_img
    
    def RemoveBG(self, img, img_BG, model,  provider_config):
        self.yolo_seg =  YOLOSeg(model, provider_config)
        self.yolo_seg.DetectFrame(img)
        self.backgroung = ImageCapture(img_BG)
        self.sensorW = img.shape[0]
        self.sensorH = img.shape[1]
        gt_ret, img_BG = self.backgroung.read()
        if gt_ret == True:
            img_BG = cv2.resize(img_BG, (int(self.sensorW), int(self.sensorH)))
            img_BG = cv2.GaussianBlur(img_BG,(0, 0), cv2.BORDER_DEFAULT)
            img_BG = convert_3channel_add_alpha(img_BG, alpha=255)
        else :
            img_BG = np.zeros((
                int(self.sensorH),
                int(self.sensorW), 
                4
            ), np.uint8)
        img_BGRA = convert_3channel_add_alpha(img, alpha=255)
        combined_img = self.yolo_seg.DrawIdentifyOverlayOnFrame(img_BGRA, img_BG, detect=False, seg=True)

        return combined_img
    
    
    def Cartoonize(self, img, model_path, provider_config):
        self.img = convert_3channel_add_alpha(img, alpha=255)
        self.cartoonizer = AnimeGAN()
        anime =self.cartoonizer.animize(self.img, model_path, provider_config)

        return anime

