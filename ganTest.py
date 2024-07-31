import onnxruntime as ort
import time, os, cv2
import numpy as np


class AnimeGaN:
    def __init__(self, video):
        self.model_path = "models/Hayao_64.onnx"
        limit = 1280
        cam = cv2.VideoCapture(video)
        width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = cam.get(cv2.CAP_PROP_FPS)
        cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            success, frame = cam.read()
            img = cv2.resize(frame, (640, 480))

