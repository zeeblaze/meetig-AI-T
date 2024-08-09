import os
import cv2
import typing
import numpy as np
import onnxruntime

def convert_3channel_add_alpha(image, alpha=255):
	b_channel, g_channel, r_channel = cv2.split(image)
	alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha 
	return cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

class AnimeGAN():
    def __init__(self) -> None:
        pass
    """ Object to image animation using AnimeGAN models
    https://github.com/TachibanaYoshino/AnimeGANv2

    onnx models:
    'https://docs.google.com/uc?export=download&id=1VPAPI84qaPUCHKHJLHiMK7BP_JE66xNe' AnimeGAN_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=17XRNQgQoUAnu6SM5VgBuhqSBO4UAVNI1' AnimeGANv2_Hayao.onnx
    'https://docs.google.com/uc?export=download&id=10rQfe4obW0dkNtsQuWg-szC4diBzYFXK' AnimeGANv2_Shinkai.onnx
    'https://docs.google.com/uc?export=download&id=1X3Glf69Ter_n2Tj6p81VpGKx7U4Dq-tI' AnimeGANv2_Paprika.onnx

    """
    _defaults = {
        "model_path": None,
        "downsize_ratio" : None,
    }

    def animize( self, frame, model_path, provider_config, downsize_ratio: float = 1.0,):
        """
        Args:
            model_path: (str) - path to onnx model file
            downsize_ratio: (float) - ratio to downsize input frame for faster inference
            provider_config: (str) - path to provider config file
        """
        self.frame = np.array(frame)
        self.downsize_ratio = downsize_ratio

        
        bgr_channel, alpha_channel = self._prepare_input(self.frame)

        outputs = self.engine_inference(bgr_channel, model_path, provider_config)

        bgr_channel = self._process_output(outputs[0], self.frame.shape[:2][::-1])

        return cv2.merge((bgr_channel, alpha_channel))

    def __to_32s(self, x):
        return 256 if x < 256 else x - x%32

    def _prepare_input(self, frame: np.ndarray, x32: bool = True) -> np.ndarray:
        """ Function to process frame to fit model input as 32 multiplier and resize to fit model input

        Args:
            frame: (np.ndarray) - frame to process
            x32: (bool) - if True, resize frame to 32 multiplier

        Returns:
            frame: (np.ndarray) - processed frame
        """
        b_channel, g_channel, r_channel, alpha_channel  = cv2.split(frame)
        frame_bgr = cv2.merge((b_channel, g_channel, r_channel))

        h, w = frame_bgr.shape[:2]
        if x32: # resize image to multiple of 32s
            frame_bgr = cv2.resize(frame_bgr, (self.__to_32s(int(w*self.downsize_ratio)), self.__to_32s(int(h*self.downsize_ratio))))
        bgr_channels = np.expand_dims(frame_bgr / 127.5 - 1.0, axis=0)

        return bgr_channels, alpha_channel

    def _process_output(self, frame: np.ndarray, wh: typing.Tuple[int, int]) -> np.ndarray:
        """ Convert model float output to uint8 image resized to original frame size

        Args:
            frame: (np.ndarray) - AnimeGaAN output frame
            wh: (typing.Tuple[int, int]) - original frame size

        Returns:
            frame: (np.ndarray) - original size animated image
        """
        if ( "Quant_output" in self.session._outputs_meta[0].name) :
            frame = np.tanh(frame).transpose((0,2,3,1)) 
        frame = frame.astype(np.float32)
        frame = (frame.squeeze() + 1.) / 2 * 255
        frame = frame.astype(np.uint8)
        frame = cv2.resize(frame, (wh[0], wh[1]))
        return frame
    def engine_inference(self, img, model_path, provider_config):
        get_providers = onnxruntime.get_available_providers()
        if 'VitisAIExecutionProvider' in get_providers:
            self.session = onnxruntime.InferenceSession(model_path,
                                                        providers=['VitisAIExecutionProvider',],
                                                        provider_options=[{"config_file": provider_config}])

        else:
            self.session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider',])
        inference_dtype = np.float16 if 'float16' in self.session.get_inputs()[0].type else np.float32
        con_img = img.astype(inference_dtype)

        # Get model info
        self.get_input_details()
        self.get_output_details()
        output = self.session.run(self.output_names, {self.input_names[0]: con_img})

        return output

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    
