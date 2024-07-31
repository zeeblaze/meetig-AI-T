import sys
import cv2
import os
import time
import numpy as np
import pyvirtualcam
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from ui.MainWindow import Ui_MainWindow
from utils.effectsTask import effects
from utils.transcriptTask import transcriber
from utils.llmTask import summary
from pathlib import Path

play = False

frame_raw = []

frame_store = np.array(frame_raw)

CURRENT_DIR = Path(__file__).parent
temp_dir = Path(CURRENT_DIR / "_temp/")
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir)

def getTimeInfo(fps, length):
    length_int = int(length)
    length_diff = (length - length_int)

    framerate_length = 1000 / fps
    framerate_length_int = int(framerate_length)
    framerate_length_diff = (framerate_length - framerate_length_int)

    total_diff = length_diff + framerate_length_diff
    total_int = framerate_length_int - length_int

    return total_diff, total_int

class VideoThread(QThread):
    UpdateImage = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.ThreadActive = False
        self.cam = None


    def run(self):
        global window
        global play
        global frame_raw
        global frame_store

        self.ThreadActive = True

        while self.ThreadActive:
            start = time.time()

            if play == True:
                if window.cam.get(cv2.CAP_PROP_POS_FRAMES) == window.cam.get(cv2.CAP_PROP_FRAME_COUNT):
                    window.cam.set(cv2.CAP_PROP_POS_FRAMES, 0)
                success, frame_raw = window.cam.read()
                if success:
                    FlippedImage = cv2.resize(cv2.flip(frame_raw, 1), (640, 480))
                    frame_raw = FlippedImage
                    with pyvirtualcam.Camera(frame_raw.shape[1], frame_raw.shape[0], fps=30) as cam:
                        if window.apply_effects == True:
                            if window.anime == True:
                                if window.effects_type == "hayao":
                                    model = Path(CURRENT_DIR / "models/quantized-models/AnimeGANv3_Hayao_36_fp16.onnx")
                                elif window.effects_type == "jp_face":
                                    model = Path(CURRENT_DIR / "models/quantized-models/AnimeGANv3_JP_face_v1.0_fp16.onnx")
                                elif window.effects_type == "shinkai":
                                    model = Path(CURRENT_DIR / "models/quantized-models/AnimeGANv3_Shinkai_37_fp16.onnx")
                                
                                model_path = model
                                config_file_path ="vaip_config.json"
                                provider_options = config_file_path
                                segmentor = effects()
                                combined_frame = segmentor.Cartoonize(frame_raw, model_path, provider_options)
                                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGBA2RGB)

                                self.UpdateImage.emit(combined_frame)
                                if window.send_to_cam == True:
                                    combined = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                                    cam.send(combined)
                                    cam.sleep_until_next_frame()
                            
                            else:
                                pass

                            if window.segment == True:
                                seg_model_path = Path(CURRENT_DIR / "models/yolov8n-seg.onnx")
                                config_file_path ="vaip_config.json"
                                provider_options = config_file_path
                                segmentor = effects()
                                combined_frame = segmentor.DrawMask(frame_raw, seg_model_path, provider_options)
                                combined_frame = cv2.cvtColor(combined_frame, cv2.COLOR_RGBA2RGB)
                                
                                                        
                            self.UpdateImage.emit(combined_frame)
                            if window.send_to_cam == True:
                                combined = cv2.cvtColor(combined_frame, cv2.COLOR_RGB2BGR)
                                cam.send(combined)
                                cam.sleep_until_next_frame()
                        
                        else:
                            self.UpdateImage.emit(frame_raw)
                            if window.send_to_cam == True:
                                combined = cv2.cvtColor(frame_raw, cv2.COLOR_RGB2BGR)
                                cam.send(combined)
                                cam.sleep_until_next_frame()

            if window.webcam == False:
                end = time.time()
                #Get duration of process in milliseconds
                length = (end-start)*1000

                _, total_int = getTimeInfo(window.fps, length)
                cv2.waitKey(total_int)
            else:
                cv2.waitKey(10)
            
            
    def stop(self):
        window.cam.release()
        self.ThreadActive = False
        self.quit()

class TranscribeMicThread(QThread):
    mic_transcript = pyqtSignal(str)

    def __init__(self, mic_index):
        super().__init__()
        self.mic_index = mic_index
        self.threadActive = False


    def run(self):
        global window

        self.threadActive = True
        while self.threadActive:
            if window.mic_transcribe:
                file_path = os.path.join(temp_dir, 'temp_mic_recording.wav')
                transcripter = transcriber()
                mic_index = self.mic_index
                config = window.transcript_config
                transcript = transcripter.recordAndTranscribe(file_path, mic_index, config)
                os.remove(file_path)

                if transcript is not None:
                    if transcript.lower() == "exit":
                        os.remove(file_path)
                        break  # This will exit the loop if "exit" is detected
                    
                    self.mic_transcript.emit(transcript)
                    print(transcript)
                else:
                    print("Input was None, continuing loop...")

            else:
                self.stop()


    def stop(self):
        self.threadActive = False
        self.quit()

class StreamAudioThread(QThread):
    def __init__(self):
        super().__init__()
        self.threadActive = False

    def run(self):
        self.threadActive = True
        while self.threadActive:
            if window.streamAudio == True:
                self.transcripter = transcriber()
                self.device_index = self.transcripter.find_virtual_audio_device()
                self.transcripter.streamAudio(self.device_index)

            else:
                self.stop()

    def stop(self):
        self.transcripter.stopAudioStream()
        self.threadActive = False
        self.quit()



class TranscribeVMicThread(QThread):
    vMic_transcript = pyqtSignal(str)
    def __init__(self):
        super().__init__()
        self.threadActive = False

    def run(self):
        global window

        self.threadActive = True
        while self.threadActive:
            if window.vMic_transcribe:
                file_path = os.path.join(temp_dir, 'temp_vMic_recording.wav')
                transcripter = transcriber()
                vMic_index = transcripter.find_virtual_audio_device()
                config = window.transcript_config
                transcript = transcripter.recordAndTranscribe(file_path, vMic_index, config)
                os.remove(file_path)

                if transcript is not None:
                    if transcript.lower() == "exit":
                        os.remove(file_path)
                        break  # This will exit the loop if "exit" is detected
                    
                    self.vMic_transcript.emit(transcript)
                    print(transcript)
                else:
                    print("Input was None, continuing loop...")
            
            else:
                self.stop()


    def stop(self):
        self.threadActive = False
        self.quit()

class SummarizeTranscript(QThread):
    transcriptSummary = pyqtSignal(str)
    def __init__(self) -> None:
        super().__init__()
        self.threadActive = False

    def run(self):
        global window

        self.threadActive = True
        if self.threadActive:
            Summarizer = summary()
            transcript = window.transcript
            summary_config = window.summary_config
            api_key = window.llm_api_key
            summaries = Summarizer.summarize(transcript, summary_config, api_key)
            if transcript is not None:
                self.transcriptSummary.emit(summaries)
                print(summaries)
            else:
                print("Input was None, continuing loop...")


    def stop(self):
        self.threadActive = False
        self.quit()


class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.displayingVideo = False
        self.webcam = False
        self.defaultmic = False
        self.virtualmic = False
        self.all_sources = False
        self.summarize_transcript = False
        self.Transcribe = False
        self.streamAudio = False
        self.remove_background = False
        self.send_to_cam = False
        self.segment = False
        self.anime = False

        self.fps = 60
        self.state = 'nothing'
        self.apply_effects = False
        self.effects_type = 'nothing'
        self.draw_effects.toggled.connect(self.drawEffects)
        self.output_text.setReadOnly(True)
        self.summary_output.setReadOnly(True)

        self.mic_transcribe = False
        self.mic_index = 0
        self.transcript_config = "nothing"
        self.mic_worker = TranscribeMicThread(self.mic_index)
        self.mic_worker.mic_transcript.connect(self.updateTranscript)

        self.vMic_transcribe = False
        self.vMic_worker = TranscribeVMicThread()
        self.vMic_worker.vMic_transcript.connect(self.updateTranscript)

        self.summary_config = "nothing"
        self.llm_api_key = "nothing"
        self.summary_worker = SummarizeTranscript()
        self.summary_worker.transcriptSummary.connect(self.updateSummary)
        self.actionTranscribe_Only.toggled.connect(self.transcribeOnly)
        self.actionStream_Transcribe.toggled.connect(self.transcribeStream)

        self.stream_worker = StreamAudioThread()
        
        self.actionWebcam.toggled.connect(self.openWebcam)
        self.actionOpen_Video.toggled.connect(self.openNormalVideo)
        self.pushButton_3.toggled.connect(self.transcribe)

        self.actionWhisper_Local.toggled.connect(self.localWhisper)
        self.actionWhisper_API.toggled.connect(self.cloudWhisper)
        self.edit_transcript.toggled.connect(self.editTranscript)
        self.pushButton_5.toggled.connect(self.summarize)
        self.checkBox.toggled.connect(self.removeBackground)
        self.vcam_stream.toggled.connect(self.send_to_vcam)
        if self.comboBox_3.currentIndex() < 0:
            self.draw_effects.setDisabled(True)
        self.pushButton.toggled.connect(self.pause_video)

    def pause_video(self):
        if self.pushButton.isChecked():
            self.pause()
            self.send_to_cam = False
            self.pushButton.setText("Resume Video")
        
        else:
            self.pause()
            self.send_to_cam = True
            self.pushButton.setText("Play/Pause")
        


    def removeBackground(self):
        if self.checkBox.isChecked():
            self.remove_background = True

    def send_to_vcam(self):
        if self.vcam_stream.isChecked():
            self.send_to_cam = True
        else:
            self.send_to_cam = False

    def drawEffects(self):
        if self.draw_effects.isChecked():
            self.comboBox_3.setDisabled(True)
            self.apply_effects = True
            if self.comboBox_3.currentIndex() == 0:
                self.effects_type = "segmentation"
                self.segment = True
                self.anime = False

            elif self.comboBox_3.currentIndex() == 1:
                self.effects_type = "hayao"
                self.segment = False
                self.anime = True

            elif self.comboBox_3.currentIndex() == 2:
                self.effects_type = "jp_face"
                self.segment = False
                self.anime = True

            elif self.comboBox_3.currentIndex() == 3:
                self.effects_type = "shinkai"
                self.segment = False
                self.anime = True
            
            elif self.comboBox_3.currentIndex() == 4:
                self.effects_type = "cute"
                self.segment = False
                self.anime = True
        
        else:
            self.comboBox_3.setDisabled(False)
            self.apply_effects = False


    def summarize(self):
        if self.pushButton_5.isChecked():
            if self.comboBox.currentIndex() == 1:
                self.summary_config = "ollama_mistral"
                self.llm_api_key = "ollama"
                
            else:
                self.summary_config = "nothing"
                self.llm_api_key = "nothing"
            self.transcript = self.output_text.toPlainText()
            self.summary_worker.start()


    def updateSummary(self, summary):
        self.summary = summary
        self.summary_output.setText(summary)
    
    def localWhisper(self):
        if self.actionWhisper_Local.isChecked():
            if self.actionWhisper_API.isChecked():
                self.actionWhisper_API.setChecked(False)
            self.whisperlocal = True
            self.transcript_config = "whisper_local"

        else:
            self.transcript_config = "nothing"
            self.whisperlocal = False

    def cloudWhisper(self):
        if self.actionWhisper_API.isChecked():
            if self.actionWhisper_Local.isChecked():
                self.actionWhisper_Local.setChecked(False)
            self.whisperApi = True
            self.transcript_config = "whisper_api"

        else:
            self.transcript_config = "nothing"
            self.whisperApi = False

    def transcribe(self):
        if self.comboBox_2.currentIndex() == 0:
            if self.virtualmic == True:
                self.virtualmic = False
            self.defaultmic = True
            self.transcribeMic()

        elif self.comboBox_2.currentIndex() == 1:
            if self.defaultmic == True:
                self.defaultmic = False
            self.virtualmic = True
            self.transcribeVMic()

        elif self.comboBox_2.currentIndex() == 2:
            self.all_sources = True
            self.transcribeAll()

        else:
            self.vMic_worker.stop()
            self.mic_worker.stop()
            self.defaultmic = False


    def transcribeMic(self):
        if self.defaultmic:
            if self.Transcribe == True:
                if self.pushButton_3.isChecked():
                    self.comboBox_2.setEnabled(False)
                    self.edit_transcript.setEnabled(False)
                    self.mic_transcribe = True
                    self.mic_worker.start()
                    self.pushButton_3.setText("Stop Mic Transcription")
                else:
                    self.comboBox_2.setEnabled(True)
                    self.edit_transcript.setEnabled(True)
                    self.mic_transcribe = False
                    self.pushButton_3.setText("Start Transcription")

    def transcribeVMic(self):
        if self.virtualmic:
            if self.Transcribe == True:
                if self.pushButton_3.isChecked():
                    self.comboBox_2.setEnabled(False)
                    self.edit_transcript.setEnabled(False)
                    self.vMic_transcribe = True
                    self.vMic_worker.start()
                    self.pushButton_3.setText("Stop VCable Transcription")
                else:
                    self.comboBox_2.setEnabled(True)
                    self.edit_transcript.setEnabled(True)
                    self.vMic_transcribe = False
                    self.pushButton_3.setText("Start Transcription")

    def transcribeAll(self):
        if self.all_sources:
            if self.Transcribe == True:
                if self.pushButton_3.isChecked():
                    self.comboBox_2.setEnabled(False)
                    self.edit_transcript.setEnabled(False)
                    self.mic_transcribe = True
                    self.vMic_transcribe = True
                    self.mic_worker.start()
                    self.vMic_worker.start()
                    self.pushButton_3.setText("Stop All Transcription")
                else:
                    self.comboBox_2.setEnabled(True)
                    self.edit_transcript.setEnabled(True)
                    self.vMic_transcribe = False
                    self.mic_transcribe = False
                    self.pushButton_3.setText("Start Transcription")

    def transcribeOnly(self):
        self.Transcribe = False

        if self.actionTranscribe_Only.isChecked():
            if self.actionStream_Transcribe.isChecked:
                self.actionStream_Transcribe.setChecked(False)
                self.streamAudio = False
                if self.streamAudio == True:
                    self.stream_worker.stop()
            self.Transcribe = True
        else:
            self.Transcribe = False

    def transcribeStream(self):
        self.Transcribe = False

        if self.actionStream_Transcribe.isChecked():
            if self.actionTranscribe_Only.isChecked:
                self.actionTranscribe_Only.setChecked(False)
            self.streamAudio = True
            self.stream_worker.start()
            self.Transcribe = True
        else:
            self.streamAudio = False
            self.stream_worker.stop()
            self.Transcribe = False

    def updateTranscript(self, transcript):
        self.output_text.append(transcript)

    def editTranscript(self):
        if self.edit_transcript.isChecked():
            self.output_text.setReadOnly(False)
            self.edit_transcript.setText("Stop Editing")

        else:
            self.output_text.setReadOnly(True)
            self.edit_transcript.setText("Edit Transcript")

    def setVideo(self):
        self.video = VideoThread()
        self.video.UpdateImage.connect(self.updateImageSlot)
        self.video.start()

        if self.webcam == False:
            self.updateUIVideo()
        else:
            self.updateUIWebcam()

        self.displayingVideo = True

    def convertCvToQT(self, cv_frame):

        rgb_image = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
        
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
            
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)

        self.video_width = self.video_frame.geometry().width()

        self.video_height = self.video_frame.geometry().height()
            
        pic = convert_to_Qt_format.scaled(self.video_width, self.video_height, Qt.AspectRatioMode.KeepAspectRatio)

        return QPixmap.fromImage(pic)


    def updateImageSlot(self, cv_frame):
        qt_frame = self.convertCvToQT(cv_frame)

        self.video_label.setPixmap(qt_frame)

    def updateUIWebcam(self):
        self.pause()

    def updateUIVideo(self):
        self.pause()

    def openWebcam(self):
        global play

        if play == True:
            play = False

        if self.actionWebcam.isChecked():
            if self.actionOpen_Video.isChecked():
                self.actionOpen_Video.setChecked(False)
            self.device_index = int(self.cam_selector.currentIndex())
            self.cam = cv2.VideoCapture(self.device_index)

            self.webcam = True

            if self.displayingVideo == False:
                self.setVideo()
            
            play = True
    
        else:
            self.cam.release()
            self.webcam = False

    def openNormalVideo(self):
        self.openVideo('normal')

        
    def openVideo(self, config):
        global play

        play = False

        if self.actionOpen_Video.isChecked():
            if self.actionWebcam.isChecked():
                self.actionWebcam.setChecked(False)
            videoFile, _ = QFileDialog.getOpenFileName(self, "Open Video", "", "All Files (*);")
            if videoFile:
                self.video_path = videoFile
                self.cam = cv2.VideoCapture(self.video_path)
                self.fps = self.cam.get(cv2.CAP_PROP_FPS)

                if self.displayingVideo == False:
                    self.setVideo()

                else:
                    self.pause()

                if config == 'normal':
                    self.webcamBox = False

                else:
                    self.webcam = True

    def pause(self):
        global play

        if play == True:
            play = False

        else:
            play = True
                    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())