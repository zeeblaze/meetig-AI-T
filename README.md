# Meetig-AI-T
Official repository of Meetig-AI-T.
Meetig-AI-T is a virtual meeting assistant powered by the AMD™ RyzenAI, that helps you take down notes during meetings by transcribing your spoken words and that of your peers using the VB-Audio Cable through OpenAI's Whisper transcription model.

## 🛠 Usage
To use Meetig-AI-T, you can follow the steps below to clone and launch.

### Installation
Clone the Meetig-AI-T repository.

```bash
    git clone https://github.com/zeeblaze/meetig-AI-T.git
```

### Launch
Running Meetig-AI-T using a conda environment.

#### 1. Navigate to Mettig-AI-T directory
```bash
    cd meetig-AI-T
```

#### 2. Setup the conda environment
```bash
    conda activate <your virtual environment>
    pip install -r requirements.txt
```
#### 3. Launch the user interface
```bash
    python App.py
```
<br/>

## ⌛ Todo list
List of Features to be implemented.

- [x] Strem webcam to meeting.
- [x] Stream pre recorded video to meeting.
- [x] OpenAI Whisper for audio transcription.
- [x] Quantize models to 16bit (fp16).
- [x] Apply Anime Effects to Video with [AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3).
- [x] Apply segmentation Effects to video with Yolov8.
- [x] Remove video background with [Yolov8n-seg](https://github.com/ultralytics/ultralytics)
- [x] Ollama Mistral support for summarizing transcripts.
- [] ChatGPT support for summarizing.
- [] Export transcript as text.
- [] Export Summary as text.
- [] Translate transcripts and summaries to any language.

<br/>

## 🎊 Demo and Article

#### Hackster Article

* Click [Here](https://www.hackster.io/habib-elediko/introducing-meetig-ai-t-your-ai-powered-meeting-assistant-ff164) to read the Hackster.io Article about Meetig-AI-T.

#### Youtube Demo Video

* Click [Here](https://youtu.be/nbnuHkKKfvk?si=e98x5uLY2lgUjpTg) to watch the Youtube demo video.

<br/>

## 💬 Updates
* ` 2024-08-04` Background removal added with yolov8n 16bit, also running on the RyzenAI NPU or CPU.
* `2024-07-30` The first Commit of Meetig-AI-T with most of the features released and Anime Effects wth [AnimeGANv3](https://github.com/TachibanaYoshino/AnimeGANv3), all running on the AMD RyzenAI NPU at 16bit or CPU for PCs without RyzenAI enabled.
