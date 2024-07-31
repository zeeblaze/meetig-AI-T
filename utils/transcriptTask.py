import pyaudio
import wave
import whisper
import os
import openai

openai.api_key = ""

class transcriber():
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100

    def transcribe(self, audio_file_path, config):
        try:
            if config == "whisper_api":
                audio_file = open(audio_file_path, "rb")
                api_transcript = openai.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                ).text
                audio_file.close()
                return api_transcript
            
            elif config == "whisper_local":
                model = whisper.load_model("base.en")
                local_transcript = model.transcribe(audio_file_path)
                return local_transcript["text"]
            
            else:
                pass
            
            # If config does not match expected values, log the issue and return a default message
            print(f"Unexpected config value: {config}")
            return "Error: Unsupported transcription config"
        
        except Exception as e:
            # Log the error and return a default message
            print(f"An error occurred during transcription: {e}")
            return "Error: Transcription failed"

    
    def streamAudio(self, device_index):
        audio = self.audio
        self.stream_in = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )
        self.stream_out = audio.open(
        format=self.FORMAT,
        channels=self.CHANNELS,
        rate=self.RATE,
        output=True,
        frames_per_buffer=self.CHUNK
        )

        try:
            while True:
                data = self.stream_in.read(self.CHUNK)


                self.stream_out.write(data)

        except Exception as e:
            print(e)
    def stopAudioStream(self):        
        self.stream_in.stop_stream()
        # self.stream_in.close()
        self.stream_out.stop_stream()
        # self.stream_out.close()

        # self.audio.terminate()

    
    def recordAndTranscribe(self, file_path, device_index, config):
        audio = self.audio
        stream_in = audio.open(
            format=self.FORMAT,
            channels=self.CHANNELS,
            rate=self.RATE,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=self.CHUNK
        )

        try:
            while True:
                frames = []

                for i in range(0, int(self.RATE / self.CHUNK * 5)):
                    data = stream_in.read(self.CHUNK)
                    frames.append(data)

                wf = wave.open(file_path,'wb')
                wf.setnchannels(self.CHANNELS)
                wf.setframerate(self.RATE)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.writeframes(b''.join(frames))
                wf.close()
                
                transcript = self.transcribe(file_path, config)

                return transcript

        except Exception as e:
            print(f"Transcription Stopped with {e}")

        finally:
            stream_in.stop_stream()
            stream_in.close()
            audio.terminate()
            
    def find_virtual_audio_device(self):
        for i in range(self.audio.get_device_count()):
            dev = self.audio.get_device_info_by_index(i)
            # Check if "CABLE Output" is in the name of the device
            if "CABLE Output (VB-Audio Virtual" in dev["name"]:
                return dev["index"]
        
def main():
    while True:
        file_path = "./temp/temp_recording.wav"
        transcripter = transcriber()
        virtual_device_index = transcripter.find_virtual_audio_device()
        input = transcripter.recordAndTranscribe(file_path, virtual_device_index, 'whisper_local')
        os.remove(file_path)

        if input.lower() == "exit":
            os.remove(file_path)

        if input.lower() == "exit":
            break

        print(input)
        

if __name__ == "__main__":
    main()