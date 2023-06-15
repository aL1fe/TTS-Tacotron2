from fastapi import FastAPI
from pydub import AudioSegment
import os
import time

import torchaudio
from speechbrain.pretrained import Tacotron2
from speechbrain.pretrained import HIFIGAN

app = FastAPI()

# Intialize TTS (tacotron2) and Vocoder (HiFIGAN)
tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech")
hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech")

@app.get("/")
def get_audio(query, file_name, dir_path="C:\\TTSAI\\Tacotron2\\Words\\"):
    a = time.time()
    mel_output, mel_length, alignment = tacotron2.encode_text(query)

    # Running Vocoder (spectrogram-to-waveform)
    waveforms = hifi_gan.decode_batch(mel_output)
    file_path_wav = f'{dir_path}{file_name}.wav'
    file_path_mp3 = f'{dir_path}{file_name}.mp3'
    torchaudio.save(file_path_wav, waveforms.squeeze(1), 22050)
    
    # Download file WAV
    audio = AudioSegment.from_wav(file_path_wav)
    os.remove(file_path_wav)
    
    # Save file MP3
    audio.export(file_path_mp3, format="mp3")
        
    return {"query": query, "executionTime": f"{round((time.time() - a) * 1000, 2)} ms"}
    