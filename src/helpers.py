import json
import os
import re
from datetime import datetime
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf
import torch
import torchaudio
from moviepy import VideoFileClip


def split_by_punctuation(text):
    parts = re.split(r'(?<=[,\.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def video_to_wav(src_path: str, output_path: str) -> None:
    video = VideoFileClip(src_path)
    video.audio.write_audiofile(output_path)

def write_json(data: Dict, output_dir: str, filename: str = "asr_result"): # TODO: cannot write np.array
    os.makedirs(output_dir, exist_ok=True)
    time = datetime.now() # TODO: strip
    filepath = os.path.join(output_dir, f"{filename}_{time}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f)

def read_json(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data

def save_wav(audio: List[np.float32], output_path: str, sample_rate: int = 24000) -> None:
    waveform = torch.tensor(audio).unsqueeze(0)
    torchaudio.save(output_path, waveform, sample_rate=sample_rate)

def final_pp(
    audio: List[np.float32],
    rate: float,
    output_dir: str,
    output_name: str = "output_sch",
    sample_rate=24000,
):
    wav_array = np.array(audio, dtype=np.float32)
    y_stretched = librosa.effects.time_stretch(wav_array, rate=rate)
    sf.write(os.path.join(output_dir, f"{output_name}.wav"), y_stretched, sample_rate)
