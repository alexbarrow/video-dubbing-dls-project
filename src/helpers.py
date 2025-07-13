import json
import os
import re
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from moviepy import VideoFileClip
from pydub import AudioSegment, effects


def split_by_punctuation(text):
    parts = re.split(r'(?<=[,\.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def split_long_string(text, cutoff = 182, indent=40):
    midpoint = len(text) // 2
    if midpoint >= cutoff:
        indent = indent*2
    substring = text[midpoint-indent:midpoint+indent]
    matches = [(m.start(), m.group()) for m in re.finditer(r'[.,;:!?]', substring)]
    if matches:
        split_idx = matches[0][0] + midpoint-indent + 1
        return [text[:split_idx].strip(), text[split_idx:].strip()]
    else:
        return split_by_punctuation(text)

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

def stretch(audio, sample_rate, target_duration):
    current_duration = len(audio) / sample_rate
    rate = current_duration / target_duration
    stretched = torchaudio.sox_effects.apply_effects_tensor(
    torch.Tensor(np.array([audio])), sample_rate, effects=[['tempo', str(rate)]]
        )
    stretched = stretched[0].numpy()
    return stretched

def np_to_audiosegment(array, sample_rate):
    # TODO: check transform to int16
    audio_int16 = np.int16(array / np.max(np.abs(array)) * 32767)
    return AudioSegment(
        audio_int16.tobytes(), frame_rate=sample_rate, sample_width=2, channels=1
    )

def apply_fade_and_normalize(segment, fade_out_ms=100):
    segment = segment.fade_out(fade_out_ms)
    segment = effects.normalize(segment)
    return segment

def overlay_on_chunk(
    original_chunk: AudioSegment,
    vad_start_ms: int,
    vad_end_ms: int,
    synth_segment: AudioSegment,
    gain_original: int = -3,
    gain_synth: int = 1,
):
    original_cut = original_chunk[vad_start_ms:vad_end_ms].apply_gain(gain_original)
    
    synth_segment = synth_segment.set_frame_rate(original_chunk.frame_rate)
    synth_segment = synth_segment.set_channels(original_chunk.channels)
    synth_segment = synth_segment.set_sample_width(original_chunk.sample_width)
    synth_segment = synth_segment[:len(original_cut)]

    mixed = original_cut.overlay(synth_segment.apply_gain(gain_synth))
    combined = original_chunk[:vad_start_ms] + mixed + original_chunk[vad_end_ms:]
    return combined

def concat_chunks(chunks: List[AudioSegment], segments: List[List], len_orig_audio: int):
    if len(chunks) != len(segments):
        raise ValueError("Count of chunks must be equal to count of segments.")

    last_chunk_end = segments[-1][1]
    reconstructed = AudioSegment.silent(duration=0)
    current_position = 0

    for (start, end), chunk in zip(segments, chunks):
        if start > current_position:
            silence_duration = start - current_position
            reconstructed += AudioSegment.silent(duration=silence_duration)
        reconstructed += chunk

        current_position = end
        if current_position == last_chunk_end:
            reconstructed += AudioSegment.silent(duration=(len_orig_audio - current_position))
    return reconstructed

def cosine_similarity(vec1: torch.Tensor, vec2: torch.Tensor) -> float:
    vec1_flat = vec1.view(-1)
    vec2_flat = vec2.view(-1)

    similarity = F.cosine_similarity(vec1_flat.unsqueeze(0), vec2_flat.unsqueeze(0)).item()
    return similarity
