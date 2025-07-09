from typing import Dict

import numpy as np
from pydub import AudioSegment

from src.asr import ASR
from src.helpers import (
    apply_fade_and_normalize,
    concat_chunks,
    np_to_audiosegment,
    overlay_on_chunk,
    stretch,
    write_json,
)
from src.mt import MT
from src.tts import XTTSv2
from src.utils import get_chunks, update_boundary


class VideoDubPipe:
    def __init__(self, chunks: Dict, speaker_path: str):
        self.speaker_path = speaker_path
        self.chunks = chunks # TODO: to method transform
        self.tts_model = XTTSv2(device="cuda") #TODO: add to config
        self.sample_rate=24000
        
    def transform(self, output_dir: str): # TODO: refactor params
        tts_chunks = self.tts_model.tts_chunks(self.chunks, self.speaker_path, speed=1.2, cutoff=0.06)
        durations = [len(ch) for ch in tts_chunks]
        durations = [round(x/24000,2) for x in durations]
        new_sp_boundary = update_boundary(self.chunks, durations)
        result_chunks = []

        for chunk, tts_sample, sp_seg in zip(self.chunks.values(), tts_chunks, new_sp_boundary):
            original_wav = AudioSegment.from_file(chunk["path"])
            if not sp_seg:
                result_chunks.append(original_wav)
                continue
            duration = np.round(sp_seg["end"], 3) - np.round(sp_seg["start"], 3)
            synth_np_stretched = stretch(np.array(tts_sample), self.sample_rate, duration)

            synth_audioseg = np_to_audiosegment(synth_np_stretched, self.sample_rate)
            synth_audioseg = apply_fade_and_normalize(synth_audioseg)

            start_ms = int(sp_seg["start"] * 1000)
            end_ms = int(sp_seg["end"] * 1000)

            processed_chunk = overlay_on_chunk(
                original_wav, start_ms, end_ms, synth_audioseg,
                gain_original=-6, gain_synth=6
            )
            result_chunks.append(processed_chunk)

        final_audio = concat_chunks(result_chunks)
        return final_audio

def temp_pipe_asr_mt(path_to_wav: str, output_dir:str, device="cuda"):
    asr_model = ASR(device=device)
    mt_model = MT()
    chunks = get_chunks(path_to_wav, output_dir)
    chunks = asr_model.transcribe(chunks, save=False)
    chunks = mt_model.translate(chunks)
    write_json(chunks, output_dir, "all_data")
