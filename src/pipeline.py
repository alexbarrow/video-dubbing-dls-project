
import numpy as np
from config import config
from pydub import AudioSegment

from src.asr import ASR
from src.helpers import (
    apply_fade_and_normalize,
    concat_chunks,
    np_to_audiosegment,
    overlay_on_chunk,
    stretch,
)
from src.mt import MT
from src.tts import XTTSv2
from src.utils import get_chunks, update_boundary


class VideoDubPipe:
    def __init__(self, config_path: str):
        print("Init config...")
        self.cfg = config.load_config(config_path)
        self.speaker_path = self.cfg["reference_wav"]
        self.sample_rate=self.cfg["sample_rate"]
        self.pp_pause = self.cfg["postprocess"]["pause"]
        self.pp_fade = self.cfg["postprocess"]["fade_out"]
        self.pp_orig_gain = self.cfg["postprocess"]["orig_gain"]
        self.pp_synth_gain = self.cfg["postprocess"]["synth_gain"]

        print("Init ASR...")
        self.asr_model = ASR(model_type=self.cfg["asr"]["model_type"], device=self.cfg["device"])
        print("Init MT...")
        self.mt_model = MT(model_name=self.cfg["mt"]["model_name"])
        print("Init TTS...")
        self.tts_model = XTTSv2(device=self.cfg["device"])
        self.tts_config = dict(self.cfg["tts"])
        print("Initialization completed.")
        
    def transform(self, path_to_wav: str,  output_dir: str): # TODO: refactor params
        print("Split on chunks...")
        chunks, orig_segments, len_orig_audio = get_chunks(path_to_wav, output_dir)
        print("Transcribe...")
        chunks = self.asr_model.transcribe(chunks, save=self.cfg["asr"]["save_results"])
        print("Translate...")
        chunks = self.mt_model.translate(chunks)
        print("TTS step...")
        tts_chunks = self.tts_model.tts_chunks(chunks, self.speaker_path, **self.tts_config)
        durations = [len(ch) for ch in tts_chunks]
        durations = [round(x/self.sample_rate,2) for x in durations]
        new_sp_boundary = update_boundary(chunks, durations, pause=self.pp_pause) # подгон сегментов с учетом пауз
        result_chunks = []
        print("Combine...")
        for chunk, tts_sample, sp_seg in zip(chunks.values(), tts_chunks, new_sp_boundary):
            original_wav = AudioSegment.from_file(chunk["path"])
            if not sp_seg:
                result_chunks.append(original_wav)
                continue
            duration = np.round(sp_seg["end"], 3) - np.round(sp_seg["start"], 3)
            synth_np_stretched = stretch(np.array(tts_sample), self.sample_rate, duration)

            synth_audioseg = np_to_audiosegment(synth_np_stretched, self.sample_rate)
            synth_audioseg = apply_fade_and_normalize(synth_audioseg, fade_out_ms=self.pp_fade)

            start_ms = int(sp_seg["start"] * 1000)
            end_ms = int(sp_seg["end"] * 1000)

            processed_chunk = overlay_on_chunk(
                original_wav, start_ms, end_ms, synth_audioseg,
                gain_original=self.pp_orig_gain, gain_synth=self.pp_synth_gain
            )
            result_chunks.append(processed_chunk)

        final_audio = concat_chunks(result_chunks, orig_segments, len_orig_audio)
        return final_audio
