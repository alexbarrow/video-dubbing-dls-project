import os
from typing import Dict, List

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

from src.helpers import write_json
from src.vad import find_timestamps


def get_chunks(
    wav_path: str,
    output_dir: str,
    min_silence_len: int = 700,
    silence_thresh: int = -40,
    save: bool = True,
) -> Dict:
    audio = AudioSegment.from_file(wav_path)
    len_orig_audio = len(audio)
    segments = detect_nonsilent(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
    )

    chunks = [audio[start:end] for start, end in segments]
    os.makedirs(output_dir, exist_ok=True)
    chunk_json = {}

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:03d}.wav"
        out_path = os.path.join(output_dir, chunk_name)
        chunk.export(out_path, format="wav")
        item = {
            "path": out_path,
            "len": round(len(chunk) / 1000, 1),
            "orig_seg": {"start": segments[i][0], "end": segments[i][1]},
        }
        chunk_json[i] = item

    find_timestamps(chunk_json)
    if save:
        write_json(chunk_json, output_dir, filename="chunk_step_result")
    return chunk_json, segments, len_orig_audio

def update_boundary( # TODO: refactor
    chunks: Dict, durations: List, sample_rate: int = 16000, pause: float = 0.005
) -> List[Dict]:

    lengths, sp_boudary, orig_boundary = [], [], []

    for chunk in chunks.values():
        lengths.append(chunk["len"])
        orig_boundary.append({"start": round(chunk["orig_seg"]["start"]/1000, 2), "end": round(chunk["orig_seg"]["end"]/1000, 2)})
        if not chunk["speech_boundary"]:
            sp_boudary.append([])
        else:
            orig_start = round(chunk["speech_boundary"][0]["start"] / sample_rate, 2)
            orig_end = round(chunk["speech_boundary"][0]["end"] / sample_rate, 2)
            sp_boudary.append([{"start": orig_start, "end": orig_end}])


    pauses = []
    for ai, bi in zip(sp_boudary, orig_boundary):
        if not ai:
            left_pause = 0.0
            right_pause = bi['end'] - bi['start']
        else:
            left_pause = ai[0]['start']
            right_pause = bi['end'] - bi['start'] - ai[-1]['end']
        
        pauses.append([left_pause, right_pause])

    new_segs = []
    for length, sb, dur, p in zip(lengths, sp_boudary, durations, pauses):
        if not sb:
            new_segs.append({})
            continue

        orig_duration = sb[0]["end"] - sb[0]["start"]
        delta = dur - orig_duration

        if delta <= 0:
            mid_orig = (sb[0]["end"] + sb[0]["start"]) / 2
            synth_start = mid_orig - dur / 2
            synth_end = synth_start + dur

            if synth_start < 0:
                synth_start = 0
                synth_end = synth_start + dur
            if synth_end > length:
                synth_end = length
                synth_start = synth_end - dur

            new_segs.append({"start": synth_start, "end": synth_end})
            continue

        new_segs.append({"start": sb[0]["start"], "end": sb[0]["end"]})

    return new_segs
