import os
from typing import Dict

from pydub import AudioSegment
from pydub.silence import split_on_silence

from src.helpers import final_pp, write_json
from src.vad import find_timestamps


def get_chunks(
    wav_path: str,
    output_dir: str,
    min_silence_len=700,
    silence_thresh=-40,
    keep_silence=300,
    save=True,
) -> Dict:
    audio = AudioSegment.from_file(wav_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )
    os.makedirs(output_dir, exist_ok=True)
    chunk_json = {}

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk_{i:03d}.wav"
        out_path = os.path.join(output_dir, chunk_name)
        chunk.export(out_path, format="wav")
        item = {"path": out_path, "len": round(len(chunk)/1000, 1)}
        chunk_json[i] = item

    find_timestamps(chunk_json)
    if save:
        write_json(chunk_json, output_dir, filename="chunk_step_result")
    return chunk_json

def post_processing_chunks(chunks : Dict, output_dir: str):
    for chunk in chunks.values():
        # TODO: instead of chunk len use len of asr segs ???
        rate = chunk["len"] / len(chunk["tts_result"])
        # TODO: update original into speech placeholder
        final_pp(chunk["tts_result"], rate=rate, output_dir=output_dir)
