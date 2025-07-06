import json
import os
from datetime import datetime
from typing import Dict

from moviepy import VideoFileClip


def video_to_wav(src_path: str, output_path: str) -> None:
    video = VideoFileClip(src_path)
    video.audio.write_audiofile(output_path)

def write_json(data: Dict, output_dir: str, filename: str = "asr_result"):
    os.makedirs(output_dir, exist_ok=True)
    time = datetime.now()
    filepath = os.path.join(output_dir, f"{filename}_{time}.json")
    with open(filepath, 'w') as f:
        json.dump(data, f)

def read_json(json_path: str):
    with open(json_path, 'r') as f:
        data = json.load(f)
        return data

def sliding_window_dict(d: Dict, n: int = 3, field: str = None):
    keys = sorted(d.keys())
    for i in range(len(keys) - n + 1):
        wkey = keys[i:i+n]
        if field:
            yield {k: d[k][field] for k in wkey}
        else:
            yield {k: d[k] for k in wkey}
