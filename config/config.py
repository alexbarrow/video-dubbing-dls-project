from typing import Literal, Optional

import yaml
from pydantic import BaseModel


class BaseModelClass(BaseModel):
    def __getitem__(self, key):
        return getattr(self, key)
    
class ASRConfig(BaseModelClass):
    model_type: Literal["small", "base", "large"]
    save_results: bool = False

class TTSConfig(BaseModelClass):
    language: str = "ru"
    speed: float = 1
    length_penalty: int = 1
    repetition_penalty: float = 10.0
    temperature: float = 0.3
    cutoff: Optional[float]

class MTConfig(BaseModelClass):
    model_name: str

class PostProcess(BaseModelClass):
    fade_out: float = 100
    pause: float = 0.005
    orig_gain: int = -6
    synth_gain: int = 6

class Config(BaseModelClass):
    device: Literal["cpu", "cuda"]
    tts_checkpoint: Optional[str]
    reference_wav: str
    sample_rate: int = 24000

    asr: ASRConfig
    tts: TTSConfig
    mt: MTConfig
    postprocess: PostProcess


def load_config(path: str = "config.yaml") -> Config:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)
