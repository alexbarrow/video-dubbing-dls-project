import os
from typing import Dict, List, Optional, Union

from TTS.api import TTS
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

from src.helpers import split_long_string


class XTTSv2:
    def __init__(
        self,
        checkpoint_dir: Optional[str] = None,
        vocab: Optional[str] = None,
        device: str = "cpu",
    ):
        self.checkpoint = False
        if checkpoint_dir:
            self.model_config = os.path.join(checkpoint_dir, "config.json")

            config = XttsConfig()
            config.load_json(self.model_config)
            model = Xtts.init_from_config(config)
            model.load_checkpoint(
                config,
                checkpoint_path=os.path.join(checkpoint_dir, "best_model.pth"),
                vocab_path=vocab,
            )
            self.model = model

        else:
            model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
            self.model = TTS(model_name).to(device)

    def tts(
        self,
        text: str,
        speaker_wav: Union[str, List],
        language: str = "ru",
        save: bool = False,
        output_path: Optional[str] = None,
        speed: float = 1,
        length_penalty: int = 1,
        repetition_penalty: float = 10.0,
        temperature: float = 0.3
    ):
        if self.checkpoint:
            raise AttributeError("Using wrong method. Use 'XTTSv2.inference()'.")
        if save and output_path is not None:
            self.model.tts_to_file(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
                speed=speed,
                temperature=temperature
            )
        else:
            audio = self.model.tts(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
                length_penalty = length_penalty,
                repetition_penalty = repetition_penalty,
                temperature=temperature
            )
            return audio

    def tts_chunks(
        self, chunks: Dict,
        speaker_wav: Union[str, List],
        language: str = "ru",
        speed: float = 1,
        length_penalty: int = 1,
        repetition_penalty: float = 10.0,
        temperature: float = 0.3,
        cutoff: Optional[float] = None
    ):
        result_chunks = []
        for chunk in chunks.values():
            if not chunk["translated_text"]:
                result_chunks.append([])
                continue
            text = chunk["translated_text"]
            if len(text) > 182: # limit from xTTSv2 for ru version
                result_audio = []
                text_list = split_long_string(text)
                for txt in text_list:
                    audio_seg = self.tts(
                        txt,
                        speaker_wav,
                        language=language,
                        speed=speed,
                        temperature=temperature,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                    )
                    result_audio.extend(list(audio_seg))
            else:
                result_audio = self.tts(
                    text,
                    speaker_wav,
                    language=language,
                    speed=speed,
                    temperature=temperature,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
            if cutoff:
                cut_idx = len(result_audio) - int(len(result_audio) * cutoff)
                result_chunks.append(result_audio[:cut_idx])
            else:
                result_chunks.append(result_audio)
        return result_chunks

    def inference(
        self,
        text: str,
        speaker_wav: Union[str, List],
        language: str = "ru",
        speed: float = 1,
        length_penalty: int = 1,
        repetition_penalty: float = 10.0,
        temperature: float = 0.3,
    ):
        if self.checkpoint is None:
            AttributeError("Checkpoint not loaded. Use 'XTTSv2.tts()'.")
        gpt_cond_latent, speaker_embedding = self.get_cond_latents(speaker_wav)

        output = self.model.inference(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            speed = speed,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
        )
        return output["wav"]

    def get_cond_latents(self, speaker_wav: Union[str, List]):
        if self.checkpoint is None:
            AttributeError("Checkpoint not loaded. Use 'XTTSv2.tts()'.")
        gpt_cond_latent, speaker_embedding = self.model.get_conditioning_latents(
            audio_path=speaker_wav
        )
        return gpt_cond_latent, speaker_embedding
