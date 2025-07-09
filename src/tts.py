from typing import Dict, Optional

from TTS.api import TTS

from src.helpers import split_long_string


class XTTSv2:
    def __init__(self, device: str = "cpu"):
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.model = TTS(model_name).to(device)

    def tts(
        self,
        text: str,
        speaker_wav: str,
        language: str = "ru",
        save: bool = False,
        output_path: Optional[str] = None,
        speed: float = 1,
        length_penalty: int = 1,
        repetition_penalty: float = 10.0,
    ):
        if save and output_path is not None:
            self.model.tts_to_file(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path,
                speed=speed
            )
        else:
            audio = self.model.tts(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
                speed=speed,
                length_penalty = length_penalty,
                repetition_penalty = repetition_penalty,
            )
            return audio

    def tts_chunks(
        self, chunks: Dict,
        speaker_wav: str,
        language: str = "ru",
        speed: float = 1,
        length_penalty: int = 1,
        repetition_penalty: float = 10.0,
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
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
            if cutoff:
                cut_idx = len(result_audio) - int(len(result_audio) * cutoff)
                result_chunks.append(result_audio[:cut_idx])
            else:
                result_chunks.append(result_audio)
        return result_chunks

    def finetune(self):
        pass
