from typing import Dict

from TTS.api import TTS

from src.helpers import split_by_punctuation


class XTTSv2:
    def __init__(self, device: str = "cpu"):
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        self.model = TTS(model_name).to(device)

    def tts(self, text, speaker_wav, language="ru", save: bool = False, output_path: str = None):
        if save and output_path is not None:
            self.model.tts_to_file(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_path
            )
        else:
            audio = self.model.tts(
                text = text,
                speaker_wav=speaker_wav,
                language=language,
            )
            return audio

    def tts_chunks(
            self, chunks: Dict,
            speaker_wav: str,
            language="ru", #pause_sec: float = 0.2
        ):
        for chunk in chunks.values():
            if not chunk["translated_text"]:
                chunk["tts_result"] = ""
                continue
            text = chunk["translated_text"]
            
            #pause = list(np.zeros(int(pause_sec * 24000), dtype=np.float32)) # take from tts tail
            result_audio = []

            text_list = split_by_punctuation(text)
            for txt in text_list:
                audio_seg = self.tts(txt, speaker_wav, language=language)
                result_audio.extend(list(audio_seg))
            chunk["tts_result"] = result_audio
        return chunks

    def finetune(self):
        pass
