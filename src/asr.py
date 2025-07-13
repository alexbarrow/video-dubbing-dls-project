
from typing import Dict, List, Optional

import numpy as np
import whisper

from src.helpers import write_json


class ASR:
    """
    A class for Automatic Speech Recognition (ASR) using the Whisper model.

    Args:
        model_type (str): The type of Whisper model to use. Default is "base".
        device (str): The device to run the model on. Default is the CUDA device if available, otherwise the CPU.
    """

    def __init__(self, model_type: str = "base", device: str = "cpu"):
        """
        Initializes the ASR class with a Whisper model.

        Args:
            model_type (str): The type of Whisper model to use. Default is "base".
            device (str): The device to run the model on. Default is the CUDA device if available, otherwise the CPU.
            save (bool): Save to json
        """
        self.model = whisper.load_model(model_type, device)
    
    def _transcribe_wav(self, wav: np.ndarray) -> List[Dict]:
        """
        Transcribes an audio file.

        Args:
            wav_path (str): The path to the audio file to transcribe.
        Returns:
            List[Dict]: list of segments.
        """
        results = self.model.transcribe(wav)
        segments = results.get("segments", [])
        if not segments:
            print("No segments")
            return []
        return segments


    def transcribe(self, chunks_json: Dict, output_dir: str = None, save: bool = True) -> Optional[Dict]:
        for chunk in chunks_json.values():
            chunk_asr = []
            for seg in chunk["speech_boundary"]:
                start, end = seg["start"], seg["end"],
                audio = whisper.load_audio(chunk["path"])
                audio = audio[start:end]

                asr_result = self._transcribe_wav(audio)
                if not asr_result:
                    chunk["asr_result"] = {}
                    continue
                item = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in asr_result]
                chunk_asr.append(item)
            chunk["asr_result"] = chunk_asr

        if save and output_dir is not None:
            write_json(chunks_json, output_dir)
        else:
            return chunks_json
