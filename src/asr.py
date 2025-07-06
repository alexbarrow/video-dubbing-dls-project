
from typing import Dict, List, Optional

import torch
import whisper

from .heplers import read_json, write_json


class ASR:
    """
    A class for Automatic Speech Recognition (ASR) using the Whisper model.

    Args:
        model_type (str): The type of Whisper model to use. Default is "base".
        device (str): The device to run the model on. Default is the CUDA device if available, otherwise the CPU.
    """
    def __init__(
        self,
        model_type: str = "base",
        device="cuda" if torch.cuda.is_available() else "cpu", # TODO: config
    ):
        """
        Initializes the ASR class with a Whisper model.

        Args:
            model_type (str): The type of Whisper model to use. Default is "base".
            device (str): The device to run the model on. Default is the CUDA device if available, otherwise the CPU.
            save (bool): Save to json
        """
        self.model = whisper.load_model(model_type, device)
    
    def _transcribe_wav(self, wav_path: str) -> List[Dict]:
        """
        Transcribes an audio file.

        Args:
            wav_path (str): The path to the audio file to transcribe.
        Returns:
            List[Dict]: list of segments.
        """
        audio = whisper.load_audio(wav_path)
        results = self.model.transcribe(audio)
        segments = results.get("segments", [])
        if not segments:
            print("No segments")
            return []
        return segments


    def transcribe(self, json_path: str, output_dir: str = None, save: bool = True) -> Optional[Dict]:
        chunks_json = read_json(json_path)

        for chunk in chunks_json.values():
            asr_result = self._transcribe_wav(chunk["path"])
            item = [{"start": seg["start"], "end": seg["end"], "text": seg["text"]} for seg in asr_result]
            chunk["asr_result"] = item

        if save and output_dir is not None:
            write_json(chunks_json, output_dir)
        else:
            return chunks_json
