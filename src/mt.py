from typing import Dict, Union

from transformers import MarianMTModel, MarianTokenizer


class MT:
    """
    A class for Machine Translation (MT) using the MarianMT model.
    """
    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-en-ru"):
        """
        Initializes the MT class with a MarianMT model.

        Args:
            model_name (str): The name of the MarianMT model to use. Default is "Helsinki-NLP/opus-mt-en-ru".
        """
        self.tokenizer = MarianTokenizer.from_pretrained(model_name)
        self.model=MarianMTModel.from_pretrained(model_name)

    def translate(self, text: Union[str, Dict]) -> Union[str, Dict]:
        """
        Translates the input text from the source language to the target language.
        Args:
            text (Union[str, Dict]): The text to translate. Can be a string or a dictionary.
        Returns:
            Union[str, Dict]: The translated text (if dict, the keys are the same as the input dict).
        """
        if isinstance(text, str):
            inputs = self.tokenizer.encode(text, return_tensors="pt", padding=True)
            outputs = self.model.generate(inputs)
            translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            return translated_text
        
        elif isinstance(text, dict):
            for chunk in text.values():
                if not chunk["asr_result"]:
                    chunk["translated_text"] = ""
                    continue
                chunk_text = ""
                for seg in chunk["asr_result"]:
                    temp = " ".join([s["text"].strip() for s in seg])
                    chunk_text += " " + temp

                inputs = self.tokenizer.encode(chunk_text, return_tensors="pt", padding=True)
                outputs = self.model.generate(inputs)
                translated_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
                chunk["translated_text"] = translated_text
            
            return text
        else:
            raise ValueError("Machine Translation: Input must be a string or a dictionary")
