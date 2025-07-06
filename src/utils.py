import os
from typing import Dict, List, Union

from pydub import AudioSegment
from pydub.silence import split_on_silence
from simalign import SentenceAligner

from .heplers import write_json


def get_chunks(
    wav_path: str,
    output_dir: str,
    chunk_size: int = 30,
    min_silence_len=700,
    silence_thresh=-40,
    keep_silence=300,
    save=True,
) -> Union[List[str], List[AudioSegment]]:
    audio = AudioSegment.from_file(wav_path)
    chunks = split_on_silence(
        audio,
        min_silence_len=min_silence_len,
        silence_thresh=silence_thresh,
        keep_silence=keep_silence
    )

    target_length = chunk_size * 1000
    output_chunks = [chunks[0]]
    for chunk in chunks[1:]:
        if len(output_chunks[-1]) < target_length:
            output_chunks[-1] += chunk
        else:
            output_chunks.append(chunk)

    if save:
        os.makedirs(output_dir, exist_ok=True)
        chunk_json = {}
        current_offset = 0

        for i, chunk in enumerate(output_chunks):
            chunk_name = f"chunk_{i:03d}.wav"
            out_path = os.path.join(output_dir, chunk_name)
            chunk.export(out_path, format="wav")
            item = {"path": out_path, "offset": current_offset / 1000}
            chunk_json[i] = item
            current_offset += len(chunk)

        write_json(chunk_json, output_dir)
    else:
        return output_chunks


def alligment(chunks: Dict):

    aligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="i")

    for chunk in chunks.values():
        chunk_text = " ".join([seg["text"].strip() for seg in chunk["asr_result"]])

        aligns = aligner.get_word_aligns(chunk_text, chunk["translated_text"])
        alignment_pairs = aligns['itermax']
        ru_tokens = chunk["translated_text"].split()

        word_to_segment = []
        word_index = 0
        for i, seg in enumerate(chunk["asr_result"]):
            word_count = len(seg["text"].split())
            word_to_segment.extend([i] * word_count)
            word_index += word_count

        segment_translations = [[] for _ in chunk["asr_result"]]
        for src_idx, tgt_idx in alignment_pairs:
            if src_idx < len(word_to_segment) and tgt_idx < len(ru_tokens):
                segment_index = word_to_segment[src_idx]
                segment_translations[segment_index].append(ru_tokens[tgt_idx])

        final_segments = []
        for i, seg in enumerate(chunk["asr_result"]):
            start = seg["start"] + chunk["offset"]
            end = seg["end"] + chunk["offset"]
            translation = " ".join(segment_translations[i])
            final_segments.append({
                "start": round(start, 2),
                "end": round(end, 2),
                "translated_text": translation
            })
        chunk["alligment"] = final_segments
    return chunks