from typing import Dict, List

from silero_vad import get_speech_timestamps, load_silero_vad, read_audio


def find_timestamps(chunks: Dict, threshold: float = 1):
    model = load_silero_vad()
    for chunk in chunks.values():
        wav_path = chunk["path"]
        wav = read_audio(wav_path)
        raw_timestamps = get_speech_timestamps(
            wav,
            model,
            return_seconds=True,  # Return speech timestamps in seconds (default is samples)
            )
        timestamps = merge_timestamps(raw_timestamps, threshold=threshold)
        chunk["speech_boundary"] = timestamps

    return chunks

def merge_timestamps(timestamps: List[Dict], threshold: float):
    if not timestamps:
        return []
    
    merged = [timestamps[0]]
    
    for current in timestamps[1:]:
        last = merged[-1]
        if current['start'] - last['end'] <=threshold:
            merged[-1] = {'start': last['start'], 'end': max(last['end'], current['end'])}
        else:
            merged.append(current)
    
    return merged
