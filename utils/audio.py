"""
Audio utilities — format conversion for Whisper/ASR models.
"""
import os
from pydub import AudioSegment


def convert_to_wav(filepath: str, output_dir: str, sample_rate: int = 16000) -> dict:
    """
    Convert any audio file to 16kHz mono WAV.

    Returns dict with: original_file, original_format, wav_file, wav_path,
    duration_seconds, duration_minutes
    """
    filename = os.path.basename(filepath)
    base_name = os.path.splitext(filename)[0]
    original_ext = os.path.splitext(filename)[1]
    wav_filename = base_name + ".wav"
    wav_path = os.path.join(output_dir, wav_filename)

    # pydub auto-detects format via ffmpeg
    audio = AudioSegment.from_file(filepath)
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    audio.export(wav_path, format="wav")

    duration_sec = len(audio) / 1000.0

    return {
        "original_file": filename,
        "original_format": original_ext,
        "wav_file": wav_filename,
        "wav_path": wav_path,
        "duration_seconds": round(duration_sec, 1),
        "duration_minutes": round(duration_sec / 60.0, 2),
    }
