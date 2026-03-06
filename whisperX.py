import whisperx
import gc
import torch
import warnings
from contextlib import contextmanager
import argparse
import sys
import os

# Suppress warnings
warnings.filterwarnings("ignore")

# PyTorch 2.6+ compatibility: Create context manager for safe loading
@contextmanager
def allow_pickle_load():
    """Allow unsafe pickle loads for model weights"""
    import pickle
    old_load_reduce = pickle.load
    
    def new_load(*args, **kwargs):
        return old_load_reduce(*args, **kwargs)
    
    try:
        # Patch torch.load to disable weights_only during model loading
        import lightning_fabric.utilities.cloud_io as cloud_io
        original_torch_load = torch.load
        
        def relaxed_torch_load(path, **kwargs):
            # Force weights_only=False for compatibility
            kwargs['weights_only'] = False
            return original_torch_load(path, **kwargs)
        
        torch.load = relaxed_torch_load
        cloud_io.torch.load = relaxed_torch_load
        yield
    finally:
        torch.load = original_torch_load
        cloud_io.torch.load = original_torch_load

import time
from datetime import datetime
from pathlib import Path
import json
import librosa
import soundfile as sf

# Detect device: use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="WhisperX transcription with speaker diarization")
parser.add_argument("audio_file", help="Path to audio file (.wav format)")
parser.add_argument("--model", type=str, default="small.en", 
                   help="WhisperX model size (tiny/base/small/medium/large/large-v2/large-v3, default: small.en)")
parser.add_argument("--num-speakers", type=int, default=2,
                   help="Expected number of speakers (default: 2, use None for auto-detect)")
parser.add_argument("--output-dir", type=str, default=None,
                   help="Custom output directory (default: transcription_output_<timestamp>)")

args = parser.parse_args()

# Validate audio file exists
audio_file = args.audio_file
if not os.path.exists(audio_file):
    print(f"✗ Error: Audio file not found: {audio_file}", flush=True)
    sys.exit(1)

print(f"Audio file: {audio_file}")

# Configuration
batch_size = 16 # reduce if low on GPU mem
compute_type = "int8" if device == "cuda" else "float32"  # int8 for GPU, float32 for CPU

# Speaker diarization parameters
num_speakers = args.num_speakers
min_speakers = num_speakers if num_speakers else 1
max_speakers = num_speakers if num_speakers else 6

# 1. Transcribe with original whisper (batched)
print("Loading WhisperX model...")
print(f"  Model: {args.model}", flush=True)
print(f"  Device: {device}, Compute type: {compute_type}", flush=True)
try:
    with allow_pickle_load():
        model = whisperx.load_model(args.model, device, compute_type=compute_type)
    print("✓ Model loaded successfully", flush=True)
except Exception as e:
    print(f"✗ Model loading failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

print("Loading audio...")
print(f"  File: {audio_file}", flush=True)
audio = whisperx.load_audio(audio_file)
print(f"  Audio loaded successfully", flush=True)

start = time.time()
print("Transcribing...")
result = model.transcribe(audio, batch_size=batch_size)
end = time.time()
print(f"Transcription time = {end - start}s") # before alignment
print(f"Language detected: {result.get('language', 'unknown')}")
print(f"Number of segments: {len(result.get('segments', []))}")

# Show first few segments
# if result.get('segments'):
#     print("\nFirst 3 segments:")
#     for i, seg in enumerate(result['segments'][:3]):
#         print(f"  [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text'][:100]}")

# delete model if low on GPU resources
gc.collect(); torch.cuda.empty_cache(); del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# delete model if low on GPU resources
gc.collect(); torch.cuda.empty_cache(); del model_a

# 3. Assign speaker labels
from whisperx.diarize import DiarizationPipeline
print("Loading diarization model...", flush=True)
try:
    with allow_pickle_load():
        diarize_model = DiarizationPipeline(use_auth_token=True, device=device)
    print("✓ Diarization model loaded successfully", flush=True)
except Exception as e:
    print(f"✗ Diarization model loading failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1)

# add min/max number of speakers if known
print("Running speaker diarization...", flush=True)
print(f"  Expected speakers: min={min_speakers}, max={max_speakers}", flush=True)
try:
    with allow_pickle_load():
        diarize_segments = diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)
    print("✓ Diarization completed", flush=True)
except Exception as e:
    print(f"✗ Diarization failed: {e}", flush=True)
    import traceback
    traceback.print_exc()
    exit(1)

result = whisperx.assign_word_speakers(diarize_segments, result)
print("\nDiarization segments:")
print(diarize_segments)
print("\nTranscription with speaker labels:")
print(result["segments"]) # segments are now assigned speaker IDs

# Create timestamped output folder
if args.output_dir:
    output_dir = Path(args.output_dir)
else:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"transcription_output_{timestamp}")
output_dir.mkdir(exist_ok=True)

# Create subdirectories for audio and transcripts
audio_dir = output_dir / "audio_clips"
transcript_dir = output_dir / "transcripts"
audio_dir.mkdir(exist_ok=True)
transcript_dir.mkdir(exist_ok=True)

print(f"\nSaving results to {output_dir}...", flush=True)

# Load audio data for clipping
print("Loading audio for clipping...", flush=True)
audio_data, sr = librosa.load(audio_file, sr=16000)

# Organize segments by speaker
speaker_segments = {}
for segment in result["segments"]:
    speaker = segment.get("speaker", "UNKNOWN")
    if speaker not in speaker_segments:
        speaker_segments[speaker] = []
    speaker_segments[speaker].append(segment)

# Save combined transcription file
try:
    combined_file = output_dir / "transcription_with_speakers.txt"
    with open(combined_file, "w", encoding="utf-8") as f:
        f.write("="*80 + "\n")
        f.write("WHISPERX TRANSCRIPTION WITH SPEAKER DIARIZATION\n")
        f.write("="*80 + "\n\n")
        
        f.write("DIARIZATION SEGMENTS:\n")
        f.write("-"*80 + "\n")
        f.write(str(diarize_segments) + "\n\n")
        
        f.write("TRANSCRIPTION WITH SPEAKER LABELS:\n")
        f.write("-"*80 + "\n")
        for i, segment in enumerate(result["segments"]):
            speaker = segment.get("speaker", "UNKNOWN")
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            text = segment.get("text", "").strip()
            f.write(f"\n[Segment {i+1}] [{speaker}] ({start:.2f}s - {end:.2f}s)\n")
            f.write(f"{text}\n")
    
    print(f"✓ Combined transcription saved to {combined_file}", flush=True)
except Exception as e:
    print(f"✗ Error saving combined file: {e}", flush=True)
    import traceback
    traceback.print_exc()

# Save per-speaker transcript files and audio clips
try:
    for speaker, segments in sorted(speaker_segments.items()):
        # Create speaker-specific transcript file
        speaker_filename = speaker.replace(" ", "_")
        transcript_file = transcript_dir / f"{speaker_filename}_transcript.txt"
        
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(f"TRANSCRIPTION FOR {speaker}\n")
            f.write("="*80 + "\n\n")
            
            for i, segment in enumerate(segments):
                start = segment.get("start", 0)
                end = segment.get("end", 0)
                text = segment.get("text", "").strip()
                f.write(f"[{i+1}] ({start:.2f}s - {end:.2f}s)\n")
                f.write(f"{text}\n\n")
        
        print(f"✓ Speaker transcript saved: {transcript_file}", flush=True)
        
        # Extract and save audio clips for each segment
        speaker_audio_dir = audio_dir / speaker_filename
        speaker_audio_dir.mkdir(exist_ok=True)
        
        for i, segment in enumerate(segments):
            start = segment.get("start", 0)
            end = segment.get("end", 0)
            
            # Convert time to sample indices
            start_sample = int(start * sr)
            end_sample = int(end * sr)
            
            # Extract audio clip
            audio_clip = audio_data[start_sample:end_sample]
            
            # Save audio clip
            clip_file = speaker_audio_dir / f"{speaker_filename}_{i+1:03d}.wav"
            sf.write(clip_file, audio_clip, sr)
        
        print(f"✓ Saved {len(segments)} audio clips for {speaker}", flush=True)
    
    print(f"\n✓ All results saved to: {output_dir.absolute()}", flush=True)
    
except Exception as e:
    print(f"✗ Error saving speaker files: {e}", flush=True)