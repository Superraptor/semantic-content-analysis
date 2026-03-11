# -----------------------------------------------------------------------------
# Setup Instructions (run once before the first execution):
#
# 1) Create and activate a Python virtual environment (recommended):
#    - Windows (PowerShell):
#        python -m venv .venv
#        .\.venv\Scripts\Activate.ps1
#    - macOS/Linux (bash/zsh):
#        python -m venv .venv
#        source .venv/bin/activate
#
# 2) Install dependencies from requirements.txt:
#        pip install -r requirements.txt
#
# 3) Run the script:
#        python whisperX.py <path_to_audio.wav>
#
# Notes:
#  - The first run may download large models (SpeechBrain, SentenceTransformers, etc.).
#  - If you want to re-run with a clean environment, delete the .venv folder and repeat step 1.
# -----------------------------------------------------------------------------

import whisperx
import gc
import torch
import warnings
from contextlib import contextmanager
import argparse
import sys
import os

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util
from speechbrain.pretrained import EncoderClassifier
from textblob import TextBlob

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


def load_analysis_models(device="cpu"):
    """Load models used for emotion, sentiment, and similarity scoring."""
    global emotion_classifier, sentiment_tokenizer, sentiment_model, similarity_model

    # Emotion (SpeechBrain)
    try:
        print("Loading SpeechBrain emotion model...", flush=True)
        emotion_classifier = EncoderClassifier.from_hparams(
            source="speechbrain/emotion-recognition-wav2vec2",
            savedir="pretrained_models/emotion",
            run_opts={"device": device}
        )
        print("✓ Emotion model loaded", flush=True)
    except Exception as e:
        print(f"✗ Emotion model failed to load: {e}", flush=True)
        emotion_classifier = None

    # Sentiment (CardiffNLP)
    try:
        print("Loading sentiment model...", flush=True)
        sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "cardiffnlp/twitter-roberta-base-sentiment"
        )
        sentiment_model.eval()
        if device == "cuda":
            sentiment_model.to(device)
        print("✓ Sentiment model loaded", flush=True)
    except Exception as e:
        print(f"✗ Sentiment model failed to load: {e}", flush=True)
        sentiment_tokenizer = None
        sentiment_model = None

    # Similarity (SentenceTransformers)
    try:
        print("Loading similarity model...", flush=True)
        similarity_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device=device)
        print("✓ Similarity model loaded", flush=True)
    except Exception as e:
        print(f"✗ Similarity model failed to load: {e}", flush=True)
        similarity_model = None


def predict_emotion_for_speaker(speaker_audio_dir):
    """Predict the predominant emotion for a speaker given their audio clips."""
    if emotion_classifier is None:
        return None

    labels = []
    for wav_path in Path(speaker_audio_dir).glob("*.wav"):
        try:
            out_prob, score, idx, text_lab = emotion_classifier.classify_file(str(wav_path))
            if text_lab:
                labels.append(text_lab[0])
        except Exception:
            continue

    if not labels:
        return None

    from collections import Counter
    return Counter(labels).most_common(1)[0][0]


def analyze_sentiment_text(text):
    """Analyze sentiment of text using the loaded sentiment model."""
    if sentiment_model is None or sentiment_tokenizer is None or not text:
        return {"label": None, "confidence": None, "score": None}

    inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = sentiment_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]
        score, idx = torch.max(probs, dim=-1)
        label = sentiment_model.config.id2label.get(idx.item(), None)

    # Map to 1-5 scale (negative=1, neutral=3, positive=5)
    mapping = {"LABEL_0": 1.0, "LABEL_1": 3.0, "LABEL_2": 5.0}
    numeric = mapping.get(label, None)

    return {"label": label, "confidence": float(score), "score": numeric}


def _get_sentence_embedding(text):
    if similarity_model is None or not text:
        return None

    if text in _similarity_embeddings:
        return _similarity_embeddings[text]

    emb = similarity_model.encode(text, convert_to_tensor=True)
    _similarity_embeddings[text] = emb
    return emb


def keyword_overlap(text1, text2):
    """Compute keyword overlap ratio between two texts."""
    try:
        blob1 = TextBlob(text1)
        blob2 = TextBlob(text2)
        keywords1 = set(blob1.noun_phrases)
        keywords2 = set(blob2.noun_phrases)

        if not keywords1 or not keywords2:
            return 0.0

        overlap = keywords1.intersection(keywords2)
        return len(overlap) / max(len(keywords1), len(keywords2))
    except Exception:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        return intersection / union if union > 0 else 0.0


def calculate_similarity(text1, text2):
    """Compute cosine similarity between two pieces of text."""
    emb1 = _get_sentence_embedding(text1)
    emb2 = _get_sentence_embedding(text2)
    if emb1 is None or emb2 is None:
        return None

    sim = util.pytorch_cos_sim(emb1, emb2)
    return float(sim.item())


def save_analysis_results(output_dir, audio_file, speaker_segments):
    """Save speaker-level and conversation-level analysis to Excel files."""
    # Build transcripts per speaker
    speaker_texts = {
        speaker: " ".join([seg.get("text", "") for seg in segments]).strip()
        for speaker, segments in speaker_segments.items()
    }

    # Compute speaker-level metrics
    rows = []
    for speaker, text in sorted(speaker_texts.items()):
        sentiment = analyze_sentiment_text(text)

        # Emotion: use speaker audio clips if available
        speaker_dir = Path(output_dir) / "audio_clips" / speaker.replace(" ", "_")
        emotion = predict_emotion_for_speaker(speaker_dir) if speaker_dir.exists() else None

        rows.append({
            "Audio_File": Path(audio_file).name,
            "Speaker_ID": speaker,
            "Emotion": emotion,
            "Sentiment_Label": sentiment.get("label"),
            "Sentiment_Score": sentiment.get("score"),
            "Sentiment_Confidence": sentiment.get("confidence"),
            "Transcript": text,
        })

    speaker_df = pd.DataFrame(rows)

    # Calculate pairwise similarity (averaged per speaker)
    similarity_values = []
    for i, speaker_a in enumerate(speaker_df["Speaker_ID"]):
        text_a = speaker_texts.get(speaker_a, "")
        sims = []
        for j, speaker_b in enumerate(speaker_df["Speaker_ID"]):
            if i == j:
                continue
            text_b = speaker_texts.get(speaker_b, "")
            sim = calculate_similarity(text_a, text_b)
            if sim is not None:
                sims.append(sim)
                similarity_values.append(sim)

        speaker_df.loc[speaker_df["Speaker_ID"] == speaker_a, "Avg_Similarity_To_Others"] = (
            float(np.mean(sims)) if sims else None
        )

    # Calculate pairwise keyword overlap (averaged per speaker)
    overlap_values = []
    for i, speaker_a in enumerate(speaker_df["Speaker_ID"]):
        text_a = speaker_texts.get(speaker_a, "")
        overlaps = []
        for j, speaker_b in enumerate(speaker_df["Speaker_ID"]):
            if i == j:
                continue
            text_b = speaker_texts.get(speaker_b, "")
            overlap = keyword_overlap(text_a, text_b)
            overlaps.append(overlap)
            overlap_values.append(overlap)

        speaker_df.loc[speaker_df["Speaker_ID"] == speaker_a, "Avg_Keyword_Overlap"] = (
            float(np.mean(overlaps)) if overlaps else None
        )

    # Conversation-level summary
    conv = {
        "Audio_File": Path(audio_file).name,
        "Num_Speakers": len(speaker_texts),
        "Avg_Similarity": float(np.mean(similarity_values)) if similarity_values else None,
        "Avg_Keyword_Overlap": float(np.mean(overlap_values)) if overlap_values else None,
        "Sentiment_Mean": float(np.nanmean([r["Sentiment_Score"] for r in rows if r.get("Sentiment_Score") is not None])) if any(r.get("Sentiment_Score") is not None for r in rows) else None,
        "Sentiment_Std": float(np.nanstd([r["Sentiment_Score"] for r in rows if r.get("Sentiment_Score") is not None])) if any(r.get("Sentiment_Score") is not None for r in rows) else None,
        "Speaker_Sentiments": json.dumps({r["Speaker_ID"]: {"label": r["Sentiment_Label"], "score": r["Sentiment_Score"], "confidence": r["Sentiment_Confidence"]} for r in rows}),
    }
    conv_df = pd.DataFrame([conv])

    # Save Excel files (optional)
    speaker_excel = Path(output_dir) / "consolidated_results.xlsx"
    conversation_excel = Path(output_dir) / "consolidated_conversation_summary.xlsx"

    try:
        speaker_df.to_excel(speaker_excel, index=False)
        conv_df.to_excel(conversation_excel, index=False)
        print(f"✓ Analysis results saved: {speaker_excel}", flush=True)
        print(f"✓ Conversation summary saved: {conversation_excel}", flush=True)
    except Exception as e:
        print(f"⚠️ Could not save Excel output (openpyxl might be missing): {e}", flush=True)

    # Always save CSV for compatibility
    speaker_df.to_csv(Path(output_dir) / "consolidated_results.csv", index=False)
    conv_df.to_csv(Path(output_dir) / "consolidated_conversation_summary.csv", index=False)
    print(f"✓ CSV output saved to {output_dir}", flush=True)


import time
from datetime import datetime
from pathlib import Path
import json
import numpy as np
import librosa
import soundfile as sf

# Detect device: use CUDA if available, otherwise CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Analysis model references (loaded later)
emotion_classifier = None
sentiment_tokenizer = None
sentiment_model = None
similarity_model = None

# Cached sentence embeddings for similarity
_similarity_embeddings = {}

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

# Load analysis models (emotion, sentiment, similarity)
load_analysis_models(device=device)

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
    
    # Generate analysis outputs (emotion, sentiment, similarity) and save Excel/CSV files
    try:
        save_analysis_results(output_dir, audio_file, speaker_segments)
    except Exception as e:
        print(f"✗ Error generating analysis outputs: {e}", flush=True)
    
    print(f"\n✓ All results saved to: {output_dir.absolute()}", flush=True)
    
except Exception as e:
    print(f"✗ Error saving speaker files: {e}", flush=True)