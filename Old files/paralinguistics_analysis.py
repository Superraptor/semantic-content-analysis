#!/usr/bin/env python
"""
Paralinguistics Analysis Script
Author: Rishabh Jain
Modified: November 2025
Version: 2.0 (Consolidated)

DESCRIPTION:
    Comprehensive audio paralinguistic analysis tool that analyzes emotion, personality traits,
    conversation dynamics, sentiment, and content similarity from audio files. Supports both
    HuggingFace transformer models (when available) and offline heuristic-based fallbacks.

FEATURES:
    - Speaker diarization (automatic identification of who spoke when)
    - Emotion recognition from speech (8 emotions: Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised)
    - Personality trait analysis (Extraversion, Openness, Conscientiousness)
    - Conversation comprehension scoring (emotion alignment, response time, keyword overlap, backchannels)
    - Sentiment analysis of transcribed speech (1-5 scale)
    - Content similarity analysis between speakers
    - Multi-file batch processing with consolidated output
    - Bootstrap iteration support for statistical analysis

MODELS USED (when available in cache):
    1. pyannote/speaker-diarization-3.1 - Professional speaker diarization (v3.1)
    2. superb/wav2vec2-large-superb-er - Emotion recognition from audio
    3. nlptown/bert-base-multilingual-uncased-sentiment - Sentiment analysis from text
    4. sentence-transformers/all-MiniLM-L6-v2 - Semantic similarity between texts
    5. speechbrain/spkrec-ecapa-voxceleb - Speaker embedding model (required for diarization)
    6. vosk-model-en-us-0.22 - Offline speech recognition (~1.8GB)
       Download from: https://alphacephei.com/vosk/models
       Place model folder at: ~/.cache/vosk/models/vosk-model-en-us-0.22

OFFLINE FALLBACKS:
    - Speaker diarization: Energy/pitch/MFCC-based heuristic analysis
    - Emotion recognition: RMS energy-based heuristic
    - Sentiment analysis: Neutral scores (3.0)
    - Similarity: Jaccard similarity (word overlap)

USAGE:
    # Single audio file analysis
    python paralinguistics_analysis.py audio.wav

    # Multiple audio files (consolidated output)
    python paralinguistics_analysis.py audio1.wav audio2.wav audio3.wav
    python paralinguistics_analysis.py folder/*.wav

    # Bootstrap mode (run N iterations for statistical analysis)
    python paralinguistics_analysis.py audio.wav --bootstrap 100

    # Custom output directory
    python paralinguistics_analysis.py audio.wav --output-dir custom_results

    # Batch mode with optimizations
    python paralinguistics_analysis.py folder/*.wav --batch-mode

COMMAND-LINE OPTIONS:
    audiofiles (positional, str, required)
        One or more audio file paths to analyze. Supports .wav format.
        Example: audio1.wav audio2.wav audio3.wav

    --output-dir (str, optional, default: analysis_results_<timestamp>)
        Custom output directory for results. Directory will be created if it doesn't exist.
        Example: --output-dir results_jan2025

    --batch-mode (flag, optional)
        Enable batch processing optimizations for multiple files. Uses process-level
        parallelization and optimized memory management.
        Example: --batch-mode

    --bootstrap (int, optional)
        Run N bootstrap iterations on the audio file(s). Each iteration analyzes the same
        file independently, useful for statistical analysis and confidence intervals.
        Example: --bootstrap 100

    --models-cache-dir (str, optional, default: ~/.cache)
        Custom models cache directory for both HuggingFace and Vosk models.
        HuggingFace models: <cache_dir>/huggingface/
        Vosk models: <cache_dir>/vosk/models/
        Example: --models-cache-dir /custom/path/to/cache

    --delete-temp-files (flag, optional)
        Delete intermediate speaker audio files after processing is complete. By default,
        speaker audio clips are preserved in the diarized_audio_clips/ directory.
        Example: --delete-temp-files

OUTPUT FILES:
    analysis_results_YYYYMMDD_HHMMSS/
    ├── consolidated_results.csv
    │   Speaker-level data: One row per speaker per audio file
    │   Columns: Audio_File, Speaker_ID, Emotion, Pitch_Mean, Pitch_Std, Intensity_Mean,
    │            Intensity_Std, MFCC_Mean, MFCC_Std, Tempo, Extraversion, Openness,
    │            Conscientiousness, Processing_Time_Sec
    │
    ├── consolidated_conversation_summary.csv
    │   Conversation-level data: One row per audio file
    │   Columns: Audio_File, Num_Speakers, Duration_Sec, Overall_Similarity_Score,
    │            Overall_Comprehension_Score, Speaker_Sentiments_JSON, Avg_Response_Time,
    │            Emotion_Alignment_Ratio, Keyword_Overlap_Ratio, Backchannel_Count,
    │            Total_Processing_Time_Sec
    │
    ├── paralinguistic_analysis.log
    │   Detailed processing log with timestamps, warnings, and error messages
    │
    ├── diarized_audio_clips/
    │   ├── audio1_speaker_00.wav  # Speaker 0 from audio1
    │   ├── audio1_speaker_01.wav  # Speaker 1 from audio1
    │   ├── audio1_speaker_02.wav  # Speaker 2 from audio1 (if >2 speakers detected)
    │   └── audio2_speaker_00.wav  # Speaker 0 from audio2
    │
    └── diarized_audio_transcripts/
        ├── audio1_speaker_00.txt  # Transcript for speaker 0 from audio1
        ├── audio1_speaker_01.txt  # Transcript for speaker 1 from audio1
        ├── audio1_speaker_02.txt  # Transcript for speaker 2 from audio1 (if >2 speakers)
        └── audio2_speaker_00.txt  # Transcript for speaker 0 from audio2

NOTES:
    - Script attempts to load HuggingFace models from cache first (no internet required)
    - If models not available, automatically falls back to offline heuristic methods
    - Supports 2+ speakers per audio file (dynamically detected)
    - All processing is logged to paralinguistic_analysis.log
    - Bootstrap mode useful for estimating confidence intervals and variance

REQUIREMENTS:
    Python 3.8+
    librosa, numpy, pandas, scipy, textblob, nltk
    transformers, torch (for HuggingFace models)
    pyannote.audio (for speaker diarization model)
    pydub (for audio manipulation)
    vosk (for speech transcription) - Install: pip install vosk
    Vosk model required - Download vosk-model-en-us-0.22 (~1.8GB) from:
    https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip
    Extract to: ~/.cache/vosk/models/

EXAMPLE WORKFLOW:
    1. Download HuggingFace models using download_and_test_huggingface_models.py
    2. Run analysis: python paralinguistics_analysis.py audio.wav
    3. Check output CSV files for results
    4. Use bootstrap mode for statistical analysis: --bootstrap 100
"""

# ============================================================================
# IMPORTS & DEPENDENCIES
# ============================================================================

# Standard library imports
import argparse
import gc
import glob
import json
import logging
import multiprocessing as mp
import os
import re
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
from io import StringIO
from pathlib import Path

# Third-party imports - Core
import librosa
import numpy as np
import pandas as pd

# Third-party imports - Audio processing
from pydub import AudioSegment

# Third-party imports - NLP
from textblob import TextBlob
from scipy.spatial.distance import cosine

# Third-party imports - Speech recognition
from vosk import Model, KaldiRecognizer
import wave
import json

# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*numba.*")

# ============================================================================
# GLOBAL CONFIGURATION
# ============================================================================

# Feature availability flags (set during HF model loading)
EMOTION_HF_AVAILABLE = False
DIARIZATION_HF_AVAILABLE = False
SENTIMENT_HF_AVAILABLE = False
SIMILARITY_HF_AVAILABLE = False
TRANSCRIPTION_VOSK_AVAILABLE = False

# Global model references (loaded once at startup)
emotion_model = None
emotion_feature_extractor = None
diarization_pipeline = None
sentiment_tokenizer = None
sentiment_model = None
similarity_tokenizer = None
similarity_model = None
vosk_model = None

# Processing configuration
MAX_CACHE_SIZE = 100  # Maximum number of cached feature extractions
DEFAULT_SAMPLE_RATE = 16000  # Standard sample rate for models
VOSK_MODEL_PATH = None  # Will be set dynamically based on cache directory


def get_vosk_model_path(custom_cache_dir=None):
    """
    Get the Vosk model path based on cache directory.

    Args:
        custom_cache_dir (str): Custom cache directory root, or None for default

    Returns:
        str: Full path to Vosk model directory
    """
    if custom_cache_dir:
        # Custom cache: <cache_dir>/vosk/models/vosk-model-en-us-0.22
        return os.path.join(custom_cache_dir, "vosk", "models", "vosk-model-en-us-0.22")
    else:
        # Default: ~/.cache/vosk/models/vosk-model-en-us-0.22
        return os.path.expanduser("~/.cache/vosk/models/vosk-model-en-us-0.22")

# ============================================================================
# OFFLINE FALLBACK FUNCTIONS (Always Available)
# ============================================================================

def keyword_overlap_offline(text1, text2):
    """
    Calculate keyword overlap using simple word-based overlap.

    This is the offline fallback when HuggingFace models are not available.
    Uses basic set intersection of words.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    overlap = len(words1.intersection(words2))
    return overlap / max(len(words1), len(words2))


def calculate_similarity_offline(text1, text2):
    """
    Calculate similarity using Jaccard similarity (word overlap).

    This is the offline fallback when HuggingFace models are not available.
    Uses intersection over union of word sets.

    Args:
        text1 (str): First text to compare
        text2 (str): Second text to compare

    Returns:
        float: Jaccard similarity score (0.0 to 1.0)
    """
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1.intersection(words2))
    union = len(words1.union(words2))

    return intersection / union if union > 0 else 0.0


def analyze_sentiment_offline(text1, text2):
    """
    Return neutral sentiment scores (offline fallback).

    This is the offline fallback when HuggingFace models are not available.
    Returns neutral scores for all metrics.

    Args:
        text1 (str): First text (unused in offline mode)
        text2 (str): Second text (unused in offline mode)

    Returns:
        dict: Dictionary with neutral sentiment scores
            - text1_sentiment: 3.0 (neutral on 1-5 scale)
            - text2_sentiment: 3.0 (neutral on 1-5 scale)
            - sentiment_difference: 0.0
    """
    return {
        'text1_sentiment': 3.0,
        'text2_sentiment': 3.0,
        'sentiment_difference': 0.0
    }


def analyze_emotion_offline(y, sr):
    """
    Analyze emotion using RMS energy-based heuristic (offline fallback).

    This is the offline fallback when HuggingFace Wav2Vec2 model is not available.
    Uses audio energy levels as a proxy for emotional intensity.

    Args:
        y (numpy.ndarray): Audio waveform
        sr (int): Sample rate

    Returns:
        str: Predicted emotion (Neutral, Calm, Happy, Sad, Angry, Surprised)
    """
    rms_energy = np.mean(librosa.feature.rms(y=y))

    # Higher energy corresponds to more active emotions
    if rms_energy > 0.1:
        emotions = ["Happy", "Angry", "Surprised"]
    elif rms_energy > 0.05:
        emotions = ["Neutral", "Calm"]
    else:
        emotions = ["Sad", "Calm"]

    return np.random.choice(emotions)


def detect_speaker_changes(y, sr, duration):
    """
    Detect speaker changes using audio feature analysis.

    Analyzes energy, pitch, and MFCC changes to identify potential speaker transitions.
    Used by offline speaker diarization fallback.

    Args:
        y (numpy.ndarray): Audio waveform
        sr (int): Sample rate
        duration (float): Audio duration in seconds

    Returns:
        dict: Speaker segments {speaker_id: [(start, end), ...]}
    """
    frame_length = int(0.5 * sr)  # 0.5 second windows
    hop_length = int(0.25 * sr)   # 0.25 second hop

    # Extract features per frame
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
    pitch_values = []

    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch = pitches[index, t]
        pitch_values.append(pitch if pitch > 0 else 0)

    pitch_values = np.array(pitch_values)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)

    # Detect significant changes
    energy_changes = np.abs(np.diff(rms)) > (np.std(rms) * 0.5)
    pitch_changes = np.abs(np.diff(pitch_values)) > (np.std(pitch_values) * 0.5)

    # Combine change indicators
    combined_changes = energy_changes | pitch_changes

    # Find speaker boundaries
    times = librosa.frames_to_time(range(len(combined_changes)), sr=sr, hop_length=hop_length)
    speaker_boundaries = [0.0]

    for i, changed in enumerate(combined_changes):
        if changed and times[i] - speaker_boundaries[-1] > 2.0:  # Min 2 sec between changes
            speaker_boundaries.append(times[i])

    speaker_boundaries.append(duration)

    # Assign alternating speakers
    segments = {}
    for i in range(len(speaker_boundaries) - 1):
        speaker_id = f"SPEAKER_{i % 2:02d}"
        start_time = speaker_boundaries[i]
        end_time = speaker_boundaries[i + 1]

        if speaker_id not in segments:
            segments[speaker_id] = []

        segments[speaker_id].append((start_time, end_time))

    return segments


def diarize_speakers_offline(audio_file):
    """
    Perform speaker diarization using offline heuristic analysis.

    This is the offline fallback when pyannote.audio model is not available.
    Uses energy, pitch, and MFCC analysis to detect speaker changes.

    Args:
        audio_file (str): Path to audio file

    Returns:
        dict: Speaker segments {speaker_id: [(start, end), ...]}
    """
    logging.info("Using offline heuristic-based speaker diarization")

    try:
        y, sr = librosa.load(audio_file, sr=16000)
        duration = len(y) / sr

        speaker_segments = detect_speaker_changes(y, sr, duration)

        logging.info(f"Detected {len(speaker_segments)} speakers")
        for speaker, segments in speaker_segments.items():
            total_time = sum(end - start for start, end in segments)
            logging.info(f"{speaker}: {len(segments)} segments, {total_time:.2f}s total")

        return speaker_segments

    except Exception as e:
        logging.error(f"Enhanced diarization failed: {e}")
        logging.info("Using simple time-based split as fallback")

        try:
            audio = AudioSegment.from_wav(audio_file)
            duration = len(audio) / 1000.0
            logging.info(f"Simple fallback: dividing {duration:.2f}s equally between 2 speakers")
            return {
                "SPEAKER_00": [(0, duration/2)],
                "SPEAKER_01": [(duration/2, duration)]
            }
        except Exception as final_error:
            logging.error(f"Final fallback also failed: {final_error}")
            return {"SPEAKER_00": [(0, 10)]}  # Absolute minimal fallback


# ============================================================================
# HUGGINGFACE MODEL LOADING (Try Cache First)
# ============================================================================

def load_huggingface_models():
    """
    Attempt to load all HuggingFace models from cache.

    Sets global availability flags and model references. If models fail to load,
    the corresponding flag is set to False and offline fallbacks will be used.

    This function is called once at script startup.
    """
    global emotion_model, emotion_feature_extractor, EMOTION_HF_AVAILABLE
    global diarization_pipeline, DIARIZATION_HF_AVAILABLE
    global sentiment_tokenizer, sentiment_model, SENTIMENT_HF_AVAILABLE
    global similarity_tokenizer, similarity_model, SIMILARITY_HF_AVAILABLE

    # Load emotion recognition model
    try:
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        import torch

        logging.info("Loading emotion recognition model from cache...")
        emotion_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        emotion_model.eval()
        EMOTION_HF_AVAILABLE = True
        logging.info("Emotion model loaded successfully")
    except Exception as e:
        logging.info(f"Emotion model not available: {e}")
        logging.info("Will use offline emotion detection")
        EMOTION_HF_AVAILABLE = False

    # Load speaker diarization pipeline
    try:
        from pyannote.audio import Pipeline

        logging.info("Loading speaker diarization model from cache...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=True,
            device = "cuda" if torch.cuda.is_available() else "cpu"
        )
        DIARIZATION_HF_AVAILABLE = True
        logging.info("Diarization model loaded successfully")
    except Exception as e:
        logging.info(f"Diarization model not available: {e}")
        logging.info("Will use offline speaker diarization")
        DIARIZATION_HF_AVAILABLE = False

    # Load sentiment analysis model
    try:
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        logging.info("Loading sentiment analysis model from cache...")
        sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        sentiment_model.eval()
        SENTIMENT_HF_AVAILABLE = True
        logging.info("Sentiment model loaded successfully")
    except Exception as e:
        logging.info(f"Sentiment model not available: {e}")
        logging.info("Will use offline sentiment analysis")
        SENTIMENT_HF_AVAILABLE = False

    # Load similarity model
    try:
        from transformers import AutoTokenizer, AutoModel

        logging.info("Loading similarity model from cache...")
        similarity_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        similarity_model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        similarity_model.eval()
        SIMILARITY_HF_AVAILABLE = True
        logging.info("Similarity model loaded successfully")
    except Exception as e:
        logging.info(f"Similarity model not available: {e}")
        logging.info("Will use offline similarity calculation")
        SIMILARITY_HF_AVAILABLE = False

    # Log summary
    models_available = sum([EMOTION_HF_AVAILABLE, DIARIZATION_HF_AVAILABLE,
                           SENTIMENT_HF_AVAILABLE, SIMILARITY_HF_AVAILABLE])
    logging.info(f"HuggingFace models loaded: {models_available}/4")


def load_vosk_model(custom_cache_dir=None):
    """
    Load Vosk speech recognition model from local cache.

    Note: Vosk Python library (pip install vosk) must be installed,
    AND the language model must be downloaded separately.

    Args:
        custom_cache_dir (str): Custom cache directory root (optional)

    The model is expected at:
        - Default: ~/.cache/vosk/models/vosk-model-en-us-0.22
        - Custom: <cache_dir>/vosk/models/vosk-model-en-us-0.22

    Sets TRANSCRIPTION_VOSK_AVAILABLE flag and vosk_model global reference.
    If model fails to load, the flag is set to False and transcription will fail.
    """
    global vosk_model, TRANSCRIPTION_VOSK_AVAILABLE, VOSK_MODEL_PATH

    # Set the path dynamically based on cache directory
    VOSK_MODEL_PATH = get_vosk_model_path(custom_cache_dir)

    try:
        from vosk import Model

        logging.info(f"Loading Vosk transcription model from: {VOSK_MODEL_PATH}")

        if not os.path.exists(VOSK_MODEL_PATH):
            logging.error(f"Vosk model not found at: {VOSK_MODEL_PATH}")
            logging.error("Download vosk-model-en-us-0.22.zip from:")
            logging.error("  https://alphacephei.com/vosk/models")
            logging.error("Extract to: ~/.cache/vosk/models/")
            TRANSCRIPTION_VOSK_AVAILABLE = False
            return

        vosk_model = Model(VOSK_MODEL_PATH)
        TRANSCRIPTION_VOSK_AVAILABLE = True
        logging.info("Vosk model loaded successfully")

    except ImportError:
        logging.error("Vosk library not installed. Install with: pip install vosk")
        TRANSCRIPTION_VOSK_AVAILABLE = False
    except Exception as e:
        logging.error(f"Failed to load Vosk model: {e}")
        TRANSCRIPTION_VOSK_AVAILABLE = False



# ============================================================================
# CORE ANALYSIS WRAPPER FUNCTIONS (HuggingFace + Offline Fallbacks)
# ============================================================================

def analyze_emotion(y, sr):
    """
    Analyze emotion from audio waveform.

    Attempts HuggingFace Wav2Vec2 model first, falls back to offline heuristic.

    Args:
        y (numpy.ndarray): Audio waveform
        sr (int): Sample rate

    Returns:
        str: Predicted emotion
    """
    if EMOTION_HF_AVAILABLE:
        try:
            import torch

            # Ensure correct sample rate
            if sr != 16000:
                y = librosa.resample(y, orig_sr=sr, target_sr=16000)
                sr = 16000

            # Limit to 10 seconds
            max_samples = 10 * sr
            if len(y) > max_samples:
                start_idx = (len(y) - max_samples) // 2
                y = y[start_idx:start_idx + max_samples]

            inputs = emotion_feature_extractor(y, sampling_rate=sr, return_tensors="pt", padding=True)

            with torch.no_grad():
                logits = emotion_model(**inputs).logits

            emotion_idx = torch.argmax(logits, dim=-1).item()
            emotions = {0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
                       4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"}

            return emotions.get(emotion_idx, "Neutral")

        except Exception as e:
            logging.warning(f"HF emotion analysis failed: {e}, using offline fallback")
            return analyze_emotion_offline(y, sr)
    else:
        return analyze_emotion_offline(y, sr)


def diarize_speakers(audio_file):
    """
    Perform speaker diarization.

    Attempts pyannote.audio model first, falls back to offline heuristic.

    Args:
        audio_file (str): Path to audio file

    Returns:
        dict: Speaker segments {speaker_id: [(start, end), ...]}
    """
    # Helper: preprocess with webrtcvad to get speech-only regions
    def preprocess_for_diarization(in_file, target_sr=16000, aggressiveness=2):
        try:
            import wave, tempfile
            import webrtcvad

            # Export resampled mono 16-bit wav to temp
            audio = AudioSegment.from_file(in_file)
            audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2)
            tmp_wav = str(Path(tempfile.gettempdir()) / f"diarize_{Path(in_file).stem}.wav")
            audio.export(tmp_wav, format="wav")

            # Read raw pcm bytes
            with wave.open(tmp_wav, 'rb') as wf:
                sample_rate = wf.getframerate()
                pcm = wf.readframes(wf.getnframes())

            vad = webrtcvad.Vad(aggressiveness)

            frame_ms = 30
            frame_bytes = int(sample_rate * (frame_ms / 1000.0)) * 2

            speech_regions = []
            is_speech = False
            region_start = 0.0
            for i in range(0, len(pcm), frame_bytes):
                frame = pcm[i:i+frame_bytes]
                timestamp = (i / 2) / sample_rate
                try:
                    speech = vad.is_speech(frame, sample_rate)
                except Exception:
                    speech = False

                if speech and not is_speech:
                    is_speech = True
                    region_start = timestamp
                elif not speech and is_speech:
                    is_speech = False
                    region_end = timestamp
                    # merge very short regions are filtered by caller
                    speech_regions.append((region_start, region_end))

            # If file ends during speech
            if is_speech:
                speech_regions.append((region_start, (len(pcm)/2) / sample_rate))

            return tmp_wav, speech_regions
        except Exception as e:
            logging.warning(f"Preprocessing/VAD failed: {e}")
            return audio_file, []

    # Helper: run pyannote pipeline and optionally filter by VAD regions
    def run_pyannote_diarization(in_file, min_speakers=2, max_speakers=6, speech_regions=None):
        try:
            from pyannote.core import Segment

            try:
                diarization = diarization_pipeline({"uri": Path(in_file).stem, "audio": in_file},
                                                   min_speakers=min_speakers, max_speakers=max_speakers)
            except Exception as exc:
                # Fallback: some environments lack torchcodec/ffmpeg; preload audio and pass waveform
                logging.warning(f"Pyannote file-based decode failed ({exc}), attempting in-memory waveform pass")
                try:
                    import librosa
                    import torch

                    y, sr = librosa.load(in_file, sr=16000)
                    waveform = torch.from_numpy(y).unsqueeze(0)
                    diarization = diarization_pipeline({"uri": Path(in_file).stem,
                                                       "audio": {"waveform": waveform, "sample_rate": sr}},
                                                       min_speakers=min_speakers, max_speakers=max_speakers)
                except Exception as exc2:
                    logging.warning(f"Pyannote in-memory decode also failed: {exc2}")
                    raise exc2

            # Convert to dictionary and optionally filter using speech_regions
            speaker_segments = {}
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                # If speech_regions provided, skip segments outside speech
                if speech_regions:
                    overlaps = False
                    for s_start, s_end in speech_regions:
                        if not (turn.end <= s_start or turn.start >= s_end):
                            overlaps = True
                            break
                    if not overlaps:
                        continue

                if speaker not in speaker_segments:
                    speaker_segments[speaker] = []
                speaker_segments[speaker].append((turn.start, turn.end))

            return speaker_segments
        except Exception as e:
            logging.warning(f"Pyannote diarization run failed: {e}")
            raise

    # Helper: embedding + clustering fallback using SpeechBrain ECAPA + HDBSCAN/HAC
    def embed_and_cluster_fallback(in_file, window_size=1.5, step=0.75, min_cluster_size=2):
        try:
            import tempfile, soundfile as sf
            from speechbrain.inference import EncoderClassifier
            import hdbscan
            from sklearn.cluster import AgglomerativeClustering

            enc = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", savedir="./.cache/speechbrain_ecapa")

            y, sr = librosa.load(in_file, sr=16000)
            dur = len(y) / sr
            windows = []
            pos = 0.0
            while pos < dur:
                start = pos
                end = min(dur, pos + window_size)
                s_idx = int(start * sr); e_idx = int(end * sr)
                seg = y[s_idx:e_idx]
                if len(seg) < int(0.2 * sr):
                    pos += step
                    continue
                tmpf = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
                sf.write(tmpf.name, seg, sr)
                emb = enc.encode_file(tmpf.name).squeeze().cpu().numpy()
                windows.append((start, end, emb))
                pos += step

            if not windows:
                return diarize_speakers_offline(in_file)

            X = np.vstack([w[2] for w in windows])

            try:
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
                labels = clusterer.fit_predict(X)
            except Exception:
                clusterer = AgglomerativeClustering(n_clusters=None, distance_threshold=1.0)
                labels = clusterer.fit_predict(X)

            # Map labels to segments
            segments = {}
            for (start, end, _), label in zip(windows, labels):
                if label < 0:
                    continue
                speaker = f"SPEAKER_{label:02d}"
                segments.setdefault(speaker, []).append((start, end))

            # Merge adjacent windows of same label
            from itertools import groupby
            merged = {}
            for spk, segs in segments.items():
                segs_sorted = sorted(segs, key=lambda x: x[0])
                merged_list = []
                cur_s, cur_e = segs_sorted[0]
                for s, e in segs_sorted[1:]:
                    if s - cur_e <= 0.5:
                        cur_e = e
                    else:
                        merged_list.append((cur_s, cur_e))
                        cur_s, cur_e = s, e
                merged_list.append((cur_s, cur_e))
                merged[spk] = merged_list

            return merged
        except Exception as e:
            logging.warning(f"Embedding+clustering fallback failed: {e}")
            return diarize_speakers_offline(in_file)

    # Main diarization flow
    if DIARIZATION_HF_AVAILABLE:
        try:
            logging.info("Running HuggingFace speaker diarization with preprocessing and hybrid fallback")
            # Preprocess with VAD to get speech regions
            try:
                tmp_wav, speech_regions = preprocess_for_diarization(audio_file)
            except Exception:
                tmp_wav, speech_regions = audio_file, []

            try:
                speaker_segments = run_pyannote_diarization(tmp_wav, min_speakers=2, max_speakers=6, speech_regions=speech_regions)
                logging.info(f"Detected {len(speaker_segments)} speakers")
                return speaker_segments
            except Exception:
                logging.info("Pyannote pipeline failed, attempting embedding+clustering fallback")
                return embed_and_cluster_fallback(audio_file)

        except Exception as e:
            logging.warning(f"HF diarization overall failed: {e}, using offline fallback")
            return diarize_speakers_offline(audio_file)
    else:
        return diarize_speakers_offline(audio_file)


def analyze_sentiment(text1, text2):
    """
    Analyze sentiment of two texts.

    Attempts BERT sentiment model first, falls back to neutral scores.

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        dict: Sentiment scores and difference
    """
    if SENTIMENT_HF_AVAILABLE:
        try:
            import torch
            import torch.nn.functional as F

            def get_sentiment_score(text):
                # Suppress tokenizer warning when checking text length
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", message="Token indices sequence length")
                    tokens = sentiment_tokenizer(text, return_tensors="pt", truncation=False)

                if tokens['input_ids'].shape[1] > 512:
                    words = text.split()
                    chunk_size = 500
                    all_scores = []

                    for i in range(0, len(words), chunk_size):
                        chunk_text = ' '.join(words[i:i+chunk_size])
                        with torch.no_grad():
                            inputs = sentiment_tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
                            outputs = sentiment_model(**inputs)
                            probs = F.softmax(outputs.logits, dim=1)
                            chunk_score = torch.argmax(probs, dim=1).item() + 1
                            all_scores.append(chunk_score)

                    return sum(all_scores) / len(all_scores)
                else:
                    with torch.no_grad():
                        inputs = sentiment_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                        outputs = sentiment_model(**inputs)
                        probs = F.softmax(outputs.logits, dim=1)
                        return torch.argmax(probs, dim=1).item() + 1

            score1 = get_sentiment_score(text1)
            score2 = get_sentiment_score(text2)

            return {
                'text1_sentiment': score1,
                'text2_sentiment': score2,
                'sentiment_difference': abs(score1 - score2)
            }

        except Exception as e:
            logging.warning(f"HF sentiment analysis failed: {e}, using offline fallback")
            return analyze_sentiment_offline(text1, text2)
    else:
        return analyze_sentiment_offline(text1, text2)


def calculate_similarity(text1, text2):
    """
    Calculate semantic similarity between two texts.

    Attempts sentence-transformers model first, falls back to Jaccard similarity.

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Similarity score (0.0 to 1.0)
    """
    if SIMILARITY_HF_AVAILABLE:
        try:
            import torch

            def get_embedding(text):
                inputs = similarity_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
                with torch.no_grad():
                    outputs = similarity_model(**inputs)
                embeddings = outputs.last_hidden_state.mean(dim=1)
                return embeddings.detach().numpy()[0]

            emb1 = get_embedding(text1)
            emb2 = get_embedding(text2)
            similarity = 1 - cosine(emb1, emb2)

            return similarity

        except Exception as e:
            logging.warning(f"HF similarity calculation failed: {e}, using offline fallback")
            return calculate_similarity_offline(text1, text2)
    else:
        return calculate_similarity_offline(text1, text2)


def keyword_overlap(text1, text2):
    """
    Calculate keyword overlap between two texts.

    Attempts TextBlob noun phrase extraction (HF-enhanced), falls back to word overlap.

    Args:
        text1 (str): First text
        text2 (str): Second text

    Returns:
        float: Overlap ratio (0.0 to 1.0)
    """
    try:
        blob1 = TextBlob(text1)
        blob2 = TextBlob(text2)

        keywords1 = set(blob1.noun_phrases)
        keywords2 = set(blob2.noun_phrases)

        overlap = keywords1.intersection(keywords2)
        overlap_ratio = len(overlap) / max(len(keywords1), 1)

        return overlap_ratio

    except Exception as e:
        logging.warning(f"Keyword overlap failed: {e}, using offline fallback")
        return keyword_overlap_offline(text1, text2)


# ============================================================================
# AUDIO PROCESSING PIPELINE
# ============================================================================

# Audio and feature caches (for performance optimization)
_audio_cache = {}
_feature_cache = {}

def load_and_preprocess_audio(audio_file, target_sr=16000, max_duration=None):
    """
    Load and preprocess audio file with caching.

    Args:
        audio_file (str): Path to audio file
        target_sr (int): Target sample rate (default: 16000)
        max_duration (float): Maximum duration in seconds (optional)

    Returns:
        tuple: (y, sr) - Audio waveform and sample rate
    """
    cache_key = f"{audio_file}_{target_sr}_{max_duration}"

    if cache_key in _audio_cache:
        return _audio_cache[cache_key]

    logging.info(f"Loading audio: {audio_file}")

    # Load with optimized parameters
    y, sr = librosa.load(audio_file, sr=target_sr, mono=True)

    # Trim silence for better performance
    y, _ = librosa.effects.trim(y, top_db=20)

    # Limit duration if specified
    if max_duration and len(y) > max_duration * sr:
        y = y[:int(max_duration * sr)]

    # Normalize audio
    y = librosa.util.normalize(y)

    duration = len(y) / sr
    logging.info(f"Audio loaded: {duration:.2f} seconds, {sr}Hz")

    # Cache if not too large
    if len(_audio_cache) < MAX_CACHE_SIZE:
        _audio_cache[cache_key] = (y, sr)

    return y, sr


@lru_cache(maxsize=32)
def extract_audio_features_cached(audio_file_path):
    """
    Cached version of audio feature extraction.

    Args:
        audio_file_path (str): Path to audio file

    Returns:
        dict: Extracted audio features
    """
    y, sr = librosa.load(audio_file_path, sr=16000)
    return extract_audio_features_optimized(audio_file_path, y, sr)


def extract_audio_features_optimized(audio_file, y=None, sr=None):
    """
    Extract audio features (pitch, intensity, MFCC, tempo) with optimization.

    Uses caching and processes only first 5 seconds for speed.

    Args:
        audio_file (str): Path to audio file
        y (numpy.ndarray, optional): Pre-loaded audio waveform
        sr (int, optional): Sample rate

    Returns:
        dict: Features dictionary with keys:
            - pitch_mean, pitch_std
            - intensity_mean, intensity_std
            - mfcc_mean, mfcc_std
            - tempo
    """
    if y is None or sr is None:
        y, sr = load_and_preprocess_audio(audio_file, target_sr=16000)

    # Check feature cache
    audio_hash = hash(y.tobytes())
    if audio_hash in _feature_cache:
        return _feature_cache[audio_hash]

    features = {}

    # Use shorter audio segment for speed (first 5 seconds)
    segment_length = min(len(y), 5 * sr)
    y_segment = y[:segment_length]

    # Fast pitch estimation using fundamental frequency
    try:
        f0 = librosa.yin(y_segment, fmin=50, fmax=400, sr=sr, frame_length=1024)
        f0_clean = f0[f0 > 0]
        if len(f0_clean) > 0:
            features["pitch_mean"] = float(np.mean(f0_clean))
            features["pitch_std"] = float(np.std(f0_clean))
        else:
            features["pitch_mean"] = 150.0
            features["pitch_std"] = 20.0
    except Exception as e:
        logging.warning(f"Pitch extraction failed: {e}, using defaults")
        features["pitch_mean"] = 150.0
        features["pitch_std"] = 20.0

    # Fast intensity calculation
    rms = librosa.feature.rms(y=y_segment, frame_length=1024, hop_length=256)[0]
    features["intensity_mean"] = float(np.mean(rms))
    features["intensity_std"] = float(np.std(rms))

    # Skip tempo calculation for speed (use default)
    features["tempo"] = 120.0

    # Reduced MFCCs for speed
    mfccs = librosa.feature.mfcc(y=y_segment, sr=sr, n_mfcc=8, hop_length=256, n_fft=1024)
    features["mfcc_mean"] = float(np.mean(mfccs))
    features["mfcc_std"] = float(np.std(mfccs))

    # Cache features
    if len(_feature_cache) < MAX_CACHE_SIZE:
        _feature_cache[audio_hash] = features

    return features


# ============================================================================
# SPEAKER SEGMENT EXTRACTION
# ============================================================================

def extract_speaker_segments(audio_file, speaker_segments, output_dir):
    """
    Extract and save individual speaker audio segments.

    Creates separate WAV files for each speaker by combining their speaking segments.
    Files are saved in the diarized_audio_clips/ subdirectory.

    Args:
        audio_file (str): Path to original audio file
        speaker_segments (dict): Speaker segments {speaker_id: [(start, end), ...]}
        output_dir (str): Base output directory

    Returns:
        dict: Mapping of speaker_id to speaker audio file path
    """
    # Create diarized audio clips directory
    clips_dir = Path(output_dir) / "diarized_audio_clips"
    clips_dir.mkdir(parents=True, exist_ok=True)

    audio_stem = Path(audio_file).stem
    speaker_audio_files = {}

    try:
        # Load audio with pydub
        audio = AudioSegment.from_wav(audio_file)
        logging.info(f"Extracting speaker segments to {clips_dir}")

        for speaker, segments in speaker_segments.items():
            combined_segment = AudioSegment.empty()

            for start, end in segments:
                start_ms = max(0, int(start * 1000))
                end_ms = min(len(audio), int(end * 1000))

                # Extract segment with small fade to avoid clicks
                segment = audio[start_ms:end_ms]
                if len(segment) > 100:  # Only if segment is long enough
                    segment = segment.fade_in(50).fade_out(50)

                combined_segment += segment

            logging.info(f"{speaker}: {len(segments)} segments, {len(combined_segment)/1000:.2f}s total")

            # Save speaker file
            speaker_file = clips_dir / f"{audio_stem}_speaker_{speaker}.wav"

            try:
                combined_segment.export(speaker_file, format="wav")
                speaker_audio_files[speaker] = str(speaker_file)
                logging.info(f"Created: {speaker_file.name}")
            except Exception as e:
                logging.warning(f"Export failed for {speaker}: {e}")
                # Fallback to librosa method
                speaker_audio_files[speaker] = audio_file

    except Exception as e:
        logging.warning(f"Pydub processing failed: {e}, using librosa fallback")
        # Fallback to librosa
        speaker_audio_files = create_speaker_files_with_librosa(audio_file, speaker_segments, clips_dir)

    return speaker_audio_files


def create_speaker_files_with_librosa(audio_file, speaker_segments, clips_dir):
    """
    Alternative speaker file creation using librosa.

    Fallback method when pydub/FFmpeg is not available.

    Args:
        audio_file (str): Path to original audio file
        speaker_segments (dict): Speaker segments
        clips_dir (Path): Output directory for clips

    Returns:
        dict: Mapping of speaker_id to audio file path
    """
    speaker_audio_files = {}
    audio_stem = Path(audio_file).stem

    # Load audio with librosa
    y, sr = librosa.load(audio_file, sr=None)

    for speaker, segments in speaker_segments.items():
        combined_audio = []

        for start, end in segments:
            start_sample = int(start * sr)
            end_sample = min(len(y), int(end * sr))

            segment = y[start_sample:end_sample]
            if len(segment) > 0:
                combined_audio.extend(segment)

        if combined_audio:
            speaker_file = clips_dir / f"{audio_stem}_speaker_{speaker}.wav"

            try:
                import soundfile as sf
                sf.write(speaker_file, np.array(combined_audio), sr)
                speaker_audio_files[speaker] = str(speaker_file)
                logging.info(f"Created with librosa: {speaker_file.name}")
            except ImportError:
                try:
                    from scipy.io import wavfile
                    # Normalize to 16-bit
                    audio_normalized = np.int16(np.array(combined_audio) * 32767)
                    wavfile.write(speaker_file, sr, audio_normalized)
                    speaker_audio_files[speaker] = str(speaker_file)
                    logging.info(f"Created with scipy: {speaker_file.name}")
                except ImportError:
                    logging.warning(f"No audio export library available for {speaker}")
                    speaker_audio_files[speaker] = audio_file
        else:
            logging.warning(f"No audio segments for {speaker}")
            speaker_audio_files[speaker] = audio_file

    return speaker_audio_files


# ============================================================================
# SPEAKER ANALYSIS
# ============================================================================

def analyze_personality(features):
    """
    Analyze personality traits based on audio features.

    Uses heuristic thresholds on pitch, intensity, and tempo to infer
    Big Five personality traits (extraversion, openness, conscientiousness).

    Args:
        features (dict): Audio features containing pitch_mean, intensity_mean, tempo

    Returns:
        dict: Personality traits with High/Low values
    """
    personality_traits = {}

    # Extraversion based on pitch
    if features["pitch_mean"] > 150:
        personality_traits["extraversion"] = "High"
    else:
        personality_traits["extraversion"] = "Low"

    # Openness based on intensity
    if features["intensity_mean"] > 0.05:
        personality_traits["openness"] = "High"
    else:
        personality_traits["openness"] = "Low"

    # Conscientiousness based on tempo
    if features["tempo"] > 100:
        personality_traits["conscientiousness"] = "High"
    else:
        personality_traits["conscientiousness"] = "Low"

    return personality_traits


def analyze_single_speaker(speaker_data):
    """
    Analyze a single speaker's audio for emotion, features, and personality.

    Uses parallel processing to extract audio features and analyze emotion
    simultaneously for better performance. Includes memory management with
    garbage collection.

    Args:
        speaker_data (tuple): (speaker_id, speaker_file_path)

    Returns:
        tuple: (speaker_id, result_dict) where result_dict contains:
            - speaker_id: Speaker identifier
            - audio_features: Extracted features (pitch, intensity, MFCC, tempo)
            - emotion: Predicted emotion label
            - personality: Personality trait analysis
            - processing_time: Analysis duration in seconds
    """
    speaker, speaker_file = speaker_data
    logging.info(f"[{speaker}] Starting analysis")

    start_time = time.time()

    try:
        # Load audio once with preprocessing
        y, sr = load_and_preprocess_audio(speaker_file, target_sr=16000, max_duration=None)

        # Run analyses in parallel where possible
        with ThreadPoolExecutor(max_workers=2) as executor:
            features_future = executor.submit(extract_audio_features_optimized, speaker_file, y, sr)
            emotion_future = executor.submit(analyze_emotion, y, sr)

            features = features_future.result()
            predicted_emotion = emotion_future.result()

        personality_traits = analyze_personality(features)

        # Clean up audio data
        del y
        gc.collect()

        analysis_time = time.time() - start_time
        logging.info(f"[{speaker}] Completed in {analysis_time:.2f}s - Emotion: {predicted_emotion}")

        result = {
            "speaker_id": speaker,
            "audio_features": features,
            "emotion": predicted_emotion,
            "personality": personality_traits,
            "processing_time": analysis_time
        }

        return speaker, result

    except Exception as e:
        logging.error(f"[{speaker}] Analysis failed: {e}")
        return speaker, {
            "speaker_id": speaker,
            "audio_features": {},
            "emotion": "Unknown",
            "personality": {},
            "processing_time": 0,
            "error": str(e)
        }


# ============================================================================
# COMPREHENSION ANALYSIS
# ============================================================================

def transcribe_audio_optimized(audio_file, model_path=None):
    """
    Transcribe audio using Vosk speech recognition.

    Requires:
        - Vosk library: pip install vosk
        - Vosk model downloaded to VOSK_MODEL_PATH

    Args:
        audio_file (str): Path to audio file
        model_path (str): Not used (for compatibility)

    Returns:
        str: Transcribed text

    Raises:
        RuntimeError: If Vosk model not available
    """
    if not TRANSCRIPTION_VOSK_AVAILABLE:
        error_msg = "Vosk transcription model not available"
        logging.error(error_msg)
        logging.error(f"Vosk model path: {VOSK_MODEL_PATH}")
        logging.error("Download vosk-model-en-us-0.22.zip from:")
        logging.error("  https://alphacephei.com/vosk/models")
        raise RuntimeError(error_msg)

    try:
        from vosk import KaldiRecognizer

        # Load audio at 16kHz (required by Vosk)
        y, sr = librosa.load(audio_file, sr=16000)

        # Convert to 16-bit PCM (Vosk requires this format)
        audio_int16 = (y * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        # Create recognizer
        recognizer = KaldiRecognizer(vosk_model, 16000)
        recognizer.SetWords(True)

        # Process audio in chunks
        chunk_size = 4000
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i+chunk_size]
            recognizer.AcceptWaveform(chunk)

        # Get final result
        result = json.loads(recognizer.FinalResult())
        transcription = result.get('text', '')

        if not transcription:
            logging.warning("Vosk returned empty transcription")
            transcription = "[no speech detected]"

        logging.info(f"Transcribed ({len(transcription)} chars): {transcription[:100]}...")
        return transcription

    except Exception as e:
        logging.error(f"Vosk transcription failed: {e}")
        raise RuntimeError(f"Transcription failed: {e}")


def transcribe_speakers_parallel(audio_file, speaker_segments, output_dir):
    """
    Transcribe all speakers in parallel.

    Args:
        audio_file (str): Path to original audio file
        speaker_segments (dict): Speaker segments
        output_dir (str): Output directory containing speaker clips

    Returns:
        dict: Transcriptions {speaker_id: transcription_text}
    """
    clips_dir = Path(output_dir) / "diarized_audio_clips"
    transcripts_dir = Path(output_dir) / "diarized_audio_transcripts"
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    audio_stem = Path(audio_file).stem

    transcription_tasks = []
    for speaker in speaker_segments.keys():
        speaker_audio = clips_dir / f"{audio_stem}_speaker_{speaker}.wav"
        transcription_tasks.append((speaker, str(speaker_audio)))

    transcriptions = {}
    with ThreadPoolExecutor(max_workers=len(transcription_tasks)) as executor:
        futures = {executor.submit(transcribe_audio_optimized, task[1]): task[0]
                  for task in transcription_tasks}

        for future in futures:
            speaker = futures[future]
            transcription_text = future.result()
            transcriptions[speaker] = transcription_text

            # Save transcript to text file
            transcript_file = transcripts_dir / f"{audio_stem}_speaker_{speaker}.txt"
            try:
                with open(transcript_file, 'w', encoding='utf-8') as f:
                    f.write(transcription_text)
                logging.info(f"Saved transcript: {transcript_file.name}")
            except Exception as e:
                logging.warning(f"Failed to save transcript for {speaker}: {e}")

    return transcriptions


def analyze_turn_taking(speaker_segments):
    """
    Analyze turn-taking patterns between speakers.

    Calculates average response time between speaker turns.

    Args:
        speaker_segments (dict): Speaker segments {speaker_id: [(start, end), ...]}

    Returns:
        float: Average response time in seconds
    """
    # Get first two speakers (handles N speakers by focusing on primary pair)
    speaker_ids = sorted(speaker_segments.keys())
    if len(speaker_ids) < 2:
        logging.warning("Less than 2 speakers, using default response time")
        return 0.5

    speaker_1_times = speaker_segments.get(speaker_ids[0], [])
    speaker_2_times = speaker_segments.get(speaker_ids[1], [])

    if not speaker_1_times or not speaker_2_times:
        logging.warning("One or both speakers have no segments")
        return 0.5

    # Combine and sort all turns
    all_turns = sorted(
        [(start, end, speaker_ids[0]) for start, end in speaker_1_times] +
        [(start, end, speaker_ids[1]) for start, end in speaker_2_times],
        key=lambda x: x[0]
    )

    # Calculate response times
    response_times = []
    for i in range(len(all_turns) - 1):
        _, end_time, speaker = all_turns[i]
        start_next, _, next_speaker = all_turns[i + 1]

        if speaker != next_speaker:
            response_time = start_next - end_time
            response_times.append(response_time)

    avg_response_time = np.mean(response_times) if response_times else 0.5
    logging.info(f"Average response time: {avg_response_time:.2f}s")
    return avg_response_time


def analyze_emotion_alignment_fixed(emotions):
    """
    Analyze emotional alignment between speakers.

    Calculates ratio of matching emotions across all speaker pairs.

    Args:
        emotions (dict): Emotions {speaker_id: emotion_label}

    Returns:
        float: Alignment ratio (0.0 to 1.0)
    """
    if not emotions or len(emotions) < 2:
        return 0.0

    # Convert to list of tuples
    emotion_list = [(speaker, emotion) for speaker, emotion in emotions.items()]

    if len(emotion_list) < 2:
        return 0.0

    alignment_scores = []

    # Compare emotions between all speaker pairs
    for i in range(len(emotion_list)):
        for j in range(i + 1, len(emotion_list)):
            speaker1, emotion1 = emotion_list[i]
            speaker2, emotion2 = emotion_list[j]
            alignment_score = int(emotion1 == emotion2)
            alignment_scores.append(alignment_score)

    alignment_ratio = np.mean(alignment_scores) if alignment_scores else 0.0
    logging.info(f"Emotion alignment ratio: {alignment_ratio:.2f}")
    return alignment_ratio


def detect_backchannels_optimized(audio_file, speaker_segments):
    """
    Detect backchannel utterances (short affirmative sounds).

    Identifies short segments with low pitch that may be backchannels
    (e.g., "uh-huh", "yeah", "mm-hmm").

    Args:
        audio_file (str): Path to audio file
        speaker_segments (dict): Speaker segments

    Returns:
        list: Backchannel events [(speaker_id, start_time, end_time), ...]
    """
    y, sr = librosa.load(audio_file, sr=16000)

    backchannels = []
    for speaker, segments in speaker_segments.items():
        for start, end in segments:
            start_sample, end_sample = int(start * sr), int(end * sr)

            # Skip very short or very long segments
            duration = end - start
            if duration < 0.1 or duration > 2.0:
                continue

            segment = y[start_sample:end_sample]
            if len(segment) == 0:
                continue

            # Use simple pitch detection
            try:
                pitches = librosa.yin(segment, fmin=50, fmax=300, sr=sr)
                avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            except:
                avg_pitch = 150  # Default

            if avg_pitch < 150 and duration < 1.0:
                backchannels.append((speaker, start, end))

    logging.info(f"Detected {len(backchannels)} backchannels")
    return backchannels


def analyze_comprehension_parallel(audio_file, speaker_segments, emotions, output_dir):
    """
    Comprehensive comprehension analysis using parallel processing.

    Analyzes turn-taking, emotion alignment, transcription, keyword overlap,
    semantic similarity, sentiment, and backchannels.

    Args:
        audio_file (str): Path to audio file
        speaker_segments (dict): Speaker segments
        emotions (dict): Emotions {speaker_id: emotion_label}
        output_dir (str): Output directory

    Returns:
        dict: Comprehension metrics including scores for all speakers
    """
    # Run independent analyses in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        turn_taking_future = executor.submit(analyze_turn_taking, speaker_segments)
        emotion_alignment_future = executor.submit(analyze_emotion_alignment_fixed, emotions)
        backchannels_future = executor.submit(detect_backchannels_optimized, audio_file, speaker_segments)
        transcription_future = executor.submit(transcribe_speakers_parallel, audio_file, speaker_segments, output_dir)

        # Collect results
        avg_response_time = turn_taking_future.result()
        alignment_ratio = emotion_alignment_future.result()
        backchannels = backchannels_future.result()
        transcriptions = transcription_future.result()

    # Count backchannels (handle N speakers dynamically)
    backchannel_count = len(backchannels)

    # Get first two speakers for pairwise comparisons
    speaker_ids = sorted(transcriptions.keys())

    # Calculate keyword overlap
    overlap_ratio = 0.0
    if len(speaker_ids) >= 2:
        try:
            overlap_ratio = keyword_overlap(transcriptions[speaker_ids[0]], transcriptions[speaker_ids[1]])
            logging.info(f"Keyword overlap: {overlap_ratio:.3f}")
        except Exception as e:
            logging.warning(f"Keyword overlap failed: {e}, using default")
            overlap_ratio = 0.0

    # Calculate similarity score
    similarity_score = 0.0
    if len(speaker_ids) >= 2:
        try:
            similarity_score = calculate_similarity(transcriptions[speaker_ids[0]], transcriptions[speaker_ids[1]])
            logging.info(f"Similarity score: {similarity_score:.3f}")
        except Exception as e:
            logging.warning(f"Similarity calculation failed: {e}, using default")
            similarity_score = 0.0

    # Calculate sentiment scores for all speakers
    sentiment_metrics = {}
    for speaker_id in speaker_ids:
        try:
            # For single speaker, compare with empty text to get individual sentiment
            result = analyze_sentiment(transcriptions[speaker_id], "")
            sentiment_metrics[f'{speaker_id}_sentiment'] = result['text1_sentiment']
        except Exception as e:
            logging.warning(f"Sentiment analysis failed for {speaker_id}: {e}")
            sentiment_metrics[f'{speaker_id}_sentiment'] = 3.0  # Neutral

    # Calculate overall comprehension score
    comprehension_score = (
        (alignment_ratio or 0) * 0.3 +
        (1 / (1 + (avg_response_time or 1))) * 0.3 +
        (overlap_ratio or 0) * 0.3 +
        (backchannel_count > 5) * 0.1
    )

    # Build comprehensive summary
    comprehension_summary = {
        "average_response_time": avg_response_time,
        "emotion_alignment_ratio": alignment_ratio,
        "keyword_overlap_ratio": overlap_ratio,
        "backchannel_count": backchannel_count,
        "overall_comprehension_score": comprehension_score,
        "overall_similarity_score": similarity_score
    }

    # Add all speaker sentiments
    comprehension_summary.update(sentiment_metrics)

    logging.info(f"Comprehension score: {comprehension_score:.3f}")
    return comprehension_summary


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_audio_file(audio_file, output_dir, file_index=1, total_files=1):
    """
    Process a single audio file through the complete pipeline.

    Orchestrates the full analysis: diarization, speaker extraction, speaker analysis,
    and comprehension analysis.

    Args:
        audio_file (str): Path to audio file
        output_dir (str): Output directory path
        file_index (int): Current file number (for logging)
        total_files (int): Total number of files being processed

    Returns:
        dict: Results dictionary with:
            - audio_file: Filename
            - num_speakers: Number of speakers detected
            - duration: Audio duration in seconds
            - speaker_analysis: Dict of speaker results
            - comprehension: Comprehension metrics
            - processing_time: Total processing time
    """
    logging.info(f"Processing {audio_file} ({file_index}/{total_files})")
    start_time = time.time()

    try:
        # 1. Speaker diarization
        logging.info("Step 1/4: Speaker diarization")
        speaker_segments = diarize_speakers(audio_file)
        logging.info(f"Detected {len(speaker_segments)} speakers")

        # 2. Extract speaker audio clips
        logging.info("Step 2/4: Extracting speaker segments")
        clips_dir = os.path.join(output_dir, "diarized_audio_clips")
        os.makedirs(clips_dir, exist_ok=True)
        speaker_files_dict = extract_speaker_segments(audio_file, speaker_segments, output_dir)

        # 3. Analyze each speaker in parallel
        logging.info("Step 3/4: Analyzing speakers")
        speaker_files = [(speaker, speaker_files_dict[speaker])
                        for speaker in speaker_segments.keys()]

        with ThreadPoolExecutor(max_workers=min(len(speaker_files), 4)) as executor:
            speaker_results = list(executor.map(analyze_single_speaker, speaker_files))

        speaker_analysis = {speaker: result for speaker, result in speaker_results}

        # 4. Comprehension analysis
        logging.info("Step 4/4: Comprehension analysis")
        emotions = {speaker: data['emotion'] for speaker, data in speaker_analysis.items()}
        comprehension = analyze_comprehension_parallel(audio_file, speaker_segments, emotions, output_dir)

        # 5. Calculate duration
        audio = AudioSegment.from_wav(audio_file)
        duration = len(audio) / 1000.0

        processing_time = time.time() - start_time
        logging.info(f"Processing completed in {processing_time:.2f}s")

        return {
            'audio_file': Path(audio_file).name,
            'num_speakers': len(speaker_segments),
            'duration': duration,
            'speaker_analysis': speaker_analysis,
            'comprehension': comprehension,
            'processing_time': processing_time
        }

    except Exception as e:
        logging.error(f"Failed to process {audio_file}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        raise


def process_batch_audio_files(audio_files, output_dir, batch_mode=False):
    """
    Process multiple audio files.

    Args:
        audio_files (list): List of audio file paths
        output_dir (str): Output directory
        batch_mode (bool): Use process-level parallelization for multiple files

    Returns:
        list: List of result dictionaries
    """
    total_files = len(audio_files)
    results = []

    logging.info(f"Processing {total_files} audio file(s)")

    if batch_mode and total_files > 1:
        logging.info(f"Batch mode: processing {total_files} files in parallel")
        with ProcessPoolExecutor(max_workers=min(mp.cpu_count(), total_files)) as executor:
            futures = [executor.submit(process_single_audio_file, f, output_dir, i+1, total_files)
                      for i, f in enumerate(audio_files)]
            results = [future.result() for future in futures]
    else:
        for i, audio_file in enumerate(audio_files):
            result = process_single_audio_file(audio_file, output_dir, i+1, total_files)
            results.append(result)

    logging.info(f"All {total_files} file(s) processed successfully")
    return results


# ============================================================================
# OUTPUT CONSOLIDATION
# ============================================================================

def consolidate_speaker_results(all_results):
    """
    Create consolidated speaker-level CSV data.

    Extracts individual speaker metrics from all processed files and
    creates a flat table with one row per speaker per audio file.

    Args:
        all_results (list): List of result dicts from process_single_audio_file

    Returns:
        pandas.DataFrame: Speaker-level data with columns:
            Audio_File, Speaker_ID, Emotion, Pitch_Mean, Pitch_Std,
            Intensity_Mean, Intensity_Std, MFCC_Mean, MFCC_Std, Tempo,
            Extraversion, Openness, Conscientiousness, Processing_Time_Sec
    """
    rows = []

    for result in all_results:
        audio_file = result['audio_file']

        for speaker_id, speaker_data in result['speaker_analysis'].items():
            features = speaker_data.get('audio_features', {})
            personality = speaker_data.get('personality', {})

            row = {
                'Audio_File': audio_file,
                'Speaker_ID': speaker_id,
                'Emotion': speaker_data.get('emotion', 'Unknown'),
                'Pitch_Mean': features.get('pitch_mean', 0.0),
                'Pitch_Std': features.get('pitch_std', 0.0),
                'Intensity_Mean': features.get('intensity_mean', 0.0),
                'Intensity_Std': features.get('intensity_std', 0.0),
                'MFCC_Mean': features.get('mfcc_mean', 0.0),
                'MFCC_Std': features.get('mfcc_std', 0.0),
                'Tempo': features.get('tempo', 0.0),
                'Extraversion': personality.get('extraversion', 'Unknown'),
                'Openness': personality.get('openness', 'Unknown'),
                'Conscientiousness': personality.get('conscientiousness', 'Unknown'),
                'Processing_Time_Sec': speaker_data.get('processing_time', 0.0)
            }
            rows.append(row)

    return pd.DataFrame(rows)


def consolidate_conversation_results(all_results):
    """
    Create consolidated conversation-level CSV data.

    Extracts conversation-level metrics from all processed files and
    creates a flat table with one row per audio file. Handles N speakers
    dynamically by storing individual sentiments as JSON array.

    Args:
        all_results (list): List of result dicts from process_single_audio_file

    Returns:
        pandas.DataFrame: Conversation-level data with columns:
            Audio_File, Num_Speakers, Duration_Sec, Overall_Similarity_Score,
            Overall_Comprehension_Score, Speaker_Sentiments_JSON,
            Avg_Response_Time, Emotion_Alignment_Ratio, Keyword_Overlap_Ratio,
            Backchannel_Count, Total_Processing_Time_Sec
    """
    rows = []

    for result in all_results:
        comp = result['comprehension']

        # Handle N speakers - create JSON array of sentiments
        speaker_sentiments = []
        for speaker_id in sorted(result['speaker_analysis'].keys()):
            # Extract from comprehension sentiment data
            sentiment_key = f'{speaker_id}_sentiment'
            if sentiment_key in comp:
                speaker_sentiments.append({
                    'speaker_id': speaker_id,
                    'sentiment': comp[sentiment_key]
                })

        row = {
            'Audio_File': result['audio_file'],
            'Num_Speakers': result['num_speakers'],
            'Duration_Sec': result['duration'],
            'Overall_Similarity_Score': comp.get('overall_similarity_score', 0.0),
            'Overall_Comprehension_Score': comp.get('overall_comprehension_score', 0.0),
            'Speaker_Sentiments_JSON': json.dumps(speaker_sentiments),
            'Avg_Response_Time': comp.get('average_response_time', 0.0),
            'Emotion_Alignment_Ratio': comp.get('emotion_alignment_ratio', 0.0),
            'Keyword_Overlap_Ratio': comp.get('keyword_overlap_ratio', 0.0),
            'Backchannel_Count': comp.get('backchannel_count', 0),
            'Total_Processing_Time_Sec': result['processing_time']
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_consolidated_results(all_results, output_dir):
    """
    Save all consolidated results to CSV files.

    Creates two CSV files:
    1. consolidated_results.csv - Speaker-level metrics
    2. consolidated_conversation_summary.csv - Conversation-level metrics

    Args:
        all_results (list): Results from all processed files
        output_dir (str): Output directory path
    """
    # Create speaker-level CSV
    speaker_df = consolidate_speaker_results(all_results)
    speaker_csv_path = os.path.join(output_dir, "consolidated_results.csv")
    speaker_df.to_csv(speaker_csv_path, index=False)
    logging.info(f"Saved speaker-level results: {speaker_csv_path}")
    logging.info(f"  {len(speaker_df)} speaker records")

    # Create conversation-level CSV
    conversation_df = consolidate_conversation_results(all_results)
    conversation_csv_path = os.path.join(output_dir, "consolidated_conversation_summary.csv")
    conversation_df.to_csv(conversation_csv_path, index=False)
    logging.info(f"Saved conversation-level results: {conversation_csv_path}")
    logging.info(f"  {len(conversation_df)} conversation records")


# ============================================================================
# BOOTSTRAP ANALYSIS
# ============================================================================

def run_bootstrap_analysis(audio_file, num_iterations, output_dir):
    """
    Run bootstrap iterations on a single audio file.

    Processes the same audio file multiple times to generate statistical
    confidence intervals and variability estimates.

    Args:
        audio_file (str): Path to audio file
        num_iterations (int): Number of iterations to run
        output_dir (str): Output directory

    Returns:
        list: List of results from all iterations
    """
    logging.info(f"Running bootstrap analysis: {num_iterations} iterations on {audio_file}")

    all_results = []
    for i in range(1, num_iterations + 1):
        logging.info(f"Bootstrap iteration {i}/{num_iterations}")
        result = process_single_audio_file(audio_file, output_dir, i, num_iterations)

        # Add iteration number to result
        result['iteration'] = i
        all_results.append(result)

        # Memory cleanup between iterations
        gc.collect()

    logging.info(f"Bootstrap analysis complete: {num_iterations} iterations")
    return all_results


# ============================================================================
# CLI AND MAIN ENTRY POINT
# ============================================================================

def setup_argument_parser():
    """
    Set up command-line argument parser.

    Returns:
        argparse.ArgumentParser: Configured parser
    """
    parser = argparse.ArgumentParser(
        description="Paralinguistics Analysis - Comprehensive audio analysis tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python paralinguistics_analysis.py audio.wav
    python paralinguistics_analysis.py audio1.wav audio2.wav audio3.wav
    python paralinguistics_analysis.py folder/*.wav --batch-mode
    python paralinguistics_analysis.py audio.wav --bootstrap 100
        """
    )

    parser.add_argument('audiofiles', nargs='+', help='Audio file(s) to analyze (.wav format)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Custom output directory (default: analysis_results_<timestamp>)')
    parser.add_argument('--batch-mode', action='store_true',
                       help='Enable batch processing optimizations for multiple files')
    parser.add_argument('--bootstrap', type=int, metavar='N',
                       help='Run N bootstrap iterations (for statistical analysis)')
    parser.add_argument('--models-cache-dir', type=str, default=None,
                       help='Custom models cache directory (for both HuggingFace and Vosk models)')
    parser.add_argument('--delete-temp-files', action='store_true',
                       help='Delete intermediate speaker audio files after processing')

    return parser


def main():
    """
    Main entry point for the paralinguistics analysis script.
    """
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Set up output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"analysis_results_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Set up logging
    log_file = os.path.join(output_dir, "paralinguistic_analysis.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logging.info("="*70)
    logging.info("Paralinguistics Analysis Started")
    logging.info("="*70)
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Audio files: {len(args.audiofiles)}")

    # Set custom cache directory if specified
    if args.models_cache_dir:
        os.environ['TRANSFORMERS_CACHE'] = args.models_cache_dir
        os.environ['HF_HOME'] = args.models_cache_dir
        logging.info(f"Using custom cache directory: {args.models_cache_dir}")

    # Load HuggingFace models
    load_huggingface_models()

    # Load Vosk transcription model (separate from HuggingFace)
    load_vosk_model(custom_cache_dir=args.models_cache_dir)

    # Verify Vosk is available before proceeding
    if not TRANSCRIPTION_VOSK_AVAILABLE:
        logging.error("VOSK MODEL NOT AVAILABLE")
        logging.error("Transcription is required for paralinguistics analysis.")
        logging.error("")
        logging.error("Vosk library installed: Check with 'pip show vosk'")
        logging.error(f"Vosk model location: {VOSK_MODEL_PATH}")
        logging.error("")
        logging.error("To download the model:")
        logging.error("  1. Download: https://alphacephei.com/vosk/models/vosk-model-en-us-0.22.zip")
        logging.error("  2. Extract to: ~/.cache/vosk/models/")
        sys.exit(1)

    # Log all models status
    models_loaded = sum([EMOTION_HF_AVAILABLE, DIARIZATION_HF_AVAILABLE,
                        SENTIMENT_HF_AVAILABLE, SIMILARITY_HF_AVAILABLE])
    logging.info(f"All models loaded: {models_loaded}/4 HuggingFace + Vosk transcription")
    
    # Process files
    try:
        if args.bootstrap:
            # Bootstrap mode - single file only
            if len(args.audiofiles) > 1:
                logging.error("Bootstrap mode requires exactly one audio file")
                sys.exit(1)

            results = run_bootstrap_analysis(args.audiofiles[0], args.bootstrap, output_dir)
        else:
            # Normal processing mode
            results = process_batch_audio_files(args.audiofiles, output_dir, args.batch_mode)

        # Save consolidated results
        save_consolidated_results(results, output_dir)

        # Clean up temporary files if requested
        if args.delete_temp_files:
            clips_dir = os.path.join(output_dir, "diarized_audio_clips")
            if os.path.exists(clips_dir):
                try:
                    import shutil
                    import stat
                    import gc
                    import time

                    # Force garbage collection to release file handles
                    gc.collect()
                    time.sleep(0.5)

                    def handle_remove_readonly(func, path, exc):
                        """Error handler for Windows readonly files"""
                        try:
                            os.chmod(path, stat.S_IWUSR)
                            func(path)
                        except Exception:
                            pass  # Ignore if chmod fails

                    shutil.rmtree(clips_dir, onerror=handle_remove_readonly)
                    logging.info("Cleaned up temporary audio clips")

                except PermissionError as e:
                    logging.warning(f"Could not remove temporary files: {e}")
                    logging.warning(f"Please manually delete: {clips_dir}")
                except Exception as e:
                    logging.warning(f"Cleanup failed: {e}")
                    logging.warning(f"Temporary files preserved at: {clips_dir}")
        else:
            clips_dir = os.path.join(output_dir, "diarized_audio_clips")
            logging.info(f"Speaker audio clips preserved at: {clips_dir}")

        logging.info("="*70)
        logging.info("Analysis completed successfully")
        logging.info(f"Results saved to: {output_dir}")
        logging.info("="*70)

    except Exception as e:
        logging.error(f"Analysis failed: {e}")
        import traceback
        logging.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
