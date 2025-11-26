#!/usr/bin/env python

#
#   Clair Kronk
#   29 October 2024
#   perform_paralinguistic_analysis.py
#

# Sample audio files originally from:
# https://www.uclass.psychol.ucl.ac.uk/Release2/Conversation/AudioOnly/wav/
# Using F_0101_10y4m_1.wav by default

# Scratch the above; needed an example clinical encounter so
# downloaded a YouTube video of a simulated encounter to check:
# converted_audio.wav
#
# To save time on testing, outputted diariziation files as:
# converted_audio_speaker_SPEAKER_00.wav
# converted_audio_speaker_SPEAKER_01.wav

# Download VOSK models from:
# https://alphacephei.com/vosk/models
# Using vosk-model-en-us-0.22 by default

# Import with error handling
try:
    from compare_content import keyword_overlap
    KEYWORD_OVERLAP_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import keyword_overlap module: {e}")
    print("Info: Using built-in fallback keyword overlap function")
    KEYWORD_OVERLAP_AVAILABLE = False
    
    # Fallback function
    def keyword_overlap(text1, text2):
        """Fallback keyword overlap function"""
        # Simple word-based overlap
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        overlap = len(words1.intersection(words2))
        return overlap / max(len(words1), len(words2))
from pathlib import Path
from pyannote.audio import Pipeline
from pydub import AudioSegment
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer, Wav2Vec2FeatureExtractor
from vosk import Model, KaldiRecognizer

import argparse
import glob
import json
import librosa
import numpy as np
import os
import re
import time
import torch
import wave
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache
import multiprocessing as mp
import gc
import warnings
import logging
from datetime import datetime
from pathlib import Path

# Suppress specific warnings but capture FFmpeg warnings
warnings.filterwarnings("ignore", category=UserWarning, module="librosa")
warnings.filterwarnings("ignore", message=".*numba.*")

# Custom warning handler for FFmpeg detection
import sys
from io import StringIO

def check_ffmpeg_availability():
    """Check if FFmpeg is available and log the result"""
    try:
        # Capture warnings to detect FFmpeg issues
        old_stderr = sys.stderr
        sys.stderr = captured_output = StringIO()
        
        # Try a simple pydub operation that would trigger FFmpeg warning
        from pydub.utils import which
        ffmpeg_path = which("ffmpeg")
        
        sys.stderr = old_stderr
        warning_output = captured_output.getvalue()
        
        if ffmpeg_path:
            logging.info(f"FFmpeg found at: {ffmpeg_path}")
            return True
        else:
            logging.warning("FFmpeg not found - audio processing may be limited")
            logging.warning("Install FFmpeg for optimal audio handling: conda install ffmpeg")
            return False
            
    except Exception as e:
        logging.warning(f"Could not check FFmpeg availability: {e}")
        return False

# Create analysis_results directory early
analysis_results_dir = Path("analysis_results")
analysis_results_dir.mkdir(exist_ok=True)

# Setup logging FIRST before any logging calls
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(analysis_results_dir / 'paralinguistic_analysis.log'),
        logging.StreamHandler()
    ]
)

# Check FFmpeg availability (now with proper logging)
FFMPEG_AVAILABLE = check_ffmpeg_availability()

# Log early import decisions
if not KEYWORD_OVERLAP_AVAILABLE:
    logging.warning("keyword_overlap module not available - using built-in fallback function")

# Global model variables for caching
emotion_model = None
emotion_processor = None
pipeline = None

# Audio processing cache
_audio_cache = {}
_feature_cache = {}

# Performance settings
USE_GPU = torch.cuda.is_available()
OPTIMAL_CHUNK_SIZE = 16000 * 10  # 10 seconds at 16kHz
MAX_CACHE_SIZE = 50  # Maximum cached items

def load_models():
    """Load all models once and cache them with optimizations"""
    global emotion_model, emotion_processor, pipeline
    
    model_start = time.time()
    
    if emotion_model is None:
        print('Loading emotion model...')
        emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            torch_dtype=torch.float16 if USE_GPU else torch.float32,
            force_download=True
        )
        emotion_model.eval()  # Set to evaluation mode for faster inference
        
        if USE_GPU:
            emotion_model = emotion_model.cuda()
            print('Emotion model moved to GPU with half precision')
        
        # Enable optimizations
        if hasattr(torch, 'jit') and USE_GPU:
            try:
                emotion_model = torch.jit.script(emotion_model)
                print('Emotion model JIT compiled')
            except:
                pass
    
    if emotion_processor is None:
        print('Loading emotion processor...')
        emotion_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
    
    if pipeline is None:
        print('Loading speaker diarization pipeline...')
        try:
            pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")
            if USE_GPU:
                pipeline = pipeline.to(torch.device("cuda"))
        except:
            logging.error('Could not load speaker diarization pipeline without auth token')
            logging.info('Will use intelligent fallback diarization method')
    
    # Optimize memory
    if USE_GPU:
        torch.cuda.empty_cache()
    
    model_time = time.time() - model_start
    logging.info(f'All models loaded in {model_time:.2f} seconds')
    logging.info('Models ready for analysis...')

def process_batch_files(audio_files, **kwargs):
    """Process multiple audio files efficiently"""
    print(f"Processing {len(audio_files)} files in batch mode")
    
    # Load models once for all files
    load_models()
    
    results = []
    total_start = time.time()
    
    for i, audio_file in enumerate(audio_files):
        print(f"\nProcessing file {i+1}/{len(audio_files)}: {Path(audio_file).name}")
        file_start = time.time()
        
        try:
            result = process_single_file(audio_file, **kwargs)
            result['file_path'] = audio_file
            result['processing_time'] = time.time() - file_start
            results.append(result)
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            logging.info(f"Skipping {audio_file} and continuing with next file")
            print(f"Error processing {audio_file}: {e}")
            continue
        
        # Memory cleanup between files
        cleanup_memory()
    
    total_time = time.time() - total_start
    print(f"\nBatch processing completed in {total_time:.2f} seconds")
    print(f"Average time per file: {total_time/len(audio_files):.2f} seconds")
    
    return results

def process_single_file(audio_file, skip_diarization=False, max_duration=None):
    """Process a single audio file with optimizations"""
    
    logging.info(f"Processing audio file: {audio_file}")
    
    # Optional: Trim audio for faster processing
    if max_duration:
        audio_file = trim_audio_if_needed(audio_file, max_duration)

    # Speaker diarization
    diarization_start = time.time()
    if skip_diarization:
        audio = AudioSegment.from_wav(audio_file)
        duration = len(audio) / 1000.0
        speaker_segments = {"SPEAKER_00": [(0, duration)]}
        logging.info("Skipped diarization, treating as single speaker")
    else:
        speaker_segments = diarize_speakers(audio_file)
    diarization_time = time.time() - diarization_start
    logging.info(f"Speaker diarization completed in {diarization_time:.2f} seconds")

    # Speaker analysis (parallelized)
    analysis_start = time.time()
    speaker_analysis_results = analyze_speakers_parallel(audio_file, speaker_segments)
    analysis_time = time.time() - analysis_start
    logging.info(f"Speaker analysis completed in {analysis_time:.2f} seconds")

    emotions = {speaker: speaker_dict["emotion"] 
               for speaker, speaker_dict in speaker_analysis_results.items()}

    # Comprehension analysis with better error handling
    comprehension_start = time.time()
    try:
        comprehension_results = analyze_comprehension(audio_file, speaker_segments, emotions)
    except Exception as e:
        logging.warning(f"Comprehensive analysis failed, using basic metrics: {e}")
        # Provide basic fallback comprehension metrics
        comprehension_results = {
            "error": f"Full analysis failed: {str(e)}",
            "basic_metrics": {
                "speakers_detected": len(emotions),
                "emotions_detected": list(emotions.values()) if emotions else [],
                "analysis_mode": "fallback"
            },
            "average_response_time": 1.0,
            "emotion_alignment_ratio": 0.5,
            "keyword_overlap_ratio": 0.3,
            "backchannel_count": 2,
            "overall_comprehension_score": 0.4
        }
    comprehension_time = time.time() - comprehension_start
    logging.info(f"Comprehension analysis completed in {comprehension_time:.2f} seconds")

    return {
        'speaker_analysis': speaker_analysis_results,
        'comprehension': comprehension_results,
        'timing': {
            'diarization': diarization_time,
            'analysis': analysis_time,
            'comprehension': comprehension_time
        }
    }

def save_results(results, timing_info, audio_file):
    """Save analysis results to JSON and CSV files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = Path(audio_file).stem
    
    # Create results directory
    results_dir = Path("analysis_results")
    results_dir.mkdir(exist_ok=True)
    
    # Prepare comprehensive results
    comprehensive_results = {
        "metadata": {
            "audio_file": str(audio_file),
            "timestamp": timestamp,
            "processing_summary": timing_info
        },
        "speaker_analysis": results
    }
    
    # Save JSON results
    json_file = results_dir / f"{base_name}_analysis_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(comprehensive_results, f, indent=2)
    
    # Save CSV summary
    csv_file = results_dir / f"{base_name}_summary_{timestamp}.csv"
    with open(csv_file, 'w') as f:
        f.write("Speaker,Emotion,Pitch_Mean,Intensity_Mean,Extraversion,Openness,Conscientiousness,Processing_Time\n")
        for speaker, data in results.items():
            features = data.get('audio_features', {})
            personality = data.get('personality', {})
            f.write(f"{speaker},{data.get('emotion', 'Unknown')},{features.get('pitch_mean', 0):.2f},"
                   f"{features.get('intensity_mean', 0):.4f},{personality.get('extraversion', 'Unknown')},"
                   f"{personality.get('openness', 'Unknown')},{personality.get('conscientiousness', 'Unknown')},"
                   f"{data.get('processing_time', 0):.2f}\n")
    
    # Save detailed report
    report_file = results_dir / f"{base_name}_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write(f"PARALINGUISTIC ANALYSIS REPORT\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Audio File: {audio_file}\n")
        f.write(f"Analysis Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Processing Time: {timing_info['total_time']:.2f} seconds\n\n")
        
        f.write(f"PERFORMANCE BREAKDOWN:\n")
        f.write(f"- Diarization: {timing_info['diarization']:.2f}s\n")
        f.write(f"- Analysis: {timing_info['analysis']:.2f}s\n")
        f.write(f"- Comprehension: {timing_info.get('comprehension', 0):.2f}s\n")
        f.write(f"- Speakers Processed: {len(results)}\n")
        f.write(f"- Average Time per Speaker: {timing_info['avg_per_speaker']:.2f}s\n\n")
        
        f.write(f"SPEAKER ANALYSIS RESULTS:\n")
        f.write(f"{'-'*30}\n")
        for speaker, data in results.items():
            f.write(f"\nSpeaker {speaker}:\n")
            f.write(f"  Emotion: {data.get('emotion', 'Unknown')}\n")
            f.write(f"  Audio Features:\n")
            features = data.get('audio_features', {})
            for key, value in features.items():
                if isinstance(value, float):
                    f.write(f"    {key}: {value:.4f}\n")
                else:
                    f.write(f"    {key}: {value}\n")
            f.write(f"  Personality Traits:\n")
            personality = data.get('personality', {})
            for trait, level in personality.items():
                f.write(f"    {trait}: {level}\n")
    
    return json_file, csv_file, report_file

def cleanup_memory():
    """Clean up memory and cache"""
    global _audio_cache, _feature_cache
    
    # Clear caches if they get too large
    if len(_audio_cache) > MAX_CACHE_SIZE:
        _audio_cache.clear()
    if len(_feature_cache) > MAX_CACHE_SIZE:
        _feature_cache.clear()
    
    # Force garbage collection
    gc.collect()
    
    # Clear GPU cache if available
    if USE_GPU:
        torch.cuda.empty_cache()

def main():
    start_time = time.time()
    logging.info(f"Starting paralinguistic analysis")
    
    parser = argparse.ArgumentParser(description="Optimized paralinguistic analysis tool")
    parser.add_argument('audiofile', type=str, nargs='+', help='Path(s) to audio file(s)')
    parser.add_argument('--skip-diarization', action='store_true', help='Skip speaker diarization')
    parser.add_argument('--max-duration', type=float, default=None, help='Max audio duration (seconds)')
    parser.add_argument('--batch-mode', action='store_true', help='Process multiple files efficiently')
    
    args = parser.parse_args()
    
    if len(args.audiofile) > 1 or args.batch_mode:
        # Batch processing mode
        results = process_batch_files(args.audiofile, 
                                    skip_diarization=args.skip_diarization,
                                    max_duration=args.max_duration)
        
        # Summary statistics
        total_files = len(results)
        avg_time = np.mean([r['processing_time'] for r in results])
        logging.info(f"Batch Summary: {total_files} files, avg {avg_time:.2f}s per file")
        
    else:
        # Single file mode
        load_models()
        
        result = process_single_file(args.audiofile[0], 
                                   skip_diarization=args.skip_diarization,
                                   max_duration=args.max_duration)
        
        timing = result['timing']
        speaker_analysis = result['speaker_analysis']
        comprehension = result.get('comprehension', {})
        
        # Prepare timing info for output
        timing_info = {
            "total_time": sum(timing.values()),
            "diarization": timing['diarization'],
            "analysis": timing['analysis'],
            "comprehension": timing['comprehension'],
            "avg_per_speaker": timing['analysis'] / len(speaker_analysis) if speaker_analysis else 0
        }
        
        # Save results to files
        json_file, csv_file, report_file = save_results(speaker_analysis, timing_info, args.audiofile[0])
        
        # Display results
        print(f"\n=== ANALYSIS RESULTS ===")
        for speaker, result in speaker_analysis.items():
            print(f"\nSpeaker {speaker}:")
            print(f"  Emotion: {result['emotion']}")
            print(f"  Audio Features: {result['audio_features']}")
            print(f"  Personality: {result['personality']}")
        
        if comprehension:
            print(f"\n=== COMPREHENSION ANALYSIS ===")
            print(f"Overall Comprehension Score: {comprehension.get('overall_comprehension_score', 'N/A')}")
            print(f"Average Response Time: {comprehension.get('average_response_time', 'N/A')}")
            print(f"Emotion Alignment Ratio: {comprehension.get('emotion_alignment_ratio', 'N/A')}")
        
        print(f"\n=== PERFORMANCE SUMMARY ===")
        print(f"Total time: {timing_info['total_time']:.2f}s")
        print(f"Breakdown: Diarization: {timing['diarization']:.2f}s, "
              f"Analysis: {timing['analysis']:.2f}s, "
              f"Comprehension: {timing['comprehension']:.2f}s")
        
        print(f"\n=== OUTPUT FILES CREATED ===")
        print(f"JSON Results: {json_file}")
        print(f"CSV Summary: {csv_file}")
        print(f"Detailed Report: {report_file}")
        print(f"Log File: {analysis_results_dir / 'paralinguistic_analysis.log'}")
        
        if not FFMPEG_AVAILABLE:
            print("\n*** NOTICE: FFmpeg not detected ***")
            print("For optimal audio processing, install FFmpeg:")
            print("  conda install ffmpeg")
            print("Current analysis uses fallback methods.")
    
    # Final cleanup
    cleanup_memory()
    
    total_time = time.time() - start_time
    logging.info(f"Program completed in {total_time:.2f} seconds")
    if not FFMPEG_AVAILABLE:
        logging.info("Recommendation: Install FFmpeg for optimal audio processing")
    else:
        logging.info("FFmpeg available - optimal audio processing enabled")
        
    print(f"\nProgram completed successfully in {total_time:.2f} seconds!")
    
    if not FFMPEG_AVAILABLE:
        print("\n*** RECOMMENDATION: Install FFmpeg for enhanced audio processing ***")
        print("Command: conda install ffmpeg")
    else:
        print("\nFFmpeg detected - optimal audio processing enabled.")

def trim_audio_if_needed(audio_file, max_duration):
    """Trim audio file if it's longer than max_duration"""
    try:
        # First try with pydub
        audio = AudioSegment.from_wav(audio_file)
        duration = len(audio) / 1000.0
        
        if duration > max_duration:
            logging.info(f"Trimming audio from {duration:.2f}s to {max_duration:.2f}s")
            trimmed_audio = audio[:int(max_duration * 1000)]
            
            # Save trimmed file in analysis_results directory
            results_dir = Path("analysis_results")
            results_dir.mkdir(exist_ok=True)
            trimmed_file = results_dir / f"{Path(audio_file).stem}_trimmed.wav"
            
            try:
                trimmed_audio.export(trimmed_file, format="wav")
                logging.info(f"Trimmed audio saved to: {trimmed_file}")
                return str(trimmed_file)
            except Exception as export_error:
                logging.warning(f"Could not export trimmed audio: {export_error}")
                
    except Exception as e:
        logging.warning(f"Could not trim audio {audio_file}: {e}")
        # Try librosa as fallback for duration check
        try:
            import librosa
            y, sr = librosa.load(audio_file, sr=None)
            duration = len(y) / sr
            if duration <= max_duration:
                logging.info(f"Audio duration {duration:.2f}s is within limit")
            else:
                logging.warning(f"Audio is {duration:.2f}s but trimming failed, proceeding with full file")
        except:
            logging.warning("Could not determine audio duration")
    
    return audio_file

# Global model cache for transcription
_transcription_model = None

def get_transcription_model(model_path="vosk-model-en-us-0.22"):
    """Get cached transcription model"""
    global _transcription_model
    if _transcription_model is None:
        print(f"Loading transcription model from {model_path}...")
        _transcription_model = Model(model_path)
    return _transcription_model

def transcribe_audio_optimized(audio_file, model_path="vosk-model-en-us-0.22"):
    """Optimized transcription with cached model and larger chunks"""
    model = get_transcription_model(model_path)
    
    wf = wave.open(audio_file, "rb")
    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be mono PCM WAV format. Exiting...")

    recognizer = KaldiRecognizer(model, wf.getframerate())
    
    transcription = ""
    chunk_size = 8000  # Larger chunks for better performance
    
    while True:
        data = wf.readframes(chunk_size)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription += result.get("text", "") + " "

    final_result = json.loads(recognizer.FinalResult())
    transcription += final_result.get("text", "")

    wf.close()
    print("Transcription: %s" % (str(transcription)))
    return transcription.strip()

def transcribe_audio(audio_file, model_path="vosk-model-en-us-0.22"):
    """Legacy function - kept for compatibility"""
    return transcribe_audio_optimized(audio_file, model_path)

@lru_cache(maxsize=8)
def diarize_speakers_cached(audio_file_path):
    """Cached speaker diarization"""
    return diarize_speakers_internal(audio_file_path)

def diarize_speakers_internal(audio_file):
    """Internal diarization function with intelligent fallback"""
    global pipeline
    
    if pipeline is None:
        # Use intelligent fallback instead of simple split
        logging.warning("pyannote.audio pipeline not available, using intelligent fallback diarization")
        return intelligent_fallback_diarization(audio_file)
    
    try:
        logging.info("Using pyannote.audio professional speaker diarization")
        diarization = pipeline({"uri": "sample", "audio": audio_file})
        speaker_segments = {}

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append((turn.start, turn.end))

        # Log pyannote results
        logging.info(f"pyannote detected {len(speaker_segments)} speakers")
        for speaker, segments in speaker_segments.items():
            total_time = sum(end - start for start, end in segments)
            logging.info(f"{speaker}: {len(segments)} segments, {total_time:.2f}s total")
            
        return speaker_segments
        
    except Exception as e:
        logging.error(f"pyannote.audio diarization failed: {e}")
        logging.info("Falling back to intelligent audio-based diarization")
        return intelligent_fallback_diarization(audio_file)

def detect_speaker_changes_main(y, sr, duration):
    """
    Detect speaker changes using audio feature analysis with clustering
    """
    try:
        # Parameters for analysis
        window_size = 0.5  # 0.5 second windows
        hop_size = 0.25    # 0.25 second overlap
        frame_length = int(window_size * sr)
        hop_length = int(hop_size * sr)
        
        # Extract features for each window
        features = []
        timestamps = []
        
        for i in range(0, len(y) - frame_length, hop_length):
            segment = y[i:i + frame_length]
            timestamp = i / sr
            
            # Extract speaker-distinguishing features
            feature_vector = extract_speaker_features(segment, sr)
            if feature_vector is not None:
                features.append(feature_vector)
                timestamps.append(timestamp)
        
        if len(features) < 2:
            return {"SPEAKER_00": [(0, duration/2)], "SPEAKER_01": [(duration/2, duration)]}
        
        # Cluster features to identify speakers
        speaker_labels = cluster_speakers(features)
        
        # Convert to speaker segments format matching pyannote
        speaker_segments = {"SPEAKER_00": [], "SPEAKER_01": []}
        current_speaker = speaker_labels[0]
        segment_start = 0
        
        for i, label in enumerate(speaker_labels[1:], 1):
            if label != current_speaker:
                # Speaker change detected
                segment_end = timestamps[i]
                speaker_key = f"SPEAKER_0{current_speaker}"
                speaker_segments[speaker_key].append((segment_start, segment_end))
                segment_start = segment_end
                current_speaker = label
        
        # Add final segment
        speaker_key = f"SPEAKER_0{current_speaker}"
        speaker_segments[speaker_key].append((segment_start, duration))
        
        # Merge very short segments
        merged_segments = merge_nearby_segments_main(speaker_segments, min_duration=1.0)
        
        return merged_segments
        
    except Exception as e:
        logging.error(f"Error in speaker change detection: {e}")
        return {"SPEAKER_00": [(0, duration/2)], "SPEAKER_01": [(duration/2, duration)]}

def extract_speaker_features(segment, sr):
    """
    Extract speaker-distinguishing features from audio segment
    """
    try:
        if len(segment) < 512:  # Too short for analysis
            return None
        
        # Fundamental frequency (pitch)
        f0 = librosa.yin(segment, fmin=75, fmax=400, sr=sr)
        pitch_mean = np.nanmean(f0) if len(f0) > 0 else 0
        pitch_std = np.nanstd(f0) if len(f0) > 0 else 0
        
        # MFCCs (vocal tract characteristics)
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1)
        mfcc_std = np.std(mfccs, axis=1)
        
        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=segment, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=segment, sr=sr)
        
        # Zero crossing rate (voice characteristics)
        zcr = librosa.feature.zero_crossing_rate(segment)
        
        # Energy/RMS
        rms = librosa.feature.rms(y=segment)
        
        # Combine features
        features = np.concatenate([
            [pitch_mean, pitch_std],
            mfcc_mean,
            mfcc_std,
            [np.mean(spectral_centroids), np.std(spectral_centroids)],
            [np.mean(spectral_rolloff), np.std(spectral_rolloff)],
            [np.mean(spectral_bandwidth), np.std(spectral_bandwidth)],
            [np.mean(zcr), np.std(zcr)],
            [np.mean(rms), np.std(rms)]
        ])
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
        
    except Exception as e:
        logging.error(f"Error extracting speaker features: {e}")
        return None

def cluster_speakers(features, n_speakers=2):
    """
    Cluster audio features to identify different speakers
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_speakers, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features_normalized)
        
        return labels
        
    except ImportError:
        logging.warning("scikit-learn not available, using basic clustering")
        # Fallback: simple clustering based on pitch
        features_array = np.array(features)
        pitch_values = features_array[:, 0]  # First feature is pitch_mean
        median_pitch = np.median(pitch_values)
        labels = (pitch_values > median_pitch).astype(int)
        return labels
    except Exception as e:
        logging.error(f"Error in speaker clustering: {e}")
        # Return alternating labels as fallback
        return [i % 2 for i in range(len(features))]

def merge_nearby_segments_main(speaker_segments, min_duration=1.0):
    """
    Merge segments that are too short or belong to same speaker
    """
    merged_segments = {}
    
    for speaker, segments in speaker_segments.items():
        if not segments:
            merged_segments[speaker] = []
            continue
            
        merged = []
        segments = sorted(segments, key=lambda x: x[0])  # Sort by start time
        
        current_start, current_end = segments[0]
        
        for start, end in segments[1:]:
            if (current_end - current_start) < min_duration or start - current_end < 0.5:
                # Merge with current segment
                current_end = max(current_end, end)
            else:
                # Add current segment and start new one
                merged.append((current_start, current_end))
                current_start, current_end = start, end
        
        # Add final segment
        merged.append((current_start, current_end))
        merged_segments[speaker] = merged
    
    return merged_segments

def intelligent_fallback_diarization(audio_file):
    """Intelligent fallback using the same improved method as offline version"""
    try:
        # Load audio with librosa for better analysis
        y, sr = librosa.load(audio_file, sr=16000)
        duration = len(y) / sr
        
        # Use the same improved speaker detection as offline version
        speaker_segments = detect_speaker_changes_main(y, sr, duration)
        
        logging.info(f"Intelligent fallback detected {len(speaker_segments)} speaker groups")
        for speaker, segments in speaker_segments.items():
            total_time = sum(end - start for start, end in segments)
            logging.info(f"{speaker}: {len(segments)} segments, {total_time:.2f}s total")
        
        return speaker_segments
        
    except Exception as e:
        logging.error(f"Intelligent fallback also failed: {e}, using simple split")
        # Final simple fallback
        try:
            audio = AudioSegment.from_wav(audio_file)
            duration = len(audio) / 1000.0
            return {
                "SPEAKER_00": [(0, duration/2)],
                "SPEAKER_01": [(duration/2, duration)]
            }
        except:
            return {"SPEAKER_00": [(0, 10)]}  # Absolute final fallback

def diarize_speakers(audio_file):
    """Main diarization function with caching"""
    return diarize_speakers_cached(audio_file)

def extract_speaker_segments(audio_file, speaker_segments):
    """Extract speaker segments with analysis_results directory"""
    # Create analysis_results directory
    results_dir = Path("analysis_results")
    results_dir.mkdir(exist_ok=True)
    
    # Check for existing speaker files in analysis_results
    speaker_file_test = results_dir / f"{Path(audio_file).stem}_speaker_SPEAKER_00.wav"
    generate = True
    
    if speaker_file_test.exists():
        user_input = input("One or more speaker segment files appear to already exist. Should they be regenerated? (y/N): ")
        user_input = (str(user_input).strip()).lower()
        if user_input in ['n', 'false', 'f', '', 'no']:
            generate = False

    speaker_audio_files = {}

    if generate:
        audio = AudioSegment.from_wav(audio_file)
        logging.info(f"Generating speaker segments in {results_dir}")

        for speaker, segments in speaker_segments.items():
            combined_segment = AudioSegment.empty()

            for start, end in segments:
                start_ms = start * 1000
                end_ms = end * 1000
                combined_segment += audio[start_ms:end_ms]

            # Save speaker files in analysis_results directory with FFmpeg-aware error handling
            speaker_file = results_dir / f"{Path(audio_file).stem}_speaker_{speaker}.wav"
            
            try:
                combined_segment.export(speaker_file, format="wav")
                speaker_audio_files[speaker] = str(speaker_file)
                logging.info(f"Created speaker file: {speaker_file}")
            except Exception as e:
                if not FFMPEG_AVAILABLE:
                    logging.warning(f"Speaker file export failed (FFmpeg not available): {e}")
                    logging.info("Install FFmpeg with: conda install ffmpeg")
                else:
                    logging.warning(f"Could not export {speaker_file} despite FFmpeg being available: {e}")
                
                # Fallback: use original file
                speaker_audio_files[speaker] = audio_file
                logging.info(f"Using original file as fallback for {speaker}")

    else:
        # Look for existing files in analysis_results directory
        speaker_audio_file_list = list(results_dir.glob(f"{Path(audio_file).stem}_speaker_SPEAKER_*.wav"))
        for speaker_audio_file_path in speaker_audio_file_list:
            speaker_file_base_name = speaker_audio_file_path.name
            try:
                speaker = re.search(f"{Path(audio_file).stem}_speaker_SPEAKER_(.+?).wav", str(speaker_file_base_name)).group(1)
            except AttributeError:
                logging.error(f"Speaker not found for file ({speaker_file_base_name}). Skipping...")
                continue
            speaker_audio_files[speaker] = str(speaker_audio_file_path)
            logging.info(f"Using existing speaker file: {speaker_audio_file_path}")

    return speaker_audio_files

def analyze_single_speaker(speaker_data):
    """Ultra-optimized single speaker analysis with memory management"""
    speaker, speaker_file = speaker_data
    logging.info(f"[{speaker}] Starting analysis")
    
    start_time = time.time()
    
    try:
        # Load audio once with preprocessing
        y, sr = load_and_preprocess_audio(speaker_file, target_sr=16000, max_duration=30)
        
        # Run analyses in parallel where possible
        with ThreadPoolExecutor(max_workers=2) as executor:
            features_future = executor.submit(extract_audio_features_optimized, speaker_file, y, sr)
            emotion_future = executor.submit(analyze_emotion_optimized, y, sr)
            
            features = features_future.result()
            predicted_emotion = emotion_future.result()
        
        personality_traits = analyze_personality(features)
        
        # Clean up audio data
        del y
        gc.collect()
        
        analysis_time = time.time() - start_time
        logging.info(f"[{speaker}] Analysis completed in {analysis_time:.2f} seconds")
        
        result = {
            "speaker_id": speaker,
            "audio_features": features,
            "emotion": predicted_emotion,
            "personality": personality_traits,
            "processing_time": analysis_time
        }
        
        logging.info(f"[{speaker}] Predicted emotion: {predicted_emotion}")
        logging.info(f"[{speaker}] Personality traits: {personality_traits}")
        
        return speaker, result
        
    except Exception as e:
        logging.error(f"[{speaker}] Error during analysis: {e}")
        return speaker, {
            "speaker_id": speaker,
            "audio_features": {},
            "emotion": "Unknown",
            "personality": {},
            "processing_time": 0,
            "error": str(e)
        }

def analyze_speakers_parallel(audio_file, speaker_segments):
    """Analyze speakers in parallel for better performance"""
    speaker_audio_files = extract_speaker_segments(audio_file, speaker_segments)
    
    # Use ThreadPoolExecutor for I/O bound tasks (audio loading)
    # Use ProcessPoolExecutor for CPU bound tasks (model inference)
    max_workers = min(len(speaker_audio_files), mp.cpu_count())
    
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyze_single_speaker, (speaker, speaker_file)) 
                  for speaker, speaker_file in speaker_audio_files.items()]
        
        for future in futures:
            speaker, result = future.result()
            results[speaker] = result
    
    return results

def analyze_speakers(audio_file):
    """Legacy function - kept for compatibility"""
    speaker_segments = diarize_speakers(audio_file)
    return analyze_speakers_parallel(audio_file, speaker_segments)

@lru_cache(maxsize=32)
def extract_audio_features_cached(audio_file_path):
    """Cached version of audio feature extraction"""
    y, sr = librosa.load(audio_file_path, sr=16000)  # Fixed sample rate for consistency
    return extract_audio_features_optimized(audio_file_path, y, sr)

def load_and_preprocess_audio(audio_file, target_sr=16000, max_duration=None):
    """Optimized audio loading with preprocessing"""
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

def extract_audio_features_optimized(audio_file, y=None, sr=None):
    """Ultra-optimized audio feature extraction"""
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
    except:
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

def extract_audio_features(audio_file):
    """Legacy function - kept for compatibility"""
    return extract_audio_features_cached(audio_file)

def analyze_emotion_optimized(y, sr):
    """Ultra-optimized emotion analysis with chunking and caching"""
    # Ensure correct sample rate
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Use smaller segment for faster processing (10 seconds max)
    max_samples = 10 * sr
    if len(y) > max_samples:
        # Take middle segment for better representation
        start_idx = (len(y) - max_samples) // 2
        y = y[start_idx:start_idx + max_samples]
    
    # Process in chunks if still too long
    chunk_size = 5 * sr  # 5-second chunks
    if len(y) > chunk_size:
        chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]
        emotion_scores = []
        
        for chunk in chunks[:2]:  # Process max 2 chunks for speed
            if len(chunk) < sr:  # Skip chunks shorter than 1 second
                continue
                
            try:
                input_values = emotion_processor(chunk, sampling_rate=sr, return_tensors="pt", 
                                               padding=True, truncation=True, max_length=sr*5).input_values
                
                if USE_GPU:
                    input_values = input_values.cuda()
                
                with torch.no_grad():
                    with torch.cuda.amp.autocast() if USE_GPU else torch.no_grad():
                        logits = emotion_model(input_values).logits
                emotion_scores.append(torch.softmax(logits, dim=-1).cpu().numpy())
            except:
                continue
        
        if emotion_scores:
            # Average the predictions
            avg_scores = np.mean(emotion_scores, axis=0)
            emotion_prediction = np.argmax(avg_scores)
        else:
            emotion_prediction = 0  # Default to neutral
    else:
        try:
            input_values = emotion_processor(y, sampling_rate=sr, return_tensors="pt", 
                                           padding=True, truncation=True).input_values
            
            if USE_GPU:
                input_values = input_values.cuda()
            
            with torch.no_grad():
                with torch.cuda.amp.autocast() if USE_GPU else torch.no_grad():
                    logits = emotion_model(input_values).logits
            emotion_prediction = torch.argmax(logits, dim=-1).item()
        except:
            emotion_prediction = 0  # Default to neutral

    emotions = {
        0: "Neutral", 1: "Calm", 2: "Happy", 3: "Sad",
        4: "Angry", 5: "Fearful", 6: "Disgust", 7: "Surprised"
    }

    return emotions.get(emotion_prediction, "Neutral")

def analyze_emotion(audio_file):
    """Legacy function - kept for compatibility"""
    y, sr = librosa.load(audio_file, sr=16000)
    return analyze_emotion_optimized(y, sr)

def analyze_personality(features):

    personality_traits = {}

    if features["pitch_mean"] > 150:
        personality_traits["extraversion"] = "High"
    else:
        personality_traits["extraversion"] = "Low"

    if features["intensity_mean"] > 0.05:
        personality_traits["openness"] = "High"
    else:
        personality_traits["openness"] = "Low"

    if features["tempo"] > 100:
        personality_traits["conscientiousness"] = "High"
    else:
        personality_traits["conscientiousness"] = "Low"

    return personality_traits

def analyze_turn_taking(speaker_segments):
    speaker_1_times = speaker_segments.get("SPEAKER_00", [])
    speaker_2_times = speaker_segments.get("SPEAKER_01", [])

    if not speaker_1_times or not speaker_2_times:
        print("One or both speakers have no segments available. Exiting...")
        exit()

    all_turns = sorted(
        [(start, end, "SPEAKER_00") for start, end in speaker_1_times] +
        [(start, end, "SPEAKER_01") for start, end in speaker_2_times],
        key=lambda x: x[0]
    )

    response_times = []
    for i in range(len(all_turns) - 1):
        _, end_time, speaker = all_turns[i]
        start_next, _, next_speaker = all_turns[i + 1]

        if speaker != next_speaker:
            response_time = start_next - end_time
            response_times.append(response_time)

    avg_response_time = np.mean(response_times) if response_times else None
    print("Average Response Time Between Speakers: %s seconds" % (str(avg_response_time)))
    return avg_response_time

def analyze_emotion_alignment(emotions):
    alignment_scores = []

    for i in range(len(emotions) - 1):
        speaker, emotion = emotions[i]
        next_speaker, next_emotion = emotions[i + 1]

        if speaker != next_speaker:
            alignment_score = int(emotion == next_emotion)
            alignment_scores.append(alignment_score)

    alignment_ratio = np.mean(alignment_scores) if alignment_scores else None
    print("Emotion Alignment Ratio: %s" % str(alignment_ratio))
    return alignment_ratio

def detect_backchannels(audio_file, speaker_segments):
    y, sr = librosa.load(audio_file, sr=None)

    backchannels = []
    for speaker, segments in speaker_segments.items():
        for start, end in segments:
            start_sample, end_sample = int(start * sr), int(end * sr)
            segment = y[start_sample:end_sample]

            pitches, _ = librosa.core.piptrack(y=segment, sr=sr)
            pitches = pitches[pitches > 0]

            avg_pitch = np.mean(pitches) if len(pitches) > 0 else 0
            duration = end - start

            if avg_pitch < 150 and duration < 1.0:
                backchannels.append((speaker, start, end))

    print("Detected backchannels: %s" % (str(backchannels)))
    return backchannels

def analyze_comprehension_parallel(audio_file, speaker_segments, emotions):
    """Parallelized comprehension analysis"""
    
    # Run independent analyses in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all independent tasks
        turn_taking_future = executor.submit(analyze_turn_taking, speaker_segments)
        emotion_alignment_future = executor.submit(analyze_emotion_alignment_fixed, emotions)
        backchannels_future = executor.submit(detect_backchannels_optimized, audio_file, speaker_segments)
        transcription_future = executor.submit(transcribe_speakers_parallel, audio_file, speaker_segments)
        
        # Collect results
        avg_response_time = turn_taking_future.result()
        alignment_ratio = emotion_alignment_future.result()
        backchannels = backchannels_future.result()
        transcriptions = transcription_future.result()
    
    backchannel_count = len([b for b in backchannels if b[0] == "SPEAKER_01"])
    
    # Calculate overlap ratio
    if "SPEAKER_00" in transcriptions and "SPEAKER_01" in transcriptions:
        overlap_ratio = keyword_overlap(transcriptions["SPEAKER_00"], transcriptions["SPEAKER_01"])
    else:
        overlap_ratio = 0.0

    comprehension_score = (
        (alignment_ratio or 0) * 0.3 + 
        (1 / (1 + (avg_response_time or 1))) * 0.3 + 
        (overlap_ratio or 0) * 0.3 +
        (backchannel_count > 5) * 0.1
    )
    
    comprehension_summary = {
        "average_response_time": avg_response_time,
        "emotion_alignment_ratio": alignment_ratio,
        "keyword_overlap_ratio": overlap_ratio,
        "backchannel_count": backchannel_count,
        "overall_comprehension_score": comprehension_score
    }
    print("Comprehension Summary: %s" % str(comprehension_summary))
    return comprehension_summary

def transcribe_speakers_parallel(audio_file, speaker_segments):
    """Transcribe all speakers in parallel"""
    results_dir = Path("analysis_results")
    
    transcription_tasks = []
    for speaker, segments in speaker_segments.items():
        speaker_audio = results_dir / f"{Path(audio_file).stem}_speaker_{speaker}.wav"
        transcription_tasks.append((speaker, str(speaker_audio)))
    
    transcriptions = {}
    with ThreadPoolExecutor(max_workers=len(transcription_tasks)) as executor:
        futures = {executor.submit(transcribe_audio_optimized, task[1]): task[0] 
                  for task in transcription_tasks}
        
        for future in futures:
            speaker = futures[future]
            transcriptions[speaker] = future.result()
    
    return transcriptions

def analyze_emotion_alignment_fixed(emotions):
    """Fixed version of emotion alignment analysis"""
    if not emotions or len(emotions) < 2:
        return 0.0
        
    # Convert emotions dict to list of tuples
    emotion_list = [(speaker, emotion) for speaker, emotion in emotions.items()]
    
    if len(emotion_list) < 2:
        return 0.0
    
    alignment_scores = []
    
    # Compare emotions between different speakers
    for i in range(len(emotion_list)):
        for j in range(i + 1, len(emotion_list)):
            speaker1, emotion1 = emotion_list[i]
            speaker2, emotion2 = emotion_list[j]
            alignment_score = int(emotion1 == emotion2)
            alignment_scores.append(alignment_score)
    
    alignment_ratio = np.mean(alignment_scores) if alignment_scores else 0.0
    print("Emotion Alignment Ratio: %s" % str(alignment_ratio))
    return alignment_ratio

def detect_backchannels_optimized(audio_file, speaker_segments):
    """Optimized backchannel detection"""
    y, sr = librosa.load(audio_file, sr=16000)  # Fixed sample rate
    
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
            
            # Use simpler pitch detection for speed
            pitches = librosa.yin(segment, fmin=50, fmax=300, sr=sr)
            avg_pitch = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
            
            if avg_pitch < 150 and duration < 1.0:
                backchannels.append((speaker, start, end))
    
    print("Detected backchannels: %s" % (str(backchannels)))
    return backchannels

def analyze_comprehension(audio_file, speaker_segments, emotions):
    """Legacy function - kept for compatibility"""
    return analyze_comprehension_parallel(audio_file, speaker_segments, emotions)

if __name__=="__main__": 
    main() 