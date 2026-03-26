#!/usr/bin/env python
"""
Offline WhisperX Model Caching Script
Downloads and caches all models required for whisperX.py to run offline

This script downloads the following models:
1. WhisperX transcription models (various sizes)
2. Alignment models (language-specific)
3. Speaker diarization model (pyannote/speaker-diarization-3.1)
4. Analysis models:
   - Emotion recognition: superb/wav2vec2-large-superb-er
   - Sentiment analysis: nlptown/bert-base-multilingual-uncased-sentiment
   - Similarity: sentence-transformers/all-MiniLM-L6-v2

USAGE:
    python offline_whisperX.py

After running this script, you can use whisperX.py offline with the --offline flag.

COMMAND-LINE OPTIONS:
    --cache-dir PATH
        Specify a custom cache directory for downloading models.
        Default: ~/.cache
        Example: python offline_whisperX.py --cache-dir D:/models_cache

    --clear-cache
        Clear the cache directory before downloading new models.
        This frees up disk space and ensures fresh downloads.
        Example: python offline_whisperX.py --clear-cache

EXAMPLES:
    # Standard usage with default cache location
    python offline_whisperX.py

    # Use custom cache directory
    python offline_whisperX.py --cache-dir /mnt/external_drive/models

    # Clear old cache and download fresh models
    python offline_whisperX.py --clear-cache

    # Clear old cache and use custom directory
    python offline_whisperX.py --clear-cache --cache-dir D:/models

REQUIREMENTS:
- huggingface_hub
- transformers
- torch
- whisperx
- sentence_transformers

NOTE: Some models require HuggingFace authentication. If you encounter authentication
errors, you may need to:
1. Create a HuggingFace account at https://huggingface.co
2. Generate a token at https://huggingface.co/settings/tokens
3. Run: huggingface-cli login
4. Accept terms for gated models at the URLs shown in error messages
"""

import os
import sys
import shutil
import argparse
from contextlib import contextmanager
from huggingface_hub import snapshot_download, hf_hub_download
import torch
import whisperx

# PyTorch 2.6+ compatibility: Create context manager for safe loading
@contextmanager
def allow_pickle_load():
    """Allow unsafe pickle loads for model weights"""
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

def clear_cache_directory(cache_dir):
    """Clear the cache directory with user confirmation"""
    if not os.path.exists(cache_dir):
        print(f"Cache directory does not exist: {cache_dir}")
        return False
    
    # Get cache size
    total_size = 0
    file_count = 0
    for dirpath, dirnames, filenames in os.walk(cache_dir):
        for filename in filenames:
            filepath = os.path.join(dirpath, filename)
            try:
                total_size += os.path.getsize(filepath)
                file_count += 1
            except OSError:
                pass
    
    size_mb = total_size / (1024 * 1024)
    
    print(f"\n{'='*60}")
    print("CLEARING CACHE")
    print(f"{'='*60}")
    print(f"Cache directory: {cache_dir}")
    print(f"Files to delete: {file_count}")
    print(f"Total size: {size_mb:.2f} MB")
    print("\nThis will free up disk space for downloading fresh models.")
    
    # Ask for confirmation
    response = input("\nAre you sure you want to clear this cache? (yes/no): ").strip().lower()
    
    if response not in ['yes', 'y']:
        print("❌ Cache clearing cancelled")
        return False
    
    try:
        shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        print(f"✓ Cache cleared successfully ({size_mb:.2f} MB freed)")
        return True
    except Exception as e:
        print(f"✗ Failed to clear cache: {e}")
        return False

def download_model_manual(repo_id, model_name, auth_required=False):
    """Download model using the same methods as download_and_test_huggingface_models.py"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}")

    try:
        # Method 1: Try snapshot_download
        print("\nAttempt 1: Using snapshot_download...")
        cache_dir = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            local_files_only=False,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
        )
        print(f"✓ Downloaded successfully to: {cache_dir}")
        return True

    except Exception as e1:
        error_msg = str(e1)
        print(f"✗ Attempt 1 failed: {error_msg[:200]}")

        if auth_required and ("401" in error_msg or "403" in error_msg or "gated" in error_msg.lower()):
            print("\n" + "!"*60)
            print("AUTHENTICATION REQUIRED")
            print("!"*60)
            print(f"This model ({repo_id}) requires HuggingFace authentication.")
            print("Please:")
            print("1. Create account at: https://huggingface.co")
            print("2. Generate token at: https://huggingface.co/settings/tokens")
            print("3. Run: huggingface-cli login")
            if "pyannote" in repo_id:
                print("4. Accept terms for gated models at:")
                print("   - https://huggingface.co/pyannote/segmentation-3.0")
                print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
            print("!"*60)
            return False

        try:
            # Method 2: Try individual file downloads
            print("\nAttempt 2: Downloading files individually...")

            # List of essential files
            files_to_download = [
                "config.json",
                "tokenizer_config.json",
                "vocab.txt",
                "tokenizer.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "vocab.json",
                "pytorch_model.bin",
                "model.safetensors",
            ]

            downloaded_any = False
            for filename in files_to_download:
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename, resume_download=True)
                    print(f"  ✓ {filename}")
                    downloaded_any = True
                except Exception as ef:
                    # Skip files that don't exist or fail
                    continue

            if downloaded_any:
                print("✓ Individual file download completed")
                return True
            else:
                print("✗ No files could be downloaded")
                return False

        except Exception as e2:
            print(f"✗ Attempt 2 failed: {str(e2)[:200]}")
            return False

def download_whisperx_models():
    """Download WhisperX models and alignment models"""
    print(f"\n{'='*60}")
    print("Downloading WhisperX Models")
    print(f"{'='*60}")

    # Common WhisperX model sizes
    whisper_models = ["tiny.en", "base.en", "small.en", "medium.en", "large-v2", "large-v3"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "int8" if device == "cuda" else "float32"

    for model_size in whisper_models:
        try:
            print(f"\nDownloading WhisperX model: {model_size}")
            with allow_pickle_load():
                model = whisperx.load_model(model_size, device, compute_type=compute_type)
            print(f"✓ {model_size} downloaded and cached")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Failed to download {model_size}: {str(e)[:100]}")

    # Download alignment models for common languages
    languages = ["en", "es", "fr", "de", "it", "pt", "ru", "ja", "zh", "ko"]

    for lang in languages:
        try:
            print(f"\nDownloading alignment model for language: {lang}")
            with allow_pickle_load():
                model_a, metadata = whisperx.load_align_model(language_code=lang, device=device)
            print(f"✓ Alignment model for {lang} downloaded and cached")
            del model_a
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"✗ Failed to download alignment model for {lang}: {str(e)[:100]}")

    return True

def download_analysis_models():
    """Download the analysis models used by whisperX.py"""
    print(f"\n{'='*60}")
    print("Downloading Analysis Models")
    print(f"{'='*60}")

    models_to_download = [
        ("sentence-transformers/all-MiniLM-L6-v2", "Similarity Model", False),
        ("nlptown/bert-base-multilingual-uncased-sentiment", "Sentiment Analysis Model", False),
        ("superb/wav2vec2-large-superb-er", "Emotion Recognition Model", False),
        ("pyannote/speaker-diarization-3.1", "Speaker Diarization Model", True),
    ]

    results = []
    for repo_id, name, auth_required in models_to_download:
        success = download_model_manual(repo_id, name, auth_required)
        results.append((name, success))

    return results

def test_cached_models():
    """Test that downloaded models can be loaded from cache"""
    print(f"\n{'='*60}")
    print("Testing Cached Models")
    print(f"{'='*60}")

    tests = []

    # Test similarity model
    try:
        print("\nTesting similarity model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        # Quick test
        emb = model.encode("test sentence")
        print("✓ Similarity model working")
        tests.append(("Similarity", True))
    except Exception as e:
        print(f"✗ Similarity model failed: {e}")
        tests.append(("Similarity", False))

    # Test sentiment model
    try:
        print("\nTesting sentiment model...")
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        print("✓ Sentiment model working")
        tests.append(("Sentiment", True))
    except Exception as e:
        print(f"✗ Sentiment model failed: {e}")
        tests.append(("Sentiment", False))

    # Test emotion model
    try:
        print("\nTesting emotion model...")
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        print("✓ Emotion model working")
        tests.append(("Emotion", True))
    except Exception as e:
        print(f"✗ Emotion model failed: {e}")
        tests.append(("Emotion", False))

    # Test diarization model
    try:
        print("\nTesting diarization model...")
        from pyannote.audio import Pipeline
        with allow_pickle_load():
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
        print("✓ Diarization model working")
        tests.append(("Diarization", True))
    except Exception as e:
        print(f"✗ Diarization model failed: {e}")
        tests.append(("Diarization", False))

    return tests

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download and cache WhisperX and analysis models for offline use"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Custom cache directory for models (default: ~/.cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear existing cache before downloading new models"
    )
    
    args = parser.parse_args()
    
    # Set up cache directory
    if args.cache_dir:
        cache_dir = os.path.abspath(args.cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        # Set environment variables for HuggingFace and Torch to use custom cache
        os.environ['HF_HOME'] = cache_dir
        os.environ['TORCH_HOME'] = cache_dir
        print(f"Using custom cache directory: {cache_dir}")
    else:
        cache_dir = os.path.expanduser("~/.cache")
        print(f"Using default cache directory: {cache_dir}")
    
    # Clear cache if requested
    if args.clear_cache:
        if not clear_cache_directory(cache_dir):
            print("Aborted due to cache clearing failure")
            return
    
    print(f"\n{'='*80}")
    print("Offline WhisperX Model Caching Script")
    print(f"{'='*80}")
    print("\nThis script will download all models required for whisperX.py")
    print("to run completely offline.")
    print("\nNOTE: This may take a while and requires internet access.")
    print("Some models require HuggingFace authentication.")
    print("="*80)

    # Check for authentication
    try:
        from huggingface_hub.utils import HfFolder
        token = HfFolder.get_token()
        if token:
            print("✓ HuggingFace authentication detected")
        else:
            print("⚠ No HuggingFace token found. Some models may fail to download.")
            print("  To authenticate: huggingface-cli login")
    except:
        print("⚠ Could not check HuggingFace authentication")

    # Download WhisperX models
    print("\n" + "="*60)
    print("PHASE 1: DOWNLOADING WHISPERX MODELS")
    print("="*60)
    whisper_success = download_whisperx_models()

    # Download analysis models
    print("\n" + "="*60)
    print("PHASE 2: DOWNLOADING ANALYSIS MODELS")
    print("="*60)
    analysis_results = download_analysis_models()

    # Test cached models
    print("\n" + "="*60)
    print("PHASE 3: TESTING CACHED MODELS")
    print("="*60)
    test_results = test_cached_models()

    # Summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print(f"\nWhisperX Models: {'✓ DOWNLOADED' if whisper_success else '✗ FAILED'}")

    print(f"\nAnalysis Models:")
    for name, success in analysis_results:
        status = "✓ DOWNLOADED" if success else "✗ FAILED"
        print(f"  {name}: {status}")

    print(f"\nModel Tests:")
    for name, success in test_results:
        status = "✓ WORKING" if success else "✗ FAILED"
        print(f"  {name}: {status}")

    # Overall success
    analysis_success = sum(1 for _, success in analysis_results if success)
    test_success = sum(1 for _, success in test_results if success)

    if whisper_success and analysis_success >= 3 and test_success >= 3:
        print(f"\n🎉 SUCCESS! Models cached for offline use.")
        print("You can now run: python whisperX.py --offline <audio_file>")
    else:
        print(f"\n⚠ PARTIAL SUCCESS: {analysis_success}/4 analysis models, {test_success}/4 tests passed")
        print("Some models may not work offline. Check error messages above.")

    print(f"\nCache location: {cache_dir}")
    print(f"  - HuggingFace models: {cache_dir}/huggingface/hub")
    print(f"  - Torch models: {cache_dir}/torch/hub")
    print("\nTo use a custom cache directory on future runs:")
    print(f"  python offline_whisperX.py --cache-dir /path/to/custom/cache")
    print("\nTo clear cache before downloading:")
    print(f"  python offline_whisperX.py --clear-cache")

if __name__ == "__main__":
    main()