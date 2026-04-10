"""
Manual HuggingFace Model Download Script for WhisperX Models

This script downloads the following models required for WhisperX and related analysis:
1. Systran/faster-whisper-small.en (WhisperX transcription)
2. jonatasgrosman/wav2vec2-large-xlsr-53-english (WhisperX alignment)
3. superb/wav2vec2-large-superb-er (emotion recognition)
4. nlptown/bert-base-multilingual-uncased-sentiment (sentiment analysis)
5. sentence-transformers/all-MiniLM-L6-v2 (similarity scoring)
6. pyannote/speaker-diarization-3.1 (speaker diarization config)
7. pyannote/segmentation-3.0 (diarization segmentation model)
8. pyannote/wespeaker-voxceleb-resnet34-LM (diarization speaker embedding model)

USAGE:
    python download_and_test_huggingface_models.py
"""

import argparse
import os
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
    pipeline,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForSequenceClassification,
)
import torch

parser = argparse.ArgumentParser(description="Download and test HuggingFace models for WhisperX")
parser.add_argument("--cache-dir", default=r"D:\huggingface\hub", help="Local HuggingFace cache directory")
parser.add_argument("--offline", action="store_true", help="Use only cached model files; do not connect to Hugging Face")
args = parser.parse_args()

CACHE_DIR = args.cache_dir
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["TORCH_HOME"] = CACHE_DIR
if args.offline:
    os.environ["HF_HUB_OFFLINE"] = "1"


REPO_REQUIRED_FILES = {
    "Systran/faster-whisper-small.en": [
        "model.bin",
        "config.json",
        "tokenizer.json",
        "vocabulary.txt",
    ],
    "jonatasgrosman/wav2vec2-large-xlsr-53-english": [
        "pytorch_model.bin",
        "config.json",
        "vocab.json",
    ],
    "superb/wav2vec2-large-superb-er": [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "vocab.json",
    ],
    "nlptown/bert-base-multilingual-uncased-sentiment": [
        "pytorch_model.bin",
        "config.json",
        "tokenizer_config.json",
        "vocab.txt",
    ],
    "sentence-transformers/all-MiniLM-L6-v2": [
        ("pytorch_model.bin", "model.safetensors"),
        "config.json",
        "tokenizer_config.json",
        "tokenizer.json",
    ],
    "pyannote/speaker-diarization-3.1": [
        "config.yaml",
    ],
    "pyannote/segmentation-3.0": [
        "config.yaml",
        "pytorch_model.bin",
    ],
    "pyannote/wespeaker-voxceleb-resnet34-LM": [
        "config.yaml",
        "pytorch_model.bin",
    ],
}

def _file_group_exists(snapshot_dir, required):
    from pathlib import Path
    if isinstance(required, (list, tuple)):
        return any((snapshot_dir / f).exists() for f in required)
    return (snapshot_dir / required).exists()


def is_repo_cached(repo_id):
    from pathlib import Path

    cache_path = Path(CACHE_DIR) / f"models--{repo_id.replace('/', '--')}"
    if not cache_path.exists():
        return False

    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists():
        return False

    snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        return False

    snapshot_dir = snapshot_dirs[0]
    required_files = REPO_REQUIRED_FILES.get(repo_id, [])
    if not required_files:
        return False

    return all(_file_group_exists(snapshot_dir, required) for required in required_files)


def download_model_manual(repo_id, model_name):
    """Download a model repository using Hugging Face snapshot download."""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}")

    if is_repo_cached(repo_id):
        print(f"✓ Already cached: {repo_id}")
        return True

    if args.offline:
        print(f"✗ Offline mode enabled and cache is incomplete for: {repo_id}")
        return False

    try:
        print("\nAttempt 1: Using snapshot_download with resume...")
        cache_dir = snapshot_download(
            repo_id=repo_id,
            resume_download=True,
            local_files_only=False,
            cache_dir=CACHE_DIR,
            ignore_patterns=["*.msgpack", "*.h5", "*.ot", "*.onnx"],
        )
        print(f"✓ Downloaded successfully to: {cache_dir}")
        return True

    except Exception as e1:
        error_msg = str(e1)
        print(f"✗ Attempt 1 failed: {error_msg[:200]}")

        if repo_id == "superb/wav2vec2-large-superb-er" and "403" in error_msg:
            print("\n" + "!"*60)
            print("FIREWALL BLOCKING EMOTION MODEL DOWNLOAD")
            print("!"*60)
            print("The large pytorch_model.bin file (1.26GB) may be blocked by your firewall.")
            print("Please download it manually from the model page if needed.")
            print("!"*60)

        try:
            print("\nAttempt 2: Downloading files individually...")

            files_to_download = [
                "config.json",
                "config.yaml",
                "tokenizer_config.json",
                "vocab.txt",
                "tokenizer.json",
                "special_tokens_map.json",
                "preprocessor_config.json",
                "vocab.json",
            ]
            model_files = ["pytorch_model.bin", "model.safetensors"]

            for filename in files_to_download:
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=CACHE_DIR)
                    print(f"  ✓ {filename}")
                except Exception:
                    print(f"  ⚠ {filename} - skipped or not available")

            weights_downloaded = False
            for model_file in model_files:
                try:
                    print(f"\n  Downloading {model_file} (this may take a while)...")
                    hf_hub_download(repo_id=repo_id, filename=model_file, resume_download=True, cache_dir=CACHE_DIR)
                    print(f"  ✓ {model_file}")
                    weights_downloaded = True
                    break
                except Exception as emf:
                    print(f"  ⚠ {model_file} failed: {str(emf)[:100]}")

            if not weights_downloaded and repo_id == "superb/wav2vec2-large-superb-er":
                print("\n" + "!"*60)
                print("MODEL WEIGHTS DOWNLOAD FAILED")
                print("!"*60)
                print("Config files downloaded, but pytorch_model.bin may be blocked by your firewall.")
                print("See manual download instructions on the model page.")
                print("!"*60)

            print("✓ Individual file download completed")
            return True
        except Exception as e2:
            print(f"✗ Attempt 2 failed: {str(e2)[:200]}")
            return False


def test_similarity_model():
    """Test the sentence-transformers similarity model."""
    print(f"\n{'='*60}")
    print("Testing Similarity Model")
    print(f"{'='*60}")

    try:
        print("Loading sentence-transformers/all-MiniLM-L6-v2...")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        print("✓ Model loaded successfully")

        print("\nTesting inference...")
        test_text = "This is a test sentence."
        inputs = tokenizer(test_text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"✓ Inference successful (output shape: {outputs.last_hidden_state.shape})")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_sentiment_model():
    """Test the sentiment analysis model."""
    print(f"\n{'='*60}")
    print("Testing Sentiment Analysis Model")
    print(f"{'='*60}")

    try:
        print("Loading nlptown/bert-base-multilingual-uncased-sentiment...")
        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model=model,
            tokenizer=tokenizer,
        )
        print("✓ Model loaded successfully")

        print("\nTesting inference...")
        test_text = "This product is amazing!"
        result = sentiment_analyzer(test_text)[0]
        print(f"✓ Inference successful")
        print(f"  Result: {result}")
        return True

    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_emotion_recognition_model():
    """Test the audio emotion recognition model."""
    print(f"\n{'='*60}")
    print("Testing Emotion Recognition Model")
    print(f"{'='*60}")

    try:
        print("Loading superb/wav2vec2-large-superb-er...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        print("✓ Feature extractor loaded")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True,
            cache_dir=CACHE_DIR,
        )
        print("✓ Model loaded successfully")

        print("\n⚠ Audio model - requires audio file for full test")
        print("✓ Model structure loaded correctly")
        return True

    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed: {error_msg}")

        if "pytorch_model.bin" in error_msg or "NoneType" in error_msg:
            print("\n" + "!"*60)
            print("MISSING MODEL WEIGHTS FILE")
            print("!"*60)
            print("The pytorch_model.bin file is missing or incomplete.")
            print("")
            print("SOLUTION:")
            print("1. Go to: https://huggingface.co/superb/wav2vec2-large-superb-er/tree/main")
            print("2. Click on 'pytorch_model.bin' in the file list")
            print("3. Click the download button to save the file (1.26GB)")
            print("4. Place the downloaded file in:")
            cache_base = os.path.join(
                CACHE_DIR,
                "models--superb--wav2vec2-large-superb-er",
                "snapshots",
            )
            if os.path.exists(cache_base):
                snapshots = os.listdir(cache_base)
                if snapshots:
                    snapshot_dir = os.path.join(cache_base, snapshots[0])
                    print(f"   {snapshot_dir}")
                    print(f"\n   Full path should be:")
                    print(f"   {os.path.join(snapshot_dir, 'pytorch_model.bin')}")
            else:
                print(f"   {cache_base}/<hash>/pytorch_model.bin")
            print("5. Re-run this script")
            print("!"*60)
        return False


def test_whisper_model(repo_id):
    """Test the Whisper model by checking if files exist in cache."""
    print(f"\n{'='*60}")
    print(f"Testing Whisper Model: {repo_id}")
    print(f"{'='*60}")
    
    try:
        # For faster-whisper models, just check if the cache directory exists
        import os
        from pathlib import Path
        
        # Convert repo_id to cache path format
        cache_path = os.path.join(CACHE_DIR, f"models--{repo_id.replace('/', '--')}")
        
        if os.path.exists(cache_path):
            # Check if snapshots directory exists
            snapshots_dir = os.path.join(cache_path, "snapshots")
            if os.path.exists(snapshots_dir):
                snapshots = os.listdir(snapshots_dir)
                if snapshots:
                    snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
                    # Check for model files
                    model_files = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
                    found_files = [f for f in model_files if os.path.exists(os.path.join(snapshot_dir, f))]
                    
                    if found_files:
                        print(f"✓ Model files found in cache: {found_files}")
                        print(f"  Cache location: {snapshot_dir}")
                        return True
                    else:
                        print(f"⚠ Cache directory exists but model files missing: {snapshot_dir}")
                        return False
                else:
                    print(f"⚠ Cache directory exists but no snapshots: {snapshots_dir}")
                    return False
            else:
                print(f"⚠ Cache directory exists but no snapshots directory: {cache_path}")
                return False
        else:
            print(f"✗ Model not found in cache: {cache_path}")
            return False
            
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False


def test_alignment_model():
    """Test the alignment model by loading it (which caches it if needed)."""
    print(f"\n{'='*60}")
    print("Testing Alignment Model")
    print(f"{'='*60}")
    
    try:
        import whisperx
        print("Loading alignment model...")
        model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu", model_dir=CACHE_DIR)
        print("✓ Alignment model loaded successfully")
        return True
    except Exception as e:
        print(f"✗ Alignment model test failed: {e}")
        return False


def _check_model_cache(repo_id, required_files):
    from pathlib import Path

    cache_path = Path(CACHE_DIR) / f"models--{repo_id.replace('/', '--')}"
    if not cache_path.exists():
        print(f"✗ Cache directory not found: {cache_path}")
        return False

    snapshots_dir = cache_path / "snapshots"
    if not snapshots_dir.exists():
        print(f"✗ Snapshots directory not found: {snapshots_dir}")
        return False

    snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
    if not snapshot_dirs:
        print(f"✗ No snapshot folders found in: {snapshots_dir}")
        return False

    snapshot_dir = snapshot_dirs[0]
    missing_files = [f for f in required_files if not (snapshot_dir / f).exists()]
    if missing_files:
        print(f"✗ Missing files for {repo_id} in: {snapshot_dir}")
        for f in missing_files:
            print(f"  Missing: {f}")
        return False

    print(f"✓ Cached {repo_id}: {required_files}")
    return True


def test_diarization_model():
    """Test the diarization model by checking cache files for the pipeline and its dependencies."""
    print(f"\n{'='*60}")
    print("Testing Diarization Model")
    print(f"{'='*60}")

    try:
        success = True

        print("Checking speaker-diarization config cache...")
        success &= _check_model_cache("pyannote/speaker-diarization-3.1", ["config.yaml"])

        print("Checking segmentation model cache...")
        success &= _check_model_cache("pyannote/segmentation-3.0", ["config.yaml", "pytorch_model.bin"])

        print("Checking embedding model cache...")
        success &= _check_model_cache("pyannote/wespeaker-voxceleb-resnet34-LM", ["config.yaml", "pytorch_model.bin"])

        return bool(success)

    except Exception as e:
        print(f"✗ Diarization cache test failed: {e}")
        return False


def main():
    print(f"\n{'='*60}")
    print("Manual HuggingFace Model Download & Test for WhisperX")
    print(f"{'='*60}")
    print("\nNOTE: No authentication required for WhisperX models.")
    print("="*60)

    print("\n" + "="*60)
    print("PHASE 1: DOWNLOADING MODELS")
    print("="*60)

    whisper_small_downloaded = download_model_manual(
        "Systran/faster-whisper-small.en",
        "Whisper Small EN Model",
    )
    
    alignment_downloaded = download_model_manual(
        "jonatasgrosman/wav2vec2-large-xlsr-53-english",
        "WhisperX Alignment Model",
    )
    
    emotion_downloaded = download_model_manual(
        "superb/wav2vec2-large-superb-er",
        "Emotion Recognition Model",
    )
    
    sentiment_downloaded = download_model_manual(
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "Sentiment Analysis Model",
    )
    
    similarity_downloaded = download_model_manual(
        "sentence-transformers/all-MiniLM-L6-v2",
        "Similarity Model",
    )

    diarization_config_downloaded = download_model_manual(
        "pyannote/speaker-diarization-3.1",
        "Diarization Config",
    )

    segmentation_downloaded = download_model_manual(
        "pyannote/segmentation-3.0",
        "Diarization Segmentation Model",
    )

    embedding_downloaded = download_model_manual(
        "pyannote/wespeaker-voxceleb-resnet34-LM",
        "Diarization Embedding Model",
    )

    print("\n" + "="*60)
    print("PHASE 2: TESTING MODELS")
    print("="*60)

    whisper_small_ok = test_whisper_model("Systran/faster-whisper-small.en") if whisper_small_downloaded else False
    alignment_ok = test_alignment_model() if alignment_downloaded else False
    emotion_ok = test_emotion_recognition_model() if emotion_downloaded else False
    sentiment_ok = test_sentiment_model() if sentiment_downloaded else False
    similarity_ok = test_similarity_model() if similarity_downloaded else False
    diarization_ok = test_diarization_model() if (diarization_config_downloaded and segmentation_downloaded and embedding_downloaded) else False

    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Whisper Small EN:          {'✓ SUCCESS' if whisper_small_ok else '✗ FAILED'}")
    print(f"Alignment Model:           {'✓ SUCCESS' if alignment_ok else '✗ FAILED'}")
    print(f"Emotion Model:             {'✓ SUCCESS' if emotion_ok else '✗ FAILED'}")
    print(f"Sentiment Model:           {'✓ SUCCESS' if sentiment_ok else '✗ FAILED'}")
    print(f"Similarity Model:          {'✓ SUCCESS' if similarity_ok else '✗ FAILED'}")
    print(f"Diarization Model:         {'✓ SUCCESS' if diarization_ok else '✗ FAILED'}")

    total_success = sum([whisper_small_ok, alignment_ok, emotion_ok, sentiment_ok, similarity_ok, diarization_ok])

    if total_success == 6:
        print(f"\n🎉 All {total_success}/6 model groups downloaded and tested successfully!")
    elif total_success > 0:
        print(f"\n⚠ {total_success}/6 model groups working, {6-total_success} failed")
    else:
        print("\n✗ All model groups failed")

    print("\nNext steps:")
    if total_success > 0:
        print("1. The working models are now cached locally")
        print("2. Add 'local_files_only=True' to your code where applicable")
        print("3. Models will load from cache without internet access")

    if total_success < 4:
        print("\nAlternative solutions for failed models:")
        print("1. Download models on a personal device/network")
        print("2. Transfer the cache folder to your work laptop")
        print("3. Contact IT to whitelist: cas-server.xethub.hf.co")
        print(f"\nCache location: {CACHE_DIR}")


if __name__ == "__main__":
    main()
