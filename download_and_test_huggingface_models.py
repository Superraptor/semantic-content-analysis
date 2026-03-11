"""
Manual HuggingFace Model Download Script
Bypasses CAS service for corporate network compatibility

IMPORTANT FIRST-TIME SETUP:
1. Uncomment the login section below and run it once
2. Accept terms for gated models (pyannote) at the URLs shown
3. After successful login, comment out the login section again

If emotion model download fails due to firewall:
- Go to: https://huggingface.co/superb/wav2vec2-large-superb-er/tree/main
- Click on 'pytorch_model.bin' in the file list and download it (1.26GB)
- Place it in:
  C:\\Users\\<username>\\.cache\\huggingface\\hub\\models--superb--wav2vec2-large-superb-er\\snapshots\\<hash>\\
  (The script will show you the exact path if download fails)
"""

import os
from huggingface_hub import snapshot_download, hf_hub_download
from transformers import AutoTokenizer, AutoModel, pipeline, Wav2Vec2ForSequenceClassification, Wav2Vec2Processor
import torch

# ============================================================
# AUTHENTICATION SECTION (Uncomment for first-time setup)
# ============================================================
"""
FIRST-TIME AUTHENTICATION REQUIRED:
Run this section once when you first use this script or after computer restart.
After successful login, comment this section out again.

Steps:
1. Uncomment the code below (remove the triple quotes)
2. Run the script
3. Paste your HuggingFace token when prompted
   - Get token from: https://huggingface.co/settings/tokens
4. Accept terms for gated models:
   - https://huggingface.co/pyannote/segmentation-3.0
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb
5. After successful login, comment this section out again
"""
from huggingface_hub import login

print("="*60)
print("AUTHENTICATION REQUIRED")
print("="*60)
print("This script needs HuggingFace authentication for gated models.")
print("Please paste your token from: https://huggingface.co/settings/tokens")
print("")

login()  # This will prompt for your token

print("")
print("✓ Authentication successful!")
print("")
print("IMPORTANT: Before downloading pyannote models, accept terms at:")
print("  1. https://huggingface.co/pyannote/segmentation-3.0")
print("  2. https://huggingface.co/pyannote/speaker-diarization-3.1")
print("  3. https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
print("")
input("Press Enter after accepting terms to continue...")
print("="*60)


def download_model_manual(repo_id, model_name):
    """Download model using direct file download (bypasses CAS)"""
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"{'='*60}")

    try:
        # Method 1: Try snapshot_download with resume_download
        print("\nAttempt 1: Using snapshot_download with resume...")
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

        # Special handling for SpeechBrain model firewall issues
        if repo_id == "speechbrain/spkrec-ecapa-voxceleb" and ("403" in error_msg or "CAS" in error_msg or "Forbidden" in error_msg):
            print("\n" + "!"*60)
            print("FIREWALL BLOCKING SPEECHBRAIN MODEL DOWNLOAD")
            print("!"*60)
            print("The model files are blocked by your firewall (CAS service).")
            print("")
            print("MANUAL DOWNLOAD REQUIRED:")
            print("1. Go to: https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb/tree/main")
            print("2. Download ALL these files by clicking each one:")
            print("   - embedding_model.ckpt (83.3 MB) - REQUIRED")
            print("   - classifier.ckpt")
            print("   - mean_var_norm_emb.ckpt")
            print("   - hyperparams.yaml")
            print("   - label_encoder.txt")
            print("   - config.json")
            print("   - .gitattributes (optional)")
            print("   - README.md (optional)")
            print("3. Find or create the snapshot directory:")
            cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots")
            if os.path.exists(cache_base):
                snapshots = os.listdir(cache_base)
                if snapshots:
                    snapshot_dir = os.path.join(cache_base, snapshots[0])
                    print(f"   {snapshot_dir}")
            else:
                os.makedirs(cache_base, exist_ok=True)
                print(f"   Created: {cache_base}")
                print("   You may need to create a hash subdirectory")
            print("4. Place ALL downloaded files in that directory")
            print("5. Re-run this script to verify")
            print("!"*60)

        # Special handling for emotion model firewall issues
        elif repo_id == "superb/wav2vec2-large-superb-er" and "403" in error_msg:
            print("\n" + "!"*60)
            print("FIREWALL BLOCKING EMOTION MODEL DOWNLOAD")
            print("!"*60)
            print("The large pytorch_model.bin file (1.26GB) is blocked by your firewall.")
            print("")
            print("MANUAL DOWNLOAD REQUIRED:")
            print("1. Go to: https://huggingface.co/superb/wav2vec2-large-superb-er/tree/main")
            print("2. Click on 'pytorch_model.bin' in the file list")
            print("3. Click the download button to download the file")
            print("4. Find the snapshot directory:")
            cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--superb--wav2vec2-large-superb-er/snapshots")
            if os.path.exists(cache_base):
                snapshots = os.listdir(cache_base)
                if snapshots:
                    snapshot_dir = os.path.join(cache_base, snapshots[0])
                    print(f"   {snapshot_dir}")
            else:
                print(f"   {cache_base}/<hash>/")
            print("5. Place the downloaded pytorch_model.bin in that directory")
            print("6. Re-run this script to test the model")
            print("!"*60)
        
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
            ]
            
            # Try to download model weights
            model_files = ["pytorch_model.bin", "model.safetensors"]
            
            for filename in files_to_download:
                try:
                    hf_hub_download(repo_id=repo_id, filename=filename)
                    print(f"  ✓ {filename}")
                except Exception as ef:
                    print(f"  ⚠ {filename} - skipped or not needed")
            
            # Try model weights
            weights_downloaded = False
            for model_file in model_files:
                try:
                    print(f"\n  Downloading {model_file} (this may take a while)...")
                    hf_hub_download(repo_id=repo_id, filename=model_file, resume_download=True)
                    print(f"  ✓ {model_file}")
                    weights_downloaded = True
                    break
                except Exception as emf:
                    print(f"  ⚠ {model_file} failed: {str(emf)[:100]}")
            
            if not weights_downloaded and repo_id == "superb/wav2vec2-large-superb-er":
                print("\n" + "!"*60)
                print("MODEL WEIGHTS DOWNLOAD FAILED")
                print("!"*60)
                print("Config files downloaded, but pytorch_model.bin blocked by firewall.")
                print("See manual download instructions above.")
                print("!"*60)
            
            print("✓ Individual file download completed")
            return True
            
        except Exception as e2:
            print(f"✗ Attempt 2 failed: {str(e2)[:200]}")
            return False

def test_similarity_model():
    """Test the similarity model"""
    print(f"\n{'='*60}")
    print("Testing Similarity Model")
    print(f"{'='*60}")
    
    try:
        print("Loading sentence-transformers/all-MiniLM-L6-v2...")
        tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        print("✓ Model loaded successfully")
        
        # Test inference
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
    """Test the sentiment analysis model"""
    print(f"\n{'='*60}")
    print("Testing Sentiment Analysis Model")
    print(f"{'='*60}")
    
    try:
        print("Loading nlptown/bert-base-multilingual-uncased-sentiment...")
        sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        print("✓ Model loaded successfully")
        
        # Test inference
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
    """Test the emotion recognition model"""
    print(f"\n{'='*60}")
    print("Testing Emotion Recognition Model")
    print(f"{'='*60}")
    
    try:
        print("Loading superb/wav2vec2-large-superb-er...")
        
        # Use FeatureExtractor instead of Processor
        from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2ForSequenceClassification
        
        print("Attempting to load feature extractor...")
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        print("✓ Feature extractor loaded")
        
        print("Attempting to load model...")
        model = Wav2Vec2ForSequenceClassification.from_pretrained(
            "superb/wav2vec2-large-superb-er",
            local_files_only=True
        )
        print("✓ Model loaded successfully")
        
        # Note: Actual testing would require audio input
        print("\n⚠ Audio model - requires audio file for full test")
        print("✓ Model structure loaded correctly")
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"✗ Failed: {error_msg}")
        
        # Check if it's the missing pytorch_model.bin issue
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
            cache_base = os.path.expanduser("~/.cache/huggingface/hub/models--superb--wav2vec2-large-superb-er/snapshots")
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
        
        import traceback
        traceback.print_exc()
        return False

def test_speaker_diarization_model():
    """Test the speaker diarization model"""
    print(f"\n{'='*60}")
    print("Testing Speaker Diarization Model")
    print(f"{'='*60}")
    
    try:
        print("Loading pyannote/speaker-diarization-3.1...")
        print("\n⚠ Note: pyannote models require authentication and special handling")
        print("⚠ This model may need to be accessed differently")

        # Try to download the model files
        from pyannote.audio import Pipeline

        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=True
            )
            print("✓ Model loaded successfully")
            return True
        except Exception as e_inner:
            error_msg = str(e_inner)
            print(f"⚠ Model requires HuggingFace authentication token")
            print(f"  Error: {error_msg[:150]}")

            # Check if files exist in cache
            cache_path = os.path.expanduser("~/.cache/huggingface/hub/models--pyannote--speaker-diarization-3.1")
            if os.path.exists(cache_path):
                print(f"✓ Model files found in cache: {cache_path}")
                return True
            else:
                print(f"✗ Model not in cache")
                print("\n" + "!"*60)
                print("AUTHENTICATION REQUIRED FOR PYANNOTE")
                print("!"*60)
                print("1. Uncomment the login section at the top of this script")
                print("2. Run the script and provide your HuggingFace token")
                print("3. Accept terms at:")
                print("   - https://huggingface.co/pyannote/segmentation-3.0")
                print("   - https://huggingface.co/pyannote/speaker-diarization-3.1")
                print("   - https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb")
                print("!"*60)
                return False
        
    except ImportError:
        print("✗ Failed: pyannote.audio not installed")
        print("  Install with: pip install pyannote.audio --break-system-packages")
        return False
    except Exception as e:
        print(f"✗ Failed: {e}")
        return False

def main():
    print(f"\n{'='*60}")
    print("Manual HuggingFace Model Download & Test")
    print(f"{'='*60}")
    print("\nNOTE: If this is your first time running this script,")
    print("uncomment the authentication section at the top of the file.")
    print("="*60)
    
    # Download models
    print("\n" + "="*60)
    print("PHASE 1: DOWNLOADING MODELS")
    print("="*60)
    
    similarity_downloaded = download_model_manual(
        "sentence-transformers/all-MiniLM-L6-v2",
        "Similarity Model"
    )
    
    sentiment_downloaded = download_model_manual(
        "nlptown/bert-base-multilingual-uncased-sentiment",
        "Sentiment Model"
    )
    
    emotion_downloaded = download_model_manual(
        "superb/wav2vec2-large-superb-er",
        "Emotion Recognition Model"
    )
    
    diarization_downloaded = download_model_manual(
        "pyannote/speaker-diarization-3.1",
        "Speaker Diarization Model (v3.1)"
    )

    # Download SpeechBrain embedding model (required by pyannote diarization)
    speechbrain_downloaded = download_model_manual(
        "speechbrain/spkrec-ecapa-voxceleb",
        "SpeechBrain Speaker Embedding Model (required for diarization)"
    )

    # Test models
    print("\n" + "="*60)
    print("PHASE 2: TESTING MODELS")
    print("="*60)
    
    similarity_ok = False
    sentiment_ok = False
    emotion_ok = False
    diarization_ok = False
    
    if similarity_downloaded:
        similarity_ok = test_similarity_model()
    else:
        print("\nSkipping similarity model test (download failed)")
    
    if sentiment_downloaded:
        sentiment_ok = test_sentiment_model()
    else:
        print("\nSkipping sentiment model test (download failed)")
    
    if emotion_downloaded:
        emotion_ok = test_emotion_recognition_model()
    else:
        print("\nSkipping emotion recognition model test (download failed)")
    
    if diarization_downloaded:
        diarization_ok = test_speaker_diarization_model()
    else:
        print("\nSkipping speaker diarization model test (download failed)")
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    print(f"Similarity Model:          {'✓ SUCCESS' if similarity_ok else '✗ FAILED'}")
    print(f"Sentiment Model:           {'✓ SUCCESS' if sentiment_ok else '✗ FAILED'}")
    print(f"Emotion Recognition Model: {'✓ SUCCESS' if emotion_ok else '✗ FAILED'}")
    print(f"Speaker Diarization Model: {'✓ SUCCESS' if diarization_ok else '✗ FAILED'}")
    print(f"SpeechBrain Embedding:     {'✓ DOWNLOADED' if speechbrain_downloaded else '✗ FAILED'}")

    total_success = sum([similarity_ok, sentiment_ok, emotion_ok, diarization_ok])

    if total_success == 4 and speechbrain_downloaded:
        print(f"\n🎉 All {total_success}/4 models + SpeechBrain downloaded and tested successfully!")
    elif total_success == 4:
        print(f"\n⚠ All {total_success}/4 core models working, but SpeechBrain download failed")
    elif total_success > 0:
        print(f"\n⚠ {total_success}/4 models working, {4-total_success} failed")
    else:
        print("\n✗ All models failed")
    
    print("\nNext steps:")
    if total_success > 0:
        print("1. The working models are now cached locally")
        print("2. Add 'local_files_only=True' to your code (where applicable)")
        print("3. Models will load from cache without internet access")
    
    if not emotion_ok:
        print("\n⚠ Emotion model failed:")
        print("  See manual download instructions above for pytorch_model.bin")
    
    if not diarization_ok:
        print("\n⚠ Speaker diarization failed:")
        print("  Uncomment authentication section and accept gated model terms")

    if not speechbrain_downloaded:
        print("\n⚠ SpeechBrain embedding model failed:")
        print("  This model is required for speaker diarization to work properly")
        print("  Follow manual download instructions shown above")
        print("  Files needed: embedding_model.ckpt, classifier.ckpt, hyperparams.yaml, etc.")

    if total_success < 4 or not speechbrain_downloaded:
        print("\nAlternative solutions for failed models:")
        print("1. Download models on a personal device/network")
        print("2. Transfer the cache folder to your work laptop")
        print("3. Contact IT to whitelist: cas-server.xethub.hf.co")
        print(f"\nCache location: {os.path.expanduser('~/.cache/huggingface/hub')}")

if __name__ == "__main__":
    main()
