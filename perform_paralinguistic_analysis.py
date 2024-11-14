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

from compare_content import keyword_overlap
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
import torch
import wave

print('Loading emotion model...')
emotion_model = Wav2Vec2ForSequenceClassification.from_pretrained("superb/wav2vec2-large-superb-er")
print('Loading emotion processor...')
emotion_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
print('Loading speaker diarization pipeline...')
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization", use_auth_token="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")
print('Running script...')

def main():

    parser = argparse.ArgumentParser(description="Include an audio file in order to provide an analysis of the paralinguistic content.")
    parser.add_argument('audiofile', type=str, help='Path to the audio file.')

    args = parser.parse_args()

    speaker_analysis_results = analyze_speakers(args.audiofile)
    speaker_segments = diarize_speakers(args.audiofile)

    emotions = {}
    for speaker, speaker_dict in speaker_analysis_results.items():
        emotions[speaker] = speaker_dict["emotion"]

    analyze_comprehension(args.audiofile, speaker_segments, emotions)

def transcribe_audio(audio_file, model_path="vosk-model-en-us-0.22"):
    model = Model(model_path)
    
    wf = wave.open(audio_file, "rb")
    if wb.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
        raise ValueError("Audio file must be mono PCM WAV format. Exiting...")

    recognizer = KaldiRecognizer(model, wf.getframerate())

    transcription = ""
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if recognizer.AcceptWaveform(data):
            result = json.loads(recognizer.Result())
            transcription += result.get("text", "") + " "

    final_result = json.loads(recognizer.FinalResult())
    transcription += final_result.get("text", "")

    wf.close()
    print("Transcription: %s" % (str(transcription)))
    return transcription

def diarize_speakers(audio_file):
    diarization = pipeline({"uri": "sample", "audio": audio_file})
    speaker_segments = {}

    for turn, _, speaker in diarization.itertracks(yield_label=True):
        if speaker not in speaker_segments:
            speaker_segments[speaker] = []
        speaker_segments[speaker].append((turn.start, turn.end))

    return speaker_segments

def extract_speaker_segments(audio_file, speaker_segments):

    speaker_file_test = "%s_speaker_%s.wav" % (str(Path(audio_file).stem), str("SPEAKER_00"))
    generate = True
    if os.path.isfile(speaker_file_test):
        user_input = input("One or more speaker segment files appear to already exist. Should they be regenerated? (y/N): ")
        user_input = (str(user_input).strip()).lower()
        if user_input in ['n', 'false', 'f', '', 'no']:
            generate = False

    speaker_audio_files = {}

    if generate:
        audio = AudioSegment.from_wav(audio_file)

        for speaker, segments in speaker_segments.items():
            combined_segment = AudioSegment.empty()

            for start, end in segments:
                start_ms = start * 1000
                end_ms = end * 1000
                combined_segment += audio[start_ms:end_ms]

            speaker_file = "%s_speaker_%s.wav" % (str(Path(audio_file).stem), str(speaker))
            combined_segment.export(speaker_file, format="wav")
            speaker_audio_files[speaker] = speaker_file

    else:
        speaker_audio_file_list = glob.glob("%s_speaker_SPEAKER_*.wav" % (str(Path(audio_file).stem)))
        for speaker_audio_file_path in speaker_audio_file_list:
            speaker_file_base_name = os.path.basename(speaker_audio_file_path)
            try:
                speaker = re.search("%s_speaker_SPEAKER_(.+?).wav" % (str(Path(audio_file).stem)), str(speaker_file_base_name)).group(1)
            except AttributeError:
                print("Speaker not found for file (%s). Exiting..." % str(speaker_file_base_name))
                exit()
            speaker_audio_files[speaker] = speaker_audio_file_path

    return speaker_audio_files

def analyze_speakers(audio_file):
    speaker_segments = diarize_speakers(audio_file)
    speaker_audio_files = extract_speaker_segments(audio_file, speaker_segments)

    results = {}
    for speaker, speaker_file in speaker_audio_files.items():
        print("Analyzing Speaker %s" % (str(speaker)))

        features = extract_audio_features(speaker_file)
        predicted_emotion = analyze_emotion(speaker_file)
        personality_traits = analyze_personality(features)

        results[speaker] = {
            "audio_features": features,
            "emotion": predicted_emotion,
            "personality": personality_traits
        }

        print("Audio features: %s" % (str(features)))
        print("Predicted emotion: %s" % (str(predicted_emotion)))
        print("Predicted personality traits:")
        for trait, level in personality_traits.items():
            print("  - %s: %s" % (str(trait), str(level)))

    return results

def extract_audio_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    features = {}

    # Pitch, estimated using YIN pitch estimation.
    pitches, magnitude = librosa.piptrack(y=y, sr=sr)
    pitches = pitches[pitches > 0] # Filter out zero values.
    features["pitch_mean"] = np.mean(pitches)
    features["pitch_std"] = np.std(pitches)

    # Intensity
    features["intensity_mean"] = np.mean(librosa.feature.rms(y=y))
    features["intensity_std"] = np.std(librosa.feature.rms(y=y))

    # Tempo
    tempo, _= librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = tempo

    # MFCCs (Mel-frequency cepstral coefficients, used for tone analysis)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    features["mfcc_mean"] = np.mean(mfccs)
    features["mfcc_std"] = np.std(mfccs)

    return features

def analyze_emotion(audio_file):

    y, sr = librosa.load(audio_file, sr=16000)
    input_values = emotion_processor(y, sampling_rate=sr, return_tensors="pt").input_values
    with torch.no_grad():
        logits = emotion_model(input_values).logits
    emotion_prediction = torch.argmax(logits, dim=-1).item()

    emotions = {
        0: "Neutral",
        1: "Calm",
        2: "Happy",
        3: "Sad",
        4: "Angry",
        5: "Fearful",
        6: "Disgust",
        7: "Surprised"
    }

    predicted_emotion = emotions.get(emotion_prediction, "Unknown")
    return predicted_emotion

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
            alignment_Scores.append(alignment_score)

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

def analyze_comprehension(audio_file, speaker_segments, emotions):
    avg_response_time = analyze_turn_taking(speaker_segments)
    alignment_ratio = analyze_emotion_alignment(emotions)
    backchannels = detect_backchannels(audio_file, speaker_segments)
    backchannel_count = len([b for b in backchannels if b[0] == "SPEAKER_01"])

    transcriptions = {}
    for speaker, segments in speaker_segments.items():
        speaker_audio = "%s_speaker_%s.wav" % (str(audio_file.stem), str(speaker))
        transcriptions[speaker] = transcribe_audio(speaker_audio)
    overlap_ratio = keyword_overlap(speaker_1_text["SPEAKER_00"], speaker_2_text["SPEAKER_01"])

    comprehension_score = (
        (alignment_ratio or 0) * 0.3 + 
        (1 / (1 + avg_response_time or 1)) * 0.3 + 
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
    print("Comprehension Summary: %s", str(comprehension_summary))
    return comprehension_summary

if __name__=="__main__": 
    main() 