#!/usr/bin/env python

#
#   Clair Kronk
#   29 October 2024
#   compare_sentiment.py
#

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

import argparse
import textwrap
import torch

def main():

    parser = argparse.ArgumentParser(description="Include two text files to compare semantic content.")
    parser.add_argument('file1', type=str, help='Path to the first text file.')
    parser.add_argument('file2', type=str, help='Path to the second text file.')

    args = parser.parse_args()

    file1_content = None
    file2_content = None
    with open(str(args.file1), 'r') as f1, open(str(args.file2), 'r') as f2:
        file1_content = f1.read()
        file2_content = f2.read()

    if (file1_content is not None) and (file2_content is not None):
        text_1_sentiments = perform_sentiment_analysis(file1_content)
        text_2_sentiments = perform_sentiment_analysis(file2_content)
        compare_sentiment(text_1_sentiments, text_2_sentiments)

    else:
        print("One of the files did not load correctly. Exiting...")
        exit()

def perform_sentiment_analysis(sentences):
    """
    Perform sentiment analysis on sentences using BERT model.

    Attempts to load the model from local cache first (if previously downloaded),
    then falls back to downloading if cache load fails.

    For texts longer than 512 tokens (BERT's limit), splits into chunks and
    averages the sentiment scores.

    Args:
        sentences: List of sentences to analyze

    Returns:
        List of tuples: (sentence, sentiment_score) where score is 1-5
    """
    try:
        # Attempt to load model from local HuggingFace cache
        print("Loading sentiment model from cache...")
        tokenizer = AutoTokenizer.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            "nlptown/bert-base-multilingual-uncased-sentiment",
            local_files_only=True
        )
        print("SUCCESS: Loaded sentiment model from cache")
    except Exception as e:
        # Cache load failed, attempt to download the model
        print(f"Warning: Could not load from cache ({e}), attempting download...")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
            print("SUCCESS: Downloaded and loaded sentiment model")
        except Exception as e2:
            print(f"ERROR: Could not load sentiment model: {e2}")
            raise

    # Process each sentence through the model
    sentiments = []
    model.eval()  # Set model to evaluation mode

    for sentence in sentences:
        # Tokenize to check length
        tokens = tokenizer(sentence, return_tensors="pt", truncation=False)

        # Check if text exceeds BERT's 512 token limit
        if tokens['input_ids'].shape[1] > 512:
            # Split into chunks of 500 tokens (leaving room for special tokens)
            chunk_size = 500
            all_scores = []

            # Process text in overlapping chunks
            words = sentence.split()
            chunk_texts = []
            for i in range(0, len(words), chunk_size):
                chunk_text = ' '.join(words[i:i+chunk_size])
                chunk_texts.append(chunk_text)

            # Get sentiment for each chunk
            for chunk_text in chunk_texts:
                with torch.no_grad():
                    inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    chunk_score = torch.argmax(probs, dim=1).item() + 1  # Convert 0-4 to 1-5
                    all_scores.append(chunk_score)

            # Average scores from all chunks
            score = sum(all_scores) / len(all_scores)
            score = int(round(score))  # Round to nearest integer (1-5)
        else:
            # Text fits within limit, process normally
            with torch.no_grad():
                inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=512)
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                score = torch.argmax(probs, dim=1).item() + 1  # Convert 0-4 to 1-5

        sentiments.append((sentence, score))

    return sentiments

def compare_sentiment(text_1_sentiments, text_2_sentiments):
    # Fixed: Actually calculate the average sentiment scores
    total_text_1_sentiment = sum([score for _, score in text_1_sentiments]) / len(text_1_sentiments) if text_1_sentiments else 0
    total_text_2_sentiment = sum([score for _, score in text_2_sentiments]) / len(text_2_sentiments) if text_2_sentiments else 0

    print("Overall Sentiment Comparison:")
    print("- File 1 Sentiment Score (Average): %s" % (total_text_1_sentiment))
    print("- File 2 Sentiment Score (Average): %s" % (total_text_2_sentiment))
    print("- Difference in Sentiment: %s" % (str(abs(total_text_1_sentiment - total_text_2_sentiment))))

    print("\n")

    print("Sentence-Level Sentiment Differences (File 1 vs. File 2):")
    for (text_1_sentence, text_1_score), (text_2_sentence, text_2_score) in zip(text_1_sentiments, text_2_sentiments):
        if abs(text_1_score - text_2_score) >= 2:
            shortened_text_1_sentence = textwrap.shorten(text_1_sentence, width=20)
            shortened_text_2_sentence = textwrap.shorten(text_2_sentence, width=20)
            print("- Text 1 Sentence (%s), Sentiment Score: %s" % (str(shortened_text_1_sentence), str(text_1_score)))
            print("- Text 2 Sentence (%s), Sentiment Score: %s" % (str(shortened_text_2_sentence), str(text_2_score)))
            print("- Sentiment Difference: %s" % (str(abs(text_1_score - text_2_score))))
            print("\n")

def analyze_sentiment_texts(text1, text2):
    """
    Reusable function to analyze sentiment of two texts using BERT model.

    This function is imported by OFFLINE_perform_paralinguistic_analysis.py
    as analyze_sentiment_hf. It attempts to load the model from local cache first,
    then falls back to downloading if needed.

    Handles long texts (>512 tokens) by chunking into 500-word segments and
    averaging the sentiment scores across chunks.

    Args:
        text1 (str): First text to analyze
        text2 (str): Second text to analyze

    Returns:
        dict: Dictionary containing:
            - text1_sentiment: Average sentiment score for text1 (1-5 scale)
            - text2_sentiment: Average sentiment score for text2 (1-5 scale)
            - sentiment_difference: Absolute difference between scores
    """
    try:
        # Attempt to load model from local HuggingFace cache
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment",
                local_files_only=True
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment",
                local_files_only=True
            )
        except Exception:
            # Cache load failed, attempt to download the model
            tokenizer = AutoTokenizer.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                "nlptown/bert-base-multilingual-uncased-sentiment"
            )

        model.eval()

        def get_sentiment_score(text):
            """Helper function to get sentiment score for a single text"""
            # Tokenize to check if text exceeds BERT's 512 token limit
            tokens = tokenizer(text, return_tensors="pt", truncation=False)

            if tokens['input_ids'].shape[1] > 512:
                # Text is too long, split into 500-word chunks
                words = text.split()
                chunk_size = 500
                all_scores = []

                # Process each chunk separately
                for i in range(0, len(words), chunk_size):
                    chunk_text = ' '.join(words[i:i+chunk_size])
                    with torch.no_grad():
                        inputs = tokenizer(chunk_text, return_tensors="pt", truncation=True, max_length=512)
                        outputs = model(**inputs)
                        probs = F.softmax(outputs.logits, dim=1)
                        # Convert 0-4 index to 1-5 score
                        chunk_score = torch.argmax(probs, dim=1).item() + 1
                        all_scores.append(chunk_score)

                # Return average score across all chunks
                return sum(all_scores) / len(all_scores)
            else:
                # Text fits within limit, process normally
                with torch.no_grad():
                    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
                    outputs = model(**inputs)
                    probs = F.softmax(outputs.logits, dim=1)
                    # Convert 0-4 index to 1-5 score
                    return torch.argmax(probs, dim=1).item() + 1

        # Get sentiment scores for both texts
        avg_sentiment1 = get_sentiment_score(text1)
        avg_sentiment2 = get_sentiment_score(text2)

        return {
            'text1_sentiment': avg_sentiment1,
            'text2_sentiment': avg_sentiment2,
            'sentiment_difference': abs(avg_sentiment1 - avg_sentiment2)
        }

    except Exception as e:
        print(f"Warning: Sentiment analysis failed: {e}")
        # Return neutral scores on failure (fallback will be handled by caller)
        raise  # Re-raise to let OFFLINE script handle fallback to analyze_sentiment_offline

if __name__=="__main__":
    main() 