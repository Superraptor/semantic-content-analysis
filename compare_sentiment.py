#!/usr/bin/env python

#
#   Clair Kronk
#   29 October 2024
#   compare_sentiment.py
#

from transformers import pipeline

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
        compare_sentiments(text_1_sentiments, text_2_sentiments)

    else:
        print("One of the files did not load correctly. Exiting...")
        exit()

def perform_sentiment_analysis(sentences):
    sentiment_analyzer = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
    sentiments = []
    for sentence in sentences:
        result = sentiment_analyzer(sentence)[0]
        score = int(result['label'].split()[0])
        sentiments.append((sentence, score))
    return sentiments

def compare_sentiment(text_1_sentiments, text_2_sentiments):
    total_text_1_sentiment = sum([])
    total_text_2_sentiment = sum([])

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

if __name__=="__main__": 
    main() 