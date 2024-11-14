#!/usr/bin/env python

#
#   Clair Kronk
#   29 October 2024
#   compare_content.py
#

# Run the first time you run this program:
#import nltk
#nltk.download('punkt_tab')
#nltk.download('stopwords')

from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from scipy.spatial.distance import cosine
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel

import argparse
import textwrap
import torch

stop_words = set(stopwords.words("english"))

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
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        similarity_score = calculate_similarity(file1_content, file2_content, tokenizer, model)
        print("The obtained similarity score between the content in File 1 (%s) and File 2 (%s) is %s." % (str(args.file1), str(args.file2), str(similarity_score)))
        
        print("\n")

        text1_sentences = sent_tokenize(file1_content)
        text2_sentences = sent_tokenize(file2_content)
        similarity_scores = calculate_sentence_similarity(text1_sentences, text2_sentences, tokenizer, model)
        for text1_sentence, similarities in similarity_scores:
            text1_sentence_shortened = textwrap.shorten(text1_sentence, width=20)
            print("File 1 Sentence: '%s'" % str(text1_sentence_shortened))
            for text2_sentence, score in similarities[:3]:
                text2_sentence_shortened = textwrap.shorten(text2_sentence, width=20)
                print("  - File 2 Sentence, '%s': %s" % (str(text2_sentence_shortened), str(score)))

        print("\n")

        missing_content_summary = summarize_missing_content(similarity_scores)
        print("Summary of Missing Content:")
        print("  - File 1 Content Missing in File 2:")
        for keywords, sentence in missing_content_summary["text_1_content_not_in_text_2"]:
            sentence_shortened = textwrap.shorten(sentence, width=20)
            print("    - Sentence (%s), Keywords: %s" % (str(sentence_shortened), str(keywords)))
        print("  - File 2 Content Missing in File 1:")
        for keywords, sentence in missing_content_summary["text_2_content_not_in_text_1"]:
            sentence_shortened = textwrap.shorten(sentence, width=20)
            print("    - Sentence (%s), Keywords: %s" % (str(sentence_shortened), str(keywords)))

        exit()

    else:
        print("One of the files did not load correctly. Exiting...")
        exit()

def get_text_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state.mean(dim=1) # Returns the average of the embeddings for the given sentence.
    return embeddings.detach().numpy()[0]

def calculate_similarity(text1, text2, tokenizer, model):
    embedding1 = get_text_embedding(text1, tokenizer, model)
    embedding2 = get_text_embedding(text2, tokenizer, model)
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity

def calculate_sentence_similarity(text1_sentences, text2_sentences, tokenizer, model):
    similarity_scores = []
    for text1_sentence in text1_sentences:
        text1_embedding = get_text_embedding(text1_sentence, tokenizer, model)
        sentence_scores = []
        for text2_sentence in text2_sentences:
            text2_embedding = get_text_embedding(text2_sentence, tokenizer, model)
            similarity = 1 - cosine(text1_embedding, text2_embedding)
            sentence_scores.append((text2_sentence, similarity))
        similarity_scores.append((text1_sentence, sorted(sentence_scores, key=lambda x: x[1], reverse=True)))
    return similarity_scores

def extract_keywords(text):
    words = word_tokenize(text.lower())
    filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
    return filtered_words

def keyword_overlap(text1, text2):
    blob1 = TextBlob(text1)
    blob2 = TextBlob(text2)

    keywords1 = set(blob1.noun_phrases)
    keywords2 = set(blob2.noun_phrases)
    overlap = keywords1.intersection(keywords2)

    overlap_ratio = len(overlap) / max(len(keywords1), 1)
    print("Keyword Overlap Ratio: %s" % str(overlap_ratio))
    return overlap_ratio

def summarize_missing_content(similarity_scores, threshold=0.5):
    missing_content_summary = {
        "text_1_content_not_in_text_2": [],
        "text_2_content_not_in_text_1": []
    }

    for text_1_sentence, similarities in similarity_scores:
        top_score = similarities[0][1]
        if top_score < threshold:
            keywords = extract_keywords(text_1_sentence)
            missing_content_summary["text_1_content_not_in_text_2"].append((" ".join(keywords), text_1_sentence))

    text_2_sentence_scores = {}
    for _, text_2_sentence_list in similarity_scores:
        for text_2_sentence, score in text_2_sentence_list:
            if text_2_sentence not in text_2_sentence_scores or text_2_sentence_scores[text_2_sentence] < score:
                text_2_sentence_scores[text_2_sentence] = score

    for text_2_sentence, max_score in text_2_sentence_scores.items():
        if max_score < threshold:
            keywords = extract_keywords(text_2_sentence)
            missing_content_summary["text_2_content_not_in_text_1"].append((" ".join(keywords), text_2_sentence))

    return missing_content_summary

if __name__=="__main__": 
    main() 