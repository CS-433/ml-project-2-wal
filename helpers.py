# -*- coding: utf-8 -*-
"""some helper functions."""

import os
import re
import json
import nltk
import pandas as pd


##################### - processing.ipynb - #####################

### ---- Cleaning and Pre-processing for German Language ---- ### 

from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
stemmer = SnowballStemmer("german")
stop_words = set(stopwords.words("german"))

def clean_text(text, for_embedding=False):
    """
        - all lowercase
        - remove single letter chars
        - remove stopwords, punctuation and stemming
        - keep only ASCII, european chars and whitespace, no digits
        - convert all whitespaces to single whitespace (if not for embedding)
    """
    RE_WSPACE = re.compile(r"\s+", re.IGNORECASE)
    RE_ASCII = re.compile(r"[^A-Za-zÀ-ž ]", re.IGNORECASE)
    RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž]\b", re.IGNORECASE)
    if for_embedding:
        # Keep punctuation
        RE_ASCII = re.compile(r"[^A-Za-zÀ-ž,.!? ]", re.IGNORECASE)
        RE_SINGLECHAR = re.compile(r"\b[A-Za-zÀ-ž,.!?]\b", re.IGNORECASE)

    text = re.sub(RE_ASCII, " ", text)
    text = re.sub(RE_SINGLECHAR, " ", text)
    text = re.sub(RE_WSPACE, " ", text)

    word_tokens = word_tokenize(text)
    words_tokens_lower = [word.lower() for word in word_tokens]

    if for_embedding:
        words_filtered = word_tokens
    else:
        words_filtered = [
            stemmer.stem(word) for word in words_tokens_lower if word not in stop_words
        ]

    text_clean = " ".join(words_filtered)
    return text_clean

### ---- Extracting the transcript with optional filtering based on the minimum score ---- ### 

def extract_transcript_with_scores(result, min_score=None):
    """Extracting the transcript with optional filtering based on the minimum score"""
    transcript = ""
    words_with_scores = []

    for segment in result['segments']:
        for word_info in segment['words']:
            word = word_info['word']

            # Check if 'score' key exists in word_info
            if 'score' in word_info:
                score = word_info['score']

                # Only include words with a score greater than the specified minimum score
                if min_score is None or (min_score is not None and score > min_score):
                    transcript += word + ' '
                    words_with_scores.append({'word': word, 'score': score})

    return transcript.strip(), words_with_scores

### ---- Extracting the transcript from the segment---- ### 

def extract_transcript_from_segments(result, include_all_words=False):
    """Extracting the transcript from the segment"""
    transcript = ""
    for word_info in result.get('words', []):
        word = word_info.get('word')

        # Check if 'speaker', 'score', and 'start' keys exist in word_info
        if all(key in word_info for key in ['speaker', 'score', 'start']):
            speaker = word_info['speaker']
            score = word_info['score']

            # Only include words with a score greater than 0.5 if include_all_words is False
            if include_all_words or (not include_all_words and score > 0.5):
                transcript += word + ' '

    return transcript.strip()

### ---- Extracting speaker information from the result ---- ### 

def extract_speaker_info(result, file_name):
    """Extracting speaker information from the result"""
    speaker_info = []

    # Check if 'segments' key exists in the result
    if 'segments' in result:
        current_speaker_info = None

        for segment in result['segments']:
            speaker_id = segment.get('speaker', 'UNKNOWN')
            speaker_initial_transcript = extract_transcript_from_segments(segment, include_all_words=True)
            speaker_filtered_transcript = extract_transcript_from_segments(segment)

            # Skip segments without valid words
            if not speaker_filtered_transcript:
                continue

            # If the speaker changes, start a new speaker_info entry
            if current_speaker_info is None or current_speaker_info['speaker_id'] != speaker_id:
                if current_speaker_info is not None:
                    # Clean the final transcript and add to speaker_info
                    current_speaker_info['speaker_clean_final_transcript'] = clean_text(
                        current_speaker_info['speaker_filtered_transcript']
                    )

                    # Append the current_speaker_info to speaker_info list
                    speaker_info.append(current_speaker_info)

                current_speaker_info = {
                    'team_id': file_name,  # Use file_name as 'team_id'
                    'speaker_id': speaker_id,
                    'speaker_initial_transcript': '',
                    'speaker_filtered_transcript': '',
                    'speaker_clean_final_transcript': ''
                }

            # Append transcripts to the respective fields
            current_speaker_info['speaker_initial_transcript'] += speaker_initial_transcript + ' '
            current_speaker_info['speaker_filtered_transcript'] += speaker_filtered_transcript + ' '

        # Clean the final transcript and add the last current_speaker_info to speaker_info
        if current_speaker_info is not None:
            current_speaker_info['speaker_clean_final_transcript'] = clean_text(
                current_speaker_info['speaker_filtered_transcript']
            )
            speaker_info.append(current_speaker_info)

    return speaker_info


##################### - main.ipynb - #####################

### ---- Calculate and print diarization metrics including missed speakers rate, false alarm rate, and SDER. ---- ### 

def diarization_metrics(diarized_speakers, true_speakers, margin):
    """
    Calculate and print diarization metrics including missed speakers rate, false alarm rate, and SDER.

    Parameters:
    - speakers_count: DataFrame
        DataFrame containing information about the number of speakers for each team.
    - margin: int, default=1
        Margin for missed and false alarms.
    """
    # Calculate missed speakers
    missed_speakers = ((diarized_speakers + margin) < true_speakers).sum()

    # Calculate false alarms
    false_alarms = ((diarized_speakers - margin) > true_speakers).sum()

    # Calculate total ground truth speakers
    total_ground_truth_speakers = true_speakers.sum()

    # Calculate missed speakers rate
    missed_speakers_rate = missed_speakers / total_ground_truth_speakers if total_ground_truth_speakers > 0 else 0

    # Calculate false alarm rate
    false_alarm_rate = false_alarms / total_ground_truth_speakers if total_ground_truth_speakers > 0 else 0

    # Calculate speaker diarization error rate (SDER)
    sder = (missed_speakers + false_alarms) / total_ground_truth_speakers if total_ground_truth_speakers > 0 else 0

    # Print the results
    print(f"Missed Speakers Rate: {missed_speakers_rate:.4f}")
    print(f"False Alarm Rate: {false_alarm_rate:.4f}")
    print(f"Speaker Diarization Error Rate (SDER): {sder:.4f}")

### ---- Rttm files processing function (pyannote) ---- ### 

def process_rttm_files(pyannote_folder_path):
    # Create an empty DataFrame to store the results
    rttm_data = pd.DataFrame(columns=['team_id', 'num_speakers_pyannote'])

    # Iterate through each RTTM file
    for rttm_file in os.listdir(pyannote_folder_path):
        if rttm_file.endswith('.rttm'):
            # Extract team_id from the file name (remove '.rttm' extension)
            team_id = os.path.splitext(rttm_file)[0]

            # Read the RTTM file and count the number of unique speakers
            unique_speakers = set()  # Create an empty set for each file
            with open(os.path.join(pyannote_folder_path, rttm_file), 'r') as file:
                lines = file.readlines()
                for line in lines:
                    if line.startswith("SPEAKER"):
                        # Use regex to extract the speaker ID
                        match = re.search(r'SPEAKER_\d+', line)
                        if match:
                            speaker_id = match.group()

                            # Add the speaker ID to the set of unique speakers
                            unique_speakers.add(speaker_id)

            # Count the number of unique speakers
            num_unique_speakers = len(unique_speakers)

            # Append the results to the DataFrame
            rttm_data = pd.concat([rttm_data, pd.DataFrame({'team_id': [team_id], 'num_speakers_pyannote': [num_unique_speakers]})], ignore_index=True)

    return rttm_data

### ---- Json files processing function (deepgram) ---- ### 

def process_deepgram_files(deepgram_folder_path):
    # Create an empty DataFrame to store the results
    deepgram_data = pd.DataFrame(columns=['team_id', 'num_speakers_deepgram'])

    # Iterate through each JSON file
    for deepgram_file in os.listdir(deepgram_folder_path):
        if deepgram_file.endswith('.json'):
            # Extract team_id from the file name (remove '.json' extension)
            team_id = os.path.splitext(deepgram_file)[0]

            # Read the JSON file
            with open(os.path.join(deepgram_folder_path, deepgram_file), 'r') as file:
                data = json.load(file)

            # Extract unique speaker identifiers from the transcript
            speakers = set(word['speaker'] for alternative in data['results']['channels'][0]['alternatives'] for word in alternative['words'])

            # Count the number of unique speakers
            num_speakers = len(speakers)

            # Append the results to the DataFrame
            deepgram_data = pd.concat([deepgram_data, pd.DataFrame({'team_id': [team_id], 'num_speakers_deepgram': [num_speakers]})], ignore_index=True)

    return deepgram_data

### ---- This function is used to extract the continue speaking time for each speaker ---- ### 

def extract_speaking_time(data):
    """
    This function is used to extract the continue speaking time for each speaker.
    Args:
        data: speaking time and speaking content of each speaker (dictionary)
    Returns:
        s_cont: the length of continuous speaking time of each speaker (pd.series)
        s: the total length of speaking time of each speaker (pd.series)
    """

    # transfer json to dataframe
    data_aux = pd.DataFrame(data['segments'])

    # build up the distribution of continuous speaking time
    data_aux['length'] = data_aux['end'] - data_aux['start']
    s_cont = data_aux[['length','speaker']]

    # length of the segments
    #s = data_aux.groupby('speaker').sum()[['length']]

    return s_cont

### ---- This function is used to calculate the dominance score for each speaker ---- ### 

def dominance(data):
    """
    calculate dominance score
    """

    speaker_num = len(data['speaker'].unique())

    # Group by 'speaker' and calculate the sum of 'length' for each group
    speaker_sum_length = data.groupby('speaker')['length'].sum()

    # Sort the results in descending order
    sorted_speaker_sum_length = speaker_sum_length.sort_values(ascending=False)

    t1 = sorted_speaker_sum_length[0]
    t2 = sorted_speaker_sum_length[1:].sum()/len(sorted_speaker_sum_length[1:])
    
    # calculate the difference
    difference = t1 - t2

    return difference

### ---- Min-Max Normalization ---- ### 

def normalize(data):
    """
    normalize the data using min-max 
    """
    return (data - data.min()) /(data.max()-data.min())

