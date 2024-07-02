# This function can be used to preprocess the xml files into txt files. 
# It deletes all implicit targets and it splits sentences that have multiple targets into the same sentence with each target separate

# Based on HAABSA code Hollander and Wallaart

import os
import xml.etree.ElementTree as ET
from collections import Counter
import re
import nltk
import en_core_web_sm
n_nlp = en_core_web_sm.load()
import numpy as np

# Ensure nltk data is available
nltk.download('punkt')

def window(iterable, size):
    """Stack overflow solution for sliding window"""
    i = iter(iterable)
    win = []
    for e in range(0, size):
        win.append(next(i))
    yield win
    for e in i:
        win = win[1:] + [e]
        yield win

def _get_data_tuple(sptoks, asp_termIn, label):
    """Find the ids of aspect term"""
    aspect_is = []
    asp_term = ' '.join(sp for sp in asp_termIn).lower()
    for _i, group in enumerate(window(sptoks, len(asp_termIn))):
        if asp_term == ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break
        elif asp_term in ' '.join([g.lower() for g in group]):
            aspect_is = list(range(_i, _i + len(asp_termIn)))
            break

    print(aspect_is)
    pos_info = []
    for _i, sptok in enumerate(sptoks):
        pos_info.append(min([abs(_i - i) for i in aspect_is]))

    lab = None
    if label == 'negative':
        lab = -1
    elif label == 'neutral':
        lab = 0
    elif label == "positive":
        lab = 1
    else:
        raise ValueError("Unknown label: %s" % lab)

    return pos_info, lab

def read_data_2016(fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name):
    """This function reads data from the xml file"""
    if not os.path.isfile(fname):
        raise FileNotFoundError(f"[!] Data {fname} not found")

    # Parse xml file to tree
    tree = ET.parse(fname)
    root = tree.getroot()

    processed_data_folder = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/'
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    outF = open(os.path.join(processed_data_folder, file_name), "w")

    # Initialize variables
    source_words, target_words, max_sent_len, max_target_len = [], [], 0, 0
    target_phrases = []

    countConfl = 0
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        for sp in sptoks:
            source_words.append(sp.lower())
        if len(sptoks) > max_sent_len:
            max_sent_len = len(sptoks)
        for opinions in sentence.iter('Opinions'):
            for opinion in opinions.findall('Opinion'):
                if opinion.get("polarity") == "conflict":
                    countConfl += 1
                    continue
                asp = opinion.get('target')
                if asp != 'NULL':
                    aspNew = re.sub(' +', ' ', asp)
                    t_sptoks = nltk.word_tokenize(aspNew)
                    for sp in t_sptoks:
                        target_words.append(sp.lower())
                    target_phrases.append(' '.join(t_sptoks).lower())
                    if len(t_sptoks) > max_target_len:
                        max_target_len = len(t_sptoks)
    if len(source_count) == 0:
        source_count.append(['<pad>', 0])
    source_count.extend(Counter(source_words + target_words).most_common())
    target_count.extend(Counter(target_phrases).most_common())

    for word, _ in source_count:
        if word not in source_word2idx:
            source_word2idx[word] = len(source_word2idx)

    for phrase, _ in target_count:
        if phrase not in target_phrase2idx:
            target_phrase2idx[phrase] = len(target_phrase2idx)

    source_data, source_loc_data, target_data, target_label = [], [], [], []

    # Collect output data and write to .txt file
    for sentence in root.iter('sentence'):
        sent = sentence.find('text').text
        sentenceNew = re.sub(' +', ' ', sent)
        sptoks = nltk.word_tokenize(sentenceNew)
        if sptoks:
            idx = [source_word2idx[sp.lower()] for sp in sptoks]
            for opinions in sentence.iter('Opinions'):
                for opinion in opinions.findall('Opinion'):
                    if opinion.get("polarity") == "conflict":
                        continue
                    asp = opinion.get('target')
                    if asp != 'NULL':  # Removes implicit targets
                        aspNew = re.sub(' +', ' ', asp)
                        t_sptoks = nltk.word_tokenize(aspNew)
                        source_data.append(idx)
                        outputtext = ' '.join(sptoks).lower().replace(' '.join(t_sptoks).lower(), '$T$')
                        outF.write(outputtext + "\n")
                        outF.write(' '.join(t_sptoks).lower() + "\n")
                        pos_info, lab = _get_data_tuple(sptoks, t_sptoks, opinion.get('polarity'))
                        pos_info = [(1 - (i / len(idx))) for i in pos_info]
                        source_loc_data.append(pos_info)
                        targetdata = ' '.join(t_sptoks).lower()
                        target_data.append(target_phrase2idx[targetdata])
                        target_label.append(lab)
                        outF.write(str(lab) + "\n")

    outF.close()
    print(f"Read {len(source_data)} aspects from {fname}")
    print(countConfl)
    return source_data, source_loc_data, target_data, target_label, max_sent_len, max_target_len

def main():
    # Define input arguments
    year = '15'  # Change to '15' or '16' as needed
    data_type = 'Test'  # Change to 'Train' or 'Test' as needed

    fname = f'/content/drive/MyDrive/Thesis/DataAugmentation/data/externalData/ABSA{year}_Restaurants_{data_type}.xml'
    source_count = []
    source_word2idx = {}
    target_count = []
    target_phrase2idx = {}
    file_name = f'data_{year}_{data_type.lower()}.txt'

    processed_data_folder = '/content/drive/MyDrive/Thesis/DataAugmentation/data/processedData/'

    # Call the function
    source_data, source_loc_data, target_data, target_label, max_sent_len, max_target_len = read_data_2016(
        fname, source_count, source_word2idx, target_count, target_phrase2idx, file_name
    )

    # Print the outputs
    print("Source Data:", source_data)
    print("Source Location Data:", source_loc_data)
    print("Target Data:", target_data)
    print("Target Label:", target_label)
    print("Max Sentence Length:", max_sent_len)
    print("Max Target Length:", max_target_len)

if __name__ == "__main__":
    main()
