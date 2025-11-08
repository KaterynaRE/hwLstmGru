import csv
import os
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import glob
from tqdm import tqdm
import librosa
from librosa.util import example_info

def clean_text(text: str) -> str:
    # joki Linux Сі ХIХ
    import re
    text = text.lower()
    # Залишити тільки букви, апостроф і пробіли
    # text = re.sub(r"[^а-щьюяґєіїa-z0-9'՚`’\-\s]", "", text)
    # Видалити символи
    text = re.sub(r"[,.!?:;_́«»“”„…\"]", "", text)
    text = re.sub(r"[`՚’]", "'", text)
    text = re.sub(r"[–—−‒]", "-", text)
    text = re.sub(r"\s-\s", " ", text)
    text = re.sub(r"-\s", " ", text)
    text = re.sub(r"\s-", " ", text)
    text = re.sub(r"joki", "йокі", text)
    text = re.sub(r"linux", "лінукс", text)
    text = re.sub(r"ci", "сі", text)
    text = re.sub(r"xix", "дев’ятнадцяте", text)
    #
    text = re.sub(r"a", "а", text)
    text = re.sub(r"c", "с", text)
    text = re.sub(r"e", "е", text)
    text = re.sub(r"i", "і", text)
    text = re.sub(r"m", "м", text)
    text = re.sub(r"o", "о", text)
    text = re.sub(r"p", "р", text)
    text = re.sub(r"x", "х", text)
    text = re.sub(r"y", "у", text)
    text = re.sub(r"ы", "и", text)
    # Прибрати зайві пробіли
    text = re.sub(r"\s+", " ", text).strip()
    return text

#Обробка аудіо і текстів
def load_tsv(tsv_path: str, data_path: str, example_count: int) -> DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep='\t',
        engine="python",
        quoting=3,  # csv.QUOTE_NONE = 3
    )
    #Беремо рядки з validated.tsv які містять тільки path до існуючого файлу
    df = df[df.apply(lambda row: os.path.exists(os.path.join(data_path, row.path)), axis=1)]
    #Беремо певну кількість (example_count) перших рядків
    df = df.head(example_count)
    df['sentence'] = df['sentence'].apply(clean_text)
    return df

def load_tsv_full(tsv_path: str):
    df = pd.read_csv(tsv_path, sep='\t')
    df['sentence'] = df['sentence'].apply(clean_text)
    return df

def load_symbols(df:DataFrame):
    unique_symbols = set()
    for line in df.sentence:
        for symbol in line:
            unique_symbols.add(symbol)
    alphabet = sorted(list(unique_symbols))
    char2idx = {c: i+1 for i, c in enumerate(alphabet)} #0 = blank \0
    idx2char = {i+1 : c for i, c in enumerate(alphabet)}
    num_classes =  len(alphabet) + 1
    return alphabet, char2idx, idx2char, num_classes

def optimal_length_df(data_path: str, df: DataFrame, num_features: int):
    files = glob.glob(os.path.join(data_path, "*.mp3"))
    samples = random.sample(files, min(EXAMPLE_COUNT, len(files)))
    lengths = []
    for path in samples:
        y, sr = librosa.load(path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T
        lengths.append(len(mfcc))
    percen = np.percentile(lengths, [50, 75, 90, 95, 99, 100])
    return int(max(percen))

def extract_features(path: str, num_features: int, input_length: int):
    y, sr = librosa.load(path, sr=16000)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=num_features).T
    if len(mfcc) < input_length:
        mfcc = np.pad(mfcc, ((0, input_length - len(mfcc)), (0, 0)), mode='constant')
    else:
        mfcc = mfcc[:input_length, :]
    return mfcc

def encode_text(text: str, char2idx: dict):
    text = text.lower().strip()
    return [char2idx[c] for c in text if c in char2idx]

def fill_data(data_path: str, df: DataFrame, num_features: int, input_length: int, char2idx: dict):
    X, Y, input_lengths, label_lengths = [], [], [], []
    for i, row in tqdm(df.iterrows(), total=len(df)):
        file_path = os.path.join(data_path, row.path)
        if not os.path.exists(file_path):
            continue
        mfcc = extract_features(file_path, num_features, input_length)
        label = encode_text(row.sentence, char2idx)
        if len(label) == 0:
            continue
        X.append(mfcc)
        Y.append(label)
        input_lengths.append(len(mfcc))
        label_lengths.append(len(label))
    # Вирівнювання
    max_label_len = np.max(label_lengths)
    Y_padded = np.zeros((len(Y), max_label_len))
    for i in range(len(Y)):
        Y_padded[i, :len(Y[i])] = Y[i]
    X = np.array(X)
    input_lengths = np.array(input_lengths)
    label_lengths = np.array(label_lengths)
    return X, Y, Y_padded, input_lengths, label_lengths


###############
# Шляхи
TSV_PATH = "content/uk/validated.tsv"
DATA_PATH = "content/uk/clips"
# Змінні
EXAMPLE_COUNT = 300 # Для швидкої перевірки
NUM_FEATURES = 40 # 13-40 кількість MFCC коефіцієнтів

DF = load_tsv(TSV_PATH, DATA_PATH, EXAMPLE_COUNT )
alphabet, char2idx, idx2char, num_classes = (
    load_symbols(load_tsv_full(TSV_PATH))
)
print(alphabet)

input_length = optimal_length_df(DATA_PATH, DF, NUM_FEATURES)
print(input_length)
X, Y, Y_padded, input_lengths, label_lengths = (
    fill_data(DATA_PATH, DF, NUM_FEATURES, input_length, char2idx)
)