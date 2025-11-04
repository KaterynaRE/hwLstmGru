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
def load_tsv(tsv_path: str, example_count: int) -> DataFrame:
    df = pd.read_csv(
        tsv_path,
        sep='\t',
        engine="python",
        quoting=3
    )
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
    num_classes =  len(alphabet)
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


###############
# Шляхи
TSV_PATH = "content/uk/validated.tsv"
DATA_PATH = "content/uk/clips"
# Змінні
EXAMPLE_COUNT = 300 # Для швидкої перевірки
NUM_FEATURES = 40 # 13-40 кількість MFCC коефічієнтів

DF = load_tsv(TSV_PATH, EXAMPLE_COUNT)
alphabet, char2idx, idx2char, num_classes = (
 load_symbols(load_tsv_full(TSV_PATH))
)
input_length = optimal_length_df(DATA_PATH, DF, NUM_FEATURES)
print(input_length)