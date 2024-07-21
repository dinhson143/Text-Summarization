from datetime import datetime

import pandas as pd
import numpy as np
import nltk
import re
import torch

from torch import nn
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

ROOT = "E:/DATN_Thacsi/Code_Kmeans/models/"
ROOT_DATA = "E:/DATN_Thacsi/data/"
CURRENT_DATE = datetime.today().date()


class ReduceDimensionality(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ReduceDimensionality, self).__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, features):
        features.update({'sentence_embedding': self.dense(features['sentence_embedding'])})
        return features

def reading_data_from_files() -> (pd.DataFrame, pd.DataFrame):
    print('1. Reading data from csv files......')
    try:
        data_train = pd.read_csv(f'{ROOT_DATA}data_train.csv', encoding="utf-8")
        data_test = pd.read_csv(f'{ROOT_DATA}test_data.csv', encoding="utf-8")
        print('==> Reading data successfully......')
        return data_train, data_test
    except Exception as e:
        print(f'Reading data failed {e}')
        raise

def check_data(data_train: pd.DataFrame, data_test: pd.DataFrame):
    print('2. Shape train and test data ......')
    print("The number of row and column of data_train:", data_train.shape)
    print("The number of row and column of data_test:", data_test.shape)
    print('==> Shape train and test data successfully......')


def create_tokens_from_Sentences(data: pd.DataFrame) -> list:
    print('4. Sentences tokenization.....Create token')

    index = []
    paras = []
    reg = "[^\w\s]"

    print('==> Sentences tokenization => data_train, test.....')
    for i in data.index:
        normalized_text = data.loc[i, 'original'].lower().replace('\n', ' ').strip()
        data.loc[i, 'original'] = normalized_text

        # Tokenize token
        cleaned_sentence = None
        sentences = nltk.sent_tokenize(normalized_text)
        for sentence in sentences:
            cleaned_sentence = re.sub(reg, '', sentence).replace('   ', ' ').replace('  ', ' ').strip()
            paras.append(cleaned_sentence)
        # print(i)
        index.append(i)
        # break
    return paras


def convert_sentence_to_vector(paras: list, embeddings_index: dict) -> list:
    print('5.Sentences => Embedding.....Convert sentences to vector')
    model = SentenceTransformer('keepitreal/vietnamese-sbert')

    input_dim = model.get_sentence_embedding_dimension()
    reduce_dim = ReduceDimensionality(input_dim, 400)
    model.add_module('reduce_dim', reduce_dim)

    # embedding (1 sentence -> 1 vector)
    paras_encode = []
    print('SentenceTransformer successfully')
    for para in paras:
        # print(f'para first {para}')
        sentence_encode = []
        for sentence in tqdm(para, desc="Processing sentences", leave=False):
            sentence_vec = model.encode(sentence)
            sentence_encode.append(sentence_vec)
        # print(f'{para} successfully')
        paras_encode.append(sentence_encode)
    print('==> Sentences => Embedding.....Convert sentences to vector successfully\n')
    return paras_encode


def train_model(paras_encode: list, paras: list) -> (list, KMeans):
    result = []

    for i in range(len(paras_encode)):
        print(i)
        X = paras_encode[i]
        try:
            n_clusters = 4
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans = kmeans.fit(X)
        except:
            try:
                n_clusters = 3
                kmeans = KMeans(n_clusters=n_clusters)
                kmeans = kmeans.fit(X)
            except:
                try:
                    n_clusters = 2
                    kmeans = KMeans(n_clusters=n_clusters)
                    kmeans = kmeans.fit(X)
                except:
                    result.append(paras[i])
                    continue

        avg = []
        for j in range(n_clusters):
            idx = np.where(kmeans.labels_ == j)[0]
            avg.append(np.mean(idx))

        closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
        ordering = sorted(range(kmeans.cluster_centers_), key=lambda k: avg[k])
        summary = ' '.join([paras[i][closest[idx]] for idx in ordering])
        result.append(summary)

    print('==> KMeans Summarizing successfully')
    return result, kmeans


def save_model(kmeans: KMeans):
    print('7. Saving model.....')
    import pickle

    with open(f'kmeans_{CURRENT_DATE}.pkl', "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"label_{CURRENT_DATE}.pkl", "wb") as f:
        pickle.dump(kmeans.labels_, f)
    print('==> Saving model successfully')


def result_train_model(result: list, data_train: pd.DataFrame, data_test: pd.DataFrame):
    print('8. Result training model.....')

    # 8.1 Writing result to file
    print('8.1 Writing result to file========')
    output_path = f"result_train_test_{CURRENT_DATE}_sentoken.txt"
    with open(output_path, "w", encoding="utf-8") as output:
        for item in result:
            output.write(f"{item}\n")
    print('==> Writing result to file successfully========')

    # 8.2 Reading result from file
    print('8.2 Reading result from file ========')
    lines_train_test = []
    with open(output_path, encoding="utf-8") as file:
        lines_train_test = [line.strip() for line in file]
    print('==> Result training model is done')

    # 8.3 Calculate ROUGE score for train data
    print('8.3 Calculate ROUGE score for train data ========')
    calculate_rouge(data_train, lines_train_test=lines_train_test)

    # 8.4 Calculate ROUGE score for test data
    print('8.4 Calculate ROUGE score for test data ========')
    calculate_rouge(data_test, start_idx=len(data_train), lines_train_test=lines_train_test)

    print('==> All processes completed successfully')


def calculate_rouge(data: pd.DataFrame = None, start_idx: int = 0, lines_train_test: list = None):
    rouge_1, rouge_2, rouge_L = [], [], []
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for i in data.index:
        if i + start_idx < len(lines_train_test):
            scores = scorer.score(data.summary[i], lines_train_test[i + start_idx])
            rouge_1.append(list(scores['rouge1'][0:3]))
            rouge_2.append(list(scores['rouge2'][0:3]))
            rouge_L.append(list(scores['rougeL'][0:3]))

    rouge_1 = pd.DataFrame(rouge_1, columns=['precision', 'recall', 'fmeasure'])
    rouge_2 = pd.DataFrame(rouge_2, columns=['precision', 'recall', 'fmeasure'])
    rouge_L = pd.DataFrame(rouge_L, columns=['precision', 'recall', 'fmeasure'])

    for metric in ['precision', 'recall', 'fmeasure']:
        print(f'File {metric} score')
        print(f'Rouge_1: {rouge_1[metric].mean() * 100}')
        print(f'Rouge_2: {rouge_2[metric].mean() * 100}')
        print(f'Rouge_L: {rouge_L[metric].mean() * 100} \n')