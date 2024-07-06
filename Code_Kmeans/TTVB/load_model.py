import pandas as pd
import numpy as np
import nltk
import pickle

from sklearn.metrics import pairwise_distances_argmin_min
from TTVB.services import create_tokens_from_Sentences, convert_sentence_to_vector

ROOT = "E:/DATN_Thacsi/Code_Kmeans/models"
ROOT_DATA = "E:/DATN_Thacsi/data"


# load data
print('1. Reading data need to be predicted......')
data_du_bao = pd.read_csv(f'{ROOT_DATA}/dubao.csv')
print('==> Reading data need to be predicted successfully......')

print('2. Shape train and test data ......')
print("==> The number of row and column of data_dubao:", data_du_bao.shape)

print('4. Sentences=> tokenization.....')
#nltk.download('punkt')
paras_du_bao = create_tokens_from_Sentences(data_du_bao)

print('5. Sentences => Embedding.....')
print("Converting sentences to vectors...")
paras_encode = convert_sentence_to_vector(paras_du_bao, None)

print('6. Loading model Kmeans')
with open(f'{ROOT}/kmeans_2024-07-01.pkl', "rb") as f:
    kmeans = pickle.load(f)
with open(f'{ROOT}/label_2024-05-23.pkl', "rb") as f:
    kmeans.labels_ = pickle.load(f)
print('==> Load model Kmeans successfully')

print('6. Summarize data du bao:')
result_du_bao = []
for i in range(len(paras_encode)):
    X = paras_encode[i]

    avg = []
    for j in range(len(paras_encode)):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(2), key=lambda k: avg[k])
    summary = ' '.join([paras_du_bao[i][closest[idx]] for idx in ordering])
    print(summary)
    result_du_bao.append(summary)

print('==> Summarize data dubao successfully')