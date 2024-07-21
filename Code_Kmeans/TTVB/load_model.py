import pandas as pd
import numpy as np
import nltk
import pickle
import re

from sentence_transformers import SentenceTransformer
from sklearn.metrics import pairwise_distances_argmin_min, pairwise_distances

from TTVB.services import create_tokens_from_Sentences, convert_sentence_to_vector, find_closests

ROOT = "E:/DATN_Thacsi/Code_Kmeans/models"
ROOT_DATA = "E:/DATN_Thacsi/data"


# load data
print('1. Reading data need to be predicted......')
data_du_bao = pd.read_csv(f'{ROOT_DATA}/dubao.csv')
print('==> Reading data need to be predicted successfully......')

print('2. Shape train and test data ......')
print("==> The number of row and column of data_dubao:", data_du_bao.shape)


print('4. Sentences tokenization.....Create token')

paras_du_bao = []
reg = "[^\w\s]"

print('==> Sentences tokenization => data_train, test.....')
for i in data_du_bao.index:
    normalized_text = data_du_bao.loc[i, 'original'].lower().replace('\n', ' ').strip()
    data_du_bao.loc[i, 'original'] = normalized_text

    paras_ = nltk.sent_tokenize(data_du_bao.original[i])
    # Tokenize token
    for j in range(len(paras_)):
        paras_[j] = re.sub(reg, ' ', paras_[j])
        paras_[j] = paras_[j].replace('   ', ' ').replace('  ', ' ')
        paras_[j] = paras_[j].strip()
    paras_du_bao.append(paras_)
print('==> Sentences tokenization successfully.\n')



print('5. Sentences => Embedding.....')
print("Converting sentences to vectors...")
paras_encode = convert_sentence_to_vector(paras_du_bao)


print('6. Loading model Kmeans')
with open(f'{ROOT}/kmeans_2024-07-11.pkl', "rb") as f:
    kmeans = pickle.load(f)
with open(f'{ROOT}/label_2024-07-11.pkl', "rb") as f:
    kmeans.labels_ = pickle.load(f)
print('==> Load model Kmeans successfully')

print('6. Summarize data du bao:')
result_du_bao = []
# print("1. " + data_du_bao['original'][0])
# print("2. " + data_du_bao['original'][1])
for i in range(len(paras_encode)):
    X = paras_encode[i]

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    summary = ' '.join([paras_du_bao[i][idx] for idx in closest])
    result_du_bao.append(summary)
print(result_du_bao)

print('==> Summarize data dubao successfully')