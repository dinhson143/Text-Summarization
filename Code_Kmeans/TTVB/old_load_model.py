import pandas as pd
import numpy as np
import nltk
import re

# load data
print('1. Reading data need to be predicted......')
data_dubao = pd.read_csv('E:/DATN_Thacsi/data/dubao.csv')
print('==> Reading data need to be predicted successfully......')

print('2. Shape train and test data ......')
print("==> The number of row and column of data_dubao:", data_dubao.shape)

print('3. Reading embedding......')
# Read embedding
word_dict = []
embeddings_index = {}
# embedding_dim = 300
# max_feature = len(embeddings_index) + 2

f = open("E:/DATN_Thacsi/data/vi.vec", encoding='utf-8')
next(f)
for line in f:
    line = line.strip()
    values = line.split(' ')
    word = values[0]
    word_dict.append(word)
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        if len(coefs) == 100:
            coefs = np.concatenate([coefs, np.random.randn(300)])
        embeddings_index[word] = coefs
    except Exception as e:
        pass
f.close()
print('==> Embedding successfully.....')

print('4. Sentences=> tokenization.....')
#nltk.download('punkt')
dubao_index = []
dubao_paras = []
reg="[^\w\s]"
for i in data_dubao.index:
    x = data_dubao.original[i].lower()
    y = x.replace('\n', ' ')
    z = y.strip()
    # data_test.original[i] = z
    data_dubao.loc[i, 'original'] = z
    print(i)
    parasdb_ = nltk.sent_tokenize(data_dubao.original[i])
    for j in range(len(parasdb_)):
        parasdb_[j] = re.sub(reg, '', parasdb_[j])
        parasdb_[j] = parasdb_[j].replace('   ', ' ').replace('  ', ' ')
        parasdb_[j] = parasdb_[j].strip()
    dubao_paras.append(parasdb_)
    dubao_index.append(i)
print('==> Sentences tokenization successfully.\n')

print('5. Sentences => Embedding.....')
# embedding (1 câu -> 1 vector) tập prediect
dubao_paras_encode = []
for para in dubao_paras:
    sentence_encode=[]
    for sentence in para:
        # sentence = gensim.utils.simple_preprocess(sentence)
        # sentence = ' '.join(sentence)
        # sentence_tokenized = ViTokenizer.tokenize(i)
        # print(sentence_tokenized)
        words = sentence.split(" ")
        sentence_vec = np.zeros(400)
        for word in words:
            if word in embeddings_index.keys():
                sentence_vec += embeddings_index[word]
            else:
                sentence_vec += np.random.randn(400)
        sentence_vec = sentence_vec/len(words)
        sentence_encode.append(sentence_vec)
    dubao_paras_encode.append(sentence_encode)
print('==> Sentences => Embedding..... successfully')

print('6. Loading model Kmeans')
from sklearn.metrics import pairwise_distances_argmin_min
import pickle
with open("E:/DATN_Thacsi/data/kmeans_2024-05-26.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("E:/DATN_Thacsi/data/label_2024-05-26.pkl", "rb") as f:
    kmeans.labels_ = pickle.load(f)
print('==> Load model Kmeans successfully')

print('6. Summarize data dubao:')
dubao_result = []
print(data_dubao['original'])
print(len(dubao_paras_encode))
for i in range(len(dubao_paras_encode)):
    X = dubao_paras_encode[i]

    avg = []
    for j in range(len(dubao_paras_encode)):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(2), key=lambda k: avg[k])
    summary = ' '.join([dubao_paras[i][closest[idx]] for idx in ordering])
    print(summary)
    dubao_result.append(summary)

print('==> Summarize data dubao successfully')