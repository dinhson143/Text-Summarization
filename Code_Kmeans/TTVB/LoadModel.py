# load data
import pandas as pd
import numpy as np

print('1. Đọc dữ liệu dự báo......')

data_dubao = pd.read_csv('E:/DATN_Thacsi/vietnews_data/data/dubao.csv')
print('===Hoàn thành Đọc dữ liệu dự báo......')
# Shpare
data_dubao.shape

print('Dự báo top 10......')
X_train = data_dubao['original']
for i in range(2):
    print(X_train[i])

print('2. Đọc embedding......')
# Read embedding
word_dict = []
embeddings_index = {}
embedding_dim = 300
max_feature = len(embeddings_index) + 2

f = open("E:/DATN_Thacsi/vietnews_data/data/W2V_ner.vec",encoding='utf-8')
for line in f:
    values = line.split(' ')
    word = values[0]
    word_dict.append(word)
    try:
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except Exception as e:
        pass
f.close()
print('Embedding hoàn thành')

print('3. Sentences=> tokenization.....')
# tách câu trong từng đoạn
# tập test
import nltk
import re
#nltk.download('punkt')
dubao_index=[]
dubao_paras=[]
reg="[^\w\s]"
for i in data_dubao.index:
    x = data_dubao.original[i].lower()
    y = x.replace('\n', ' ')
    z = y.strip()
    # data_test.original[i] = z
    data_dubao.loc[i, 'original'] = z
    print(i)
    parasdb_=nltk.sent_tokenize(data_dubao.original[i])
    for i in range(len(parasdb_)) :
       parasdb_[i] = re.sub(reg,'', parasdb_[i])
       parasdb_[i] = parasdb_[i].replace('   ',' ').replace('  ',' ')
       parasdb_[i]=parasdb_[i].strip()
    dubao_paras.append(parasdb_)
    dubao_index.append(i)
print('Hoàn thành Sentences => tokenization.....')

print('4. Sentences => Embedding.....')
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
        sentence_vec = np.zeros((300))
        for word in words:
            if word in embeddings_index.keys():
                sentence_vec+=embeddings_index[word]
            else:
                sentence_vec+=np.random.randn(300)
        sentence_vec=sentence_vec/len(words)
        sentence_encode.append(sentence_vec)
    dubao_paras_encode.append(sentence_encode)
print('4. Hoàn thành Sentences => Embedding.....')

print('5. Load model Kmeans')
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import pickle
with open("E:/DATN_Thacsi/vietnews_data/data/kmeans21022024.pkl", "rb") as f:
    kmeans = pickle.load(f)
with open("E:/DATN_Thacsi/vietnews_data/data/label.pkl", "rb") as f:
    kmeans.labels_ = pickle.load(f)
print('5. Hoàn thành Load model Kmeans')

print('6. Tóm tắt dữ báo:')
dubao_result=[]
#print(data_dubao['original'])
print(len(dubao_paras_encode))
for i in range(len(dubao_paras_encode)):
    print(i)
    X=dubao_paras_encode[i]

    avg = []
    for j in range(len(dubao_paras_encode)):
        idx = np.where(kmeans.labels_ == j)[0]
        #print(idx)
        avg.append(np.mean(idx))

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(2), key=lambda k: avg[k])
    summary = ' '.join([dubao_paras[i][closest[idx]] for idx in ordering])
    print(summary)
    dubao_result.append(summary)

print('6. Hoàn thành tóm tắt dữ báo:')