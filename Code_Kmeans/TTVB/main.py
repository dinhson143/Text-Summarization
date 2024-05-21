from datetime import datetime

import pandas as pd
import numpy as np

print('1. Reading data from csv files......')
try:
    data_train = pd.read_csv('E:/DATN_Thacsi/data/data_train.csv', encoding="utf-8")
    data_test = pd.read_csv('E:/DATN_Thacsi/data/test_data.csv', encoding="utf-8")
    print(data_train.encoding)
    print(data_test.encoding)
    print('Reading data successfully......')
except Exception as e:
    print(f'Reading data failed {e}')

print('2. Shape train and test data ......')
print("The number of row and column of data_train:", data_train.shape)
print("The number of row and column of data_test:", data_test.shape)
print('Shape train and test data successfully......')

print('3. Reading embedding......')
# Read embedding
word_dict = []
embeddings_index = {}
embedding_dim = 300
max_feature = len(embeddings_index) + 2
try:
    f = open("E:/DATN_Thacsi/data/vi.vec", encoding='utf-8')
    next(f)
    for line in f:
        line = line.strip()
        values = line.split(' ')
        word = values[0]
        word_dict.append(word)
        try:
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
        except Exception as e:
            pass
    f.close()
    print('Embedding successfully')
except Exception as e:
    print(f'Reading embedding failed {e}')

print('4. Sentences tokenization.....Create token')
import nltk
import re

# nltk.download('punkt')
index = []
paras = []
reg = "[^\w\s]"
print('4.1 Sentences tokenization => data_train.....')
for i in data_train.index:
    x = data_train.original[i].lower()
    y = x.replace('\n', ' ')
    z = y.strip()
    # data_test.original[i] = z
    data_train.loc[i, 'original'] = z
    print(i)
    paras_ = nltk.sent_tokenize(data_train.original[i])
    for j in range(len(paras_)):
        paras_[j] = re.sub(reg, '', paras_[j])
        paras_[j] = paras_[j].replace('   ', ' ').replace('  ', ' ')
        paras_[j] = paras_[j].strip()
    paras.append(paras_)
    index.append(i)

print('4.2 Sentences tokenization => data_test.....')
for i in data_test.index:
    x = data_test.original[i].lower()
    y = x.replace('\n', ' ')
    z = y.strip()

    data_test.loc[i, 'original'] = z
    print(i)

    paras_ = nltk.sent_tokenize(data_test.original[i])
    for j in range(len(paras_)):
        paras_[j] = re.sub(reg, '', paras_[j])
        paras_[j] = paras_[j].replace('   ', ' ').replace('  ', ' ')
        paras_[j] = paras_[j].strip()
    paras.append(paras_)
    index.append(i)
print('Sentences tokenization successfully.\n')

print('5.Sentences => Embedding.....Convert sentences to vector')
# embedding (1 câu -> 1 vector)
paras_encode = []
for para in paras:
    sentence_encode = []
    for sentence in para:
        words = sentence.split(" ")
        sentence_vec = np.zeros((100))
        for word in words:
            if word in embeddings_index.keys():
                sentence_vec += embeddings_index[word]
            else:
                sentence_vec += np.random.randn(100)
        sentence_vec = sentence_vec / len(words)
        sentence_encode.append(sentence_vec)
    paras_encode.append(sentence_encode)
print('5.Sentences => Embedding.....Convert sentences to vector successfully\n')

print('6.Kmean Summarizing.....')
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

result = []

for i in range(len(paras_encode)):
    print(i)
    X = paras_encode[i]
    try:
        n_clusters = 3
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans = kmeans.fit(X)
        # print(1)
    except:
        try:
            n_clusters = 2
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans = kmeans.fit(X)
            # print(2)
        except:
            result.append(paras[i])
            # print(3)
            continue
    avg = []
    for j in range(n_clusters):
        idx = np.where(kmeans.labels_ == j)[0]
        avg.append(np.mean(idx))

    closest, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, X)
    ordering = sorted(range(n_clusters), key=lambda k: avg[k])
    summary = ' '.join([paras[i][closest[idx]] for idx in ordering])
    result.append(summary)
print('6.Kmean Summarizing successfully')

print('7. Save model.....')
import pickle

current_datetime = datetime.now()
date_format = current_datetime.strftime('%Y-%m-%d')
with open(f"E:/DATN_Thacsi/data/kmeanstest_{date_format}.pkl", "wb") as f:
    pickle.dump(kmeans, f)
with open(f"E:/DATN_Thacsi/data/labeltest_{date_format}.pkl", "wb") as f:
    pickle.dump(kmeans.labels_, f)
print('7. Save model successfully')

print('8. Result training model.....')
print('8.1 Write result to file========')
with open(f"E:/DATN_Thacsi/data/result_train_test_{date_format}_sentoken.txt", "w", encoding="utf-8") as output:
    for item in result:
        output.write("%s\n" % item)
print('8.1 Write result to file successfully========')

print('8.2 Read result from file ========')
lines_train_test = []
with open(f'E:/DATN_Thacsi/data/result_train_test_{date_format}_sentoken.txt', encoding="utf-8") as file:
    for line in file:
        line = line.strip()  # or some other preprocessing
        lines_train_test.append(line)  # storing everything in memory

print('8.3 Calculate rouge_score train')
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_1_train = []
rouge_2_train = []
rouge_L_train = []
for i in data_train.index:
    print(i)
    scores = scorer.score(data_train.summary[i], lines_train_test[i])
    rouge_1_train.append(list(scores['rouge1'][0:3]))  # Đúng kiểu là sẽ kết hợp đc hết
    rouge_2_train.append(list(scores['rouge2'][0:3]))
    rouge_L_train.append(list(scores['rougeL'][0:3]))

rouge_1_train = pd.DataFrame(rouge_1_train, columns=['precision', 'recall', 'fmeasure'])
rouge_2_train = pd.DataFrame(rouge_2_train, columns=['precision', 'recall', 'fmeasure'])
rouge_L_train = pd.DataFrame(rouge_L_train, columns=['precision', 'recall', 'fmeasure'])

for i in ['precision', 'recall', 'fmeasure']:
    print(f'File train {i} score')
    print(f'Rouge_1: {rouge_1_train[i].mean() * 100}')
    print(f'Rouge_2: {rouge_2_train[i].mean() * 100}')
    print(f'Rouge_L: {rouge_L_train[i].mean() * 100} \n')

print('8.4 Calculate rouge_score test ========')
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

rouge_1_test = []
rouge_2_test = []
rouge_L_test = []

for i in data_test.index:
    print(i)
    scores = scorer.score(data_test.summary[i], lines_train_test[i + 105418])
    # scores = scorer.score(data_test.summary[i], lines_train_test[i + 20])
    rouge_1_test.append(list(scores['rouge1'][0:3]))  # Đúng kiểu là sẽ kết hợp đc hết
    rouge_2_test.append(list(scores['rouge2'][0:3]))
    rouge_L_test.append(list(scores['rougeL'][0:3]))

rouge_1_test = pd.DataFrame(rouge_1_test, columns=['precision', 'recall', 'fmeasure'])
rouge_2_test = pd.DataFrame(rouge_2_test, columns=['precision', 'recall', 'fmeasure'])
rouge_L_test = pd.DataFrame(rouge_L_test, columns=['precision', 'recall', 'fmeasure'])

for i in ['precision', 'recall', 'fmeasure']:
    print(f'File test {i} score')
    print(f'Rouge_1: {rouge_1_test[i].mean() * 100}')
    print(f'Rouge_2: {rouge_2_test[i].mean() * 100}')
    print(f'Rouge_L: {rouge_L_test[i].mean() * 100}\n')

if __name__ == '__main__':
    print('Using Kmean summarize text successfully')
