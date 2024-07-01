from sentence_transformers import SentenceTransformer

from TTVB.services import train_model


def convert_sentences_to_vectors(paras: list) -> list:
    print('5. Sentences => Embedding.....Convert sentences to vector')
    model = SentenceTransformer('keepitreal/vietnamese-sbert')

    # embedding (1 sentence -> 1 vector)
    paras_encode = []
    for para in paras[:10]:  # Giới hạn số phần tử là 10
        sentence_encode = []
        sentence_vec = model.encode(para)
        sentence_encode.append(sentence_vec)
        print(f'{para}')
        paras_encode.append(sentence_encode)

    return paras_encode

# Ví dụ về danh sách các câu
paras = [
    "Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1",
    "Câu 2",
    "Câu 3",
    "Câu 4",
    "Câu 5 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1 Câu 1",
    "Câu 6",
    "Câu 7",
    "Câu 8",
    "Câu 9",
    "Câu 10"
]

# Gọi hàm để chuyển đổi các câu thành vector
paras_encode = convert_sentences_to_vectors(paras)

train_model(paras_encode, paras)


print("Training model...")
result, kmeans = train_model(paras_encode, paras)

# Lưu mô hình
print("Saving model...")