from TTVB.services import (
    reading_data_from_files,
    check_data,
    read_embedding_from_word2vector,
    create_tokens_from_Sentences,
    convert_sentence_to_vector,
    train_model,
    save_model,
    result_train_model
)


def main():
    # Đọc dữ liệu từ files
    print("Reading data from files...")
    data_train, data_test = reading_data_from_files()

    # Kiểm tra dữ liệu
    print("Checking data...")
    check_data(data_train, data_test)

    # Đọc embedding từ Word2Vec
    print("Reading embeddings from Word2Vec...")
    # embeddings_index = read_embedding_from_word2vector()

    # Tạo tokens từ sentences
    print("Creating tokens from sentences...")
    paras_train = create_tokens_from_Sentences(data_train)
    paras_test = create_tokens_from_Sentences(data_test)
    paras = []
    paras.append(paras_train)
    paras.append(paras_test)  # Kết hợp các tokens

    # Chuyển đổi câu thành vector
    print("Converting sentences to vectors...")
    paras_encode = convert_sentence_to_vector(paras, None)

    # Huấn luyện mô hình
    print("Training model...")
    result, kmeans = train_model(paras_encode, paras)

    # Lưu mô hình
    print("Saving model...")
    save_model(kmeans)

    # Đánh giá kết quả huấn luyện mô hình
    print("Evaluating training results...")
    result_train_model(result, data_train, data_test)


if __name__ == "__main__":
    main()