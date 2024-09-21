import multiprocessing
import jieba
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gensim.corpora.dictionary import Dictionary
from gensim.models.word2vec import Word2Vec
from keras.models import Sequential
from keras.layers import Layer, Bidirectional, Embedding, LSTM, Dense
from keras.preprocessing import sequence
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.model_selection import KFold
from keras.regularizers import l2
from sklearn.preprocessing import MaxAbsScaler
cpu_count = multiprocessing.cpu_count()
vocab_dim = 300
n_iterations = 10
n_exposures = 5
window_size = 12
ATT_SIZE = 30
n_epoch = 10
maxlen = 200
batch_size = 32

# Stopwords file paths
hit_stopwords_path = ''
baidu_stopwords_path = ''
scu_stopwords_path = ''

model_save_path_cls = ''


def loadfile(file_path):
    data = pd.read_csv(file_path)
    # 分割积极和消极数据
    positive_data = data[data['Rating'] == 1]
    negative_data = data[data['Rating'] == 0]
    # 确定较小数量并进行抽样
    min_count = min(len(positive_data), len(negative_data))
    positive_data = positive_data.sample(n=min_count, random_state=42)
    negative_data = negative_data.sample(n=min_count, random_state=42)
    # 合并数据
    balanced_data = pd.concat([positive_data, negative_data])
    return balanced_data['Cleaned_Content'].astype(str).tolist(), balanced_data['Rating'].tolist()


def load_stopwords(stopwords_path):
    with open(stopwords_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f.readlines()]
    return set(stopwords)


def tokenizer(data):
    hit_stopwords = load_stopwords(hit_stopwords_path)
    baidu_stopwords = load_stopwords(baidu_stopwords_path)
    scu_stopwords = load_stopwords(scu_stopwords_path)
    merged_stopwords = hit_stopwords.union(baidu_stopwords, scu_stopwords)

    # 分词并过滤停用词
    text = [jieba.lcut(document.replace('\n', '')) for document in data]
    text = [[word for word in doc if word not in merged_stopwords] for doc in text]
    return text


def create_dictionaries(model=None, combined=None):
    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.index_to_key, allow_update=True)
        w2indx = {v: k + 1 for k, v in gensim_dict.items()}
        w2vec = {word: model.wv[word] for word in w2indx.keys()}

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)
                data.append(new_txt)
            return data

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def word2vec_train(combined, large_sample_file_path):
    """
    使用大样本数据训练Word2Vec模型，并继续使用该模型处理小样本数据。

    参数:
    - combined: 小样本数据的分词列表。
    - large_sample_file_path: 大样本数据文件的路径。
    """
    # 加载大样本数据并进行分词
    df_large_sample = pd.read_csv(large_sample_file_path)
    large_sample_texts = df_large_sample['Cleaned_Content'].astype(str).tolist()
    large_sample_tokenized_texts = [list(jieba.cut(text)) for text in large_sample_texts]

    # 合并大样本和小样本数据用于Word2Vec模型训练
    all_texts = large_sample_tokenized_texts + combined

    # 训练Word2Vec模型
    model = Word2Vec(vector_size=vocab_dim,
                     min_count=n_exposures,
                     window=window_size,
                     workers=cpu_count,
                     epochs=n_iterations)
    model.build_vocab(all_texts)
    model.train(all_texts, total_examples=model.corpus_count, epochs=model.epochs)
    model.save('')

    # 使用训练好的Word2Vec模型创建字典和向量表示
    index_dict, word_vectors, combined = create_dictionaries(model=model, combined=combined)
    return index_dict, word_vectors, combined


def get_data(index_dict, word_vectors, combined, y):
    n_symbols = len(index_dict) + 1
    embedding_weights = np.zeros((n_symbols, vocab_dim))

    for word, index in index_dict.items():
        embedding_weights[index, :] = word_vectors[word]

    return embedding_weights, combined, np.array(y)


class AttentionLayer(Layer):
    def __init__(self, attention_size=16, **kwargs):
        self.attention_size = attention_size
        super(AttentionLayer, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config()
        config['attention_size'] = self.attention_size
        return config

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.time_steps = input_shape[1]
        hidden_size = input_shape[2]
        if self.attention_size is None:
            self.attention_size = hidden_size

        self.W = self.add_weight(name='att_weight', shape=(hidden_size, self.attention_size),
                                 initializer='he_normal', trainable=True, regularizer=l2(1e-4))
        self.b = self.add_weight(name='att_bias', shape=(self.attention_size,),
                                 initializer='zeros', trainable=True, regularizer=l2(1e-4))
        self.V = self.add_weight(name='att_var', shape=(self.attention_size,),
                                 initializer='he_normal', trainable=True, regularizer=l2(1e-4))
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        e = tf.keras.activations.relu(tf.keras.backend.dot(x, self.W) + self.b)
        e = tf.keras.backend.dot(e, tf.keras.backend.reshape(self.V, (-1, 1)))
        a = tf.keras.activations.softmax(e, axis=1)
        a = tf.keras.backend.squeeze(a, axis=-1)
        a = tf.keras.backend.expand_dims(a)
        weighted_input = x * a
        output = tf.keras.backend.sum(weighted_input, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[2]


def train_bilstm_att(n_symbols, embedding_weights, combined, y, fold=3):
    kf = KFold(n_splits=fold, shuffle=True)
    all_y_val = []
    all_y_pred_proba = []
    model_list = []

    # 使用MaxAbsScaler将预测值从[0,1]转换为[-1,1]
    scaler = MaxAbsScaler()

    for i, (train_index, val_index) in enumerate(kf.split(combined)):
        print(f'Fold {i + 1}/{fold}')
        x_train, x_val = combined[train_index], combined[val_index]
        y_train, y_val = y[train_index], y[val_index]

        print('定义一个的Keras模型...')
        model = Sequential()
        model.add(Embedding(output_dim=vocab_dim,
                            input_dim=n_symbols,
                            weights=[embedding_weights],
                            input_length=maxlen))
        model.add(Bidirectional(LSTM(units=30, dropout=0.5, return_sequences=True)))
        model.add(AttentionLayer(attention_size=ATT_SIZE))
        model.add(Dense(1, activation='sigmoid'))  # 二分类输出

        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        # 训练模型
        model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=batch_size, epochs=n_epoch, verbose=1)

        # 进行预测
        y_pred_proba = model.predict(x_val)
        all_y_val.extend(y_val)

        # 对预测概率进行缩放
        y_pred_scaled = scaler.fit_transform(y_pred_proba)  # 将输出从[0, 1]缩放到[-1, 1]
        all_y_pred_proba.extend(y_pred_scaled)

        model_list.append(model)

    # 保存最后一个模型
    final_model_path = model_save_path_cls + '/ALl_model.h5'
    model_list[-1].save(final_model_path)

    # 二分类评估
    all_y_pred_binary = (np.array(all_y_pred_proba) > 0).astype(int)  # 使用0为阈值来转换预测为二分类

    print('模型评估结果：')
    print(classification_report(all_y_val, all_y_pred_binary))

    roc_auc = roc_auc_score(all_y_val, all_y_pred_proba)
    print(f'AUC-ROC: {roc_auc}')

    fpr, tpr, thresholds = roc_curve(all_y_val, all_y_pred_proba)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.plot([0, 1], [0, 1], 'k--')  # 对角线
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('AUC-ROC Curve')
    plt.legend(loc='lower right')
    plt.show()



if __name__ == '__main__':
    print('加载数据集...')
    file_path = ''
    combined, y = loadfile(file_path)
    print(len(combined), len(y))

    print('数据预处理...')
    combined = tokenizer(combined)

    print('训练word2vec模型...')
    large_sample_file_path = ''  # 大样本数据文件路径
    index_dict, word_vectors, combined = word2vec_train(combined, large_sample_file_path)

    print('将数据转换为情感分类模型输入所需格式...')
    # Assuming you have binary sentiment labels (0 or 1)
    y_classification = [1 if sentiment > 0 else 0 for sentiment in y]

    embedding_weights, combined, y_classification = get_data(index_dict, word_vectors, combined, y_classification)

    print('情感分类模型特征与标签大小:')
    print(combined.shape, len(y_classification))

    print('训练情感分类模型...')
    train_bilstm_att(len(index_dict) + 1, embedding_weights, combined, y_classification)
