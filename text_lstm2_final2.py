import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request

# 네이버 영화 리뷰 데이터
# !git clone https://github.com/e9t/nsmc.git


# data pre-processing
final_data = pd.read_csv("ratings.txt", delimiter='\t', quoting=3)
# id | document | label (0, 1) 형태로 들어있음

print('data check')
print(final_data.head(3))
print(final_data.info())

# 양끝의 빈공간 삭제
final_data['document'] = final_data['document'].str.strip()

# null 체크
final_data.isnull().sum()
final_data.dropna(inplace=True)
print('removal null')
print(final_data.info())

# 중복 데이터 제거
print('remove duplicated')
print(final_data['document'].duplicated().sum())
final_data.drop_duplicates(subset=['document'], inplace=True)
print(final_data.info())

# 숫자, 영문, 특수문자 있는지 확인
print(final_data[ final_data['document'].str.contains('[^가-힣 ]') ].head(3))

# 숫자, 영문자, 특수문자등의 글자는 삭제처리
# final_data['문장'].replace('[^가-힣 ]','', regex=True) : 이렇게도 가능
final_data['document'] = final_data['document'].str.replace('[^가-힣 ]','', regex=True) # 0.797
# final_data['document'] = final_data['document'].str.replace('[^가-힣 0-9\p{P}]','', regex=True) # 특수문자, 숫자

# '문장' 컬럼의 내용에서 영문, 특수문자 없음 확인
print(final_data['document'][final_data['document'].str.contains('[^가-힣 ]')].sum())
print(final_data.head(3))

# label '감정' 분포 확인 : 총 6개이며, 고루게 분포 확인. 단 기쁨이 약간 부족해 보임
print('number of labels')
print(final_data['label'].value_counts())

# X, Y 분리
features = final_data['document'].values
labels = final_data['label'].values

print('features shape, labels shape', features.shape, labels.shape)

# features 내용 3개 출력
print('features')
print(features[:3])

print('이벤트 문자열 최대 길이 :{}'.format(max(len(l) for l in features)))
print('이벤트 문자열 평균 길이 :{}'.format(sum(map(len, features))/len(features)))

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(features, labels , test_size=0.2, stratify=labels, random_state=41)
print('splat shape', x_train.shape, x_test.shape, y_train.shape, y_test.shape)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer()
tokenizer.fit_on_texts(x_train) # x_train으로 토큰 단어 사전 만들기

# print(tokenizer.word_index)
# print(tokenizer.index_word)
# print(tokenizer.word_counts) # 빈도수 확인
max_words = len(tokenizer.index_word)
print('max_words', max_words)
print('tokenizer.index_word (index -> word)')
# print(tokenizer.index_word)
print('ex[308324 : 일라이]', tokenizer.index_word[308324])

x_train_seq = tokenizer.texts_to_sequences(x_train)
x_test_seq = tokenizer.texts_to_sequences(x_test)
print('word to sequence', len(x_train_seq), len(x_test_seq)) # 문장을 숫자로 변경 후 갯수 확인

print('x_train')
print(x_train[1:3])
print('x_train_seq')
print(x_train_seq[1:3])

pad_len = max(len(line) for line in x_train_seq) # 문장 최대 길이 (에 맞춰서 0으로 채움)
print('pad_len', pad_len)

x_train_pad = pad_sequences(x_train_seq, maxlen=pad_len)
x_test_pad = pad_sequences(x_test_seq, maxlen=pad_len)
print('x_train_pad')
print(x_train_pad[:1])

from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Bidirectional, LSTM, Embedding
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

max_words = len(tokenizer.index_word) + 1  # 총 단어 갯수 + padding 0 번호
print('max_words', max_words)
max_len = max(len(line) for line in x_train_seq) # 문장 최대 길이 (에 맞춰서 0으로 채움)
print('max_len', max_len)
embedding_dim = 128

# model1
# model = Sequential()
# model.add(Embedding(max_words, embedding_dim, input_length=max_len))
# model.add(Bidirectional(LSTM(16, return_sequences=True)))
# model.add(Bidirectional(LSTM(16)))
# model.add(Flatten())
# model.add(Dense(128, activation='swish'))
# model.add(Dense(32, activation='swish'))
# model.add(Dense(2, activation='softmax'))

# model2 (5epoch 0.7911), (20epoch 0.7896)
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2
lstm_units = 32  # Increased LSTM units
dense_units = 128
dropout_rate = 0.5  # Adding dropout to reduce overfitting
l2_lambda = 0.01  # L2 regularization factor
learning_rate = 0.0001  # Tuning learning rate for better training
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(Bidirectional(LSTM(lstm_units, return_sequences=True)))
model.add(Dropout(dropout_rate))  # Dropout after first LSTM layer
model.add(Bidirectional(LSTM(lstm_units)))
model.add(Dropout(dropout_rate))  # Dropout after second LSTM layer
model.add(Flatten())
model.add(Dense(dense_units, activation='swish', kernel_regularizer=l2(l2_lambda)))  # L2 regularization
model.add(Dropout(dropout_rate))  # Dropout after Dense layer
model.add(Dense(32, activation='swish', kernel_regularizer=l2(l2_lambda)))  # L2 regularization
model.add(Dense(2, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

es = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
history = model.fit(x_train_pad, y_train, epochs=20, batch_size=512,
                    validation_data=(x_test_pad, y_test),
                    validation_split=0.2, verbose=1, callbacks=[es])

# epochs = range(1, len(history.history['accuracy']) + 1)
# plt.plot(epochs, history.history['accuracy'])
# plt.plot(epochs, history.history['val_accuracy'])
# plt.title('Bidirectional LSTM Model Accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')
# plt.show()

model.evaluate(x_test_pad, y_test)

predictions_prob = model.predict(x_test_pad)
predictions = np.argmax(predictions_prob, axis=1)

accuracy = accuracy_score(y_test, predictions)
print(f'Bidirectional LSTM Accuracy: {accuracy:.4f}')


# inference
new_data = pd.read_csv('ratings_test.txt', delimiter='\t', quoting=3)

new_data['sequence'] = new_data['document'].str.strip()
# new_data.dropna(inplace=True)
# new_data.drop_duplicates(subset=['document'], inplace=True)
new_data['sequence'] = new_data['sequence'].str.replace('[^가-힣 ]','', regex=True)

# text to dequence
new_data_sequences = tokenizer.texts_to_sequences(new_data['sequence'].values)
new_data_padded = pad_sequences(new_data_sequences, maxlen=pad_len)

# predict
predictions_prob = model.predict(new_data_padded)
predictions = np.argmax(predictions_prob, axis=1)
# print('predictions', predictions)

# save new csv
new_data['predicted'] = predictions
print(new_data.head())
new_data.drop('sequence', axis = 1)
new_data.to_csv('predicted_results.csv', index=False, encoding='cp949')  # encoding for korean


