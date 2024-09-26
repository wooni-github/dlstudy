# # [tabular : 77%] https://colab.research.google.com/drive/1IJwfwHbmKbRqJSJFYWkDWug1Vhop6pJ1?usp=sharing
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)  # No limit on columns
pd.set_option('display.width', None)  # Adjusts the width to fit all columns
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Activation, Dropout, BatchNormalization
# from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

def smote_split(x_train, y_train):
    # SMOTE를 이용하여 oversampling
    from imblearn.over_sampling import SMOTE
    smote = SMOTE(random_state=0)
    x_train_over, y_train_over = smote.fit_resample(x_train, y_train)
    print('[before] x_train', x_train.shape) # (4911, 28)
    print('[before] y_train', y_train.shape) # (4911)
    print('[after over] x_train', x_train_over.shape) # (7210, 28)
    print('[after over] y_train', y_train_over.shape) # (7210)
    print(pd.Series(y_train_over).value_counts()) # 0과 1의 개수가 똑같음!!!! (기존 7:3 비율이었는데 5:5로 맞춤)
    return x_train_over, y_train_over

def dnn4(x_train, y_train, x_test, y_test):
    import tensorflow as tf
    from keras.callbacks import EarlyStopping
    model2 = Sequential()

    model2.add(Dense(64, activation = 'relu', input_shape = (x_train.shape[-1],), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(BatchNormalization())
    model2.add(Dropout(0.3))
    model2.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(Dropout(0.3))
    model2.add(Dense(16, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model2.add(Dropout(0.3))
    # model2.add(Dense(1, activation = 'sigmoid')) # 0.5보다 크면 1 작으면 0
    model2.add(Dense(2, activation = 'softmax')) # 0.5보다 크면 1 작으면 0
    # 원핫 인코딩 안했으니까

    print('model specification. should be specify input_shape')
    print(model2.summary())
    print('True/False classification -> loss = binary_crossentropy')
    print('Multiple class -> loss = sparse_categorical_crossentropy')
    # model2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['accuracy'])
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model2.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # name = 'best_model.h5'
    # early_stop = EarlyStopping(monitor='val_loss', verbose = 1, patience = 5)
    # check_point = tf.keras.callbacks.ModelCheckpoint(filepath=name, save_weights_only=True, verbose=1)

    history = model2.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 100, batch_size = 128
                        # , callbacks = [early_stop, check_point]
                        # , callbacks = [early_stop]
                        ) # epochs임 epoch가 아니라

    best_model = model2
    pred_best_model = best_model.predict(x_test)
    y_pred = np.argmax(pred_best_model, axis = 1)

    return y_pred, best_model

def preprocessing(df, method = 'drop'):
    print('df original info')
    print(df.info())
    # 1. customerID : 도움 안되니까 삭제
    # 2. MonthlyCharges : float64 인데, TotalCharges : object임 (dtype 변경 해줘야겠지)
    # 3. 결측치 처리
    # 4. one-hot encodig 필요

    print(df.head())
    print('df original describe')
    print(df.describe()) #이상치같은거 처리해야겠지
    print()

    # 1. ID는 필요없으니 삭제
    df.drop('customerID', axis = 1, inplace = True)

    # 2. TotalCharge data type conversion
    if method == 'drop':
        # TotalCharges : 7043 -> 7032
        cond = (df['TotalCharges'] == ' ') # 공백이 있어서 float이 아닌 문자열로 읽은거. 공백이 아니라 숫자나 평균값 등 조건은 다양하겠지
        df = df[~cond] # ~ : (공백인 애들 조건) 반대. 채우거나, 제거하던가 하는 방법을 쓰면 되지
        df['TotalCharges'] = df['TotalCharges'].astype(float)
    elif method == 'fill_average':
        # Replace blank values (' ') with NaN
        df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
        # Convert the column to float after replacing the blank values with NaN
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        # Fill NaN values with the average of the column
        df['TotalCharges'].fillna(df['TotalCharges'].mean(), inplace=True)
    elif method == 'fill_common':
        # Replace blank values (' ') with NaN
        df['TotalCharges'] = df['TotalCharges'].replace(' ', float('nan'))
        # Convert the column to float after replacing the blank values with NaN
        df['TotalCharges'] = df['TotalCharges'].astype(float)
        # Fill NaN values with the most common (mode) value
        df['TotalCharges'].fillna(df['TotalCharges'].mode()[0], inplace=True)
    print(df.info())

    # 3. Null값 처리 (삭제 or 처리)
    print('df.isnull()')
    # print(df.isnull().sum())
    print(df.isnull().sum()[np.where(df.isnull().sum())[0]])

    # # DeviceProtection은 너무 결측치가 많아서 삭제
    df.drop('DeviceProtection', axis = 1, inplace=True)
    # df['SeniorCitizen'].fillna(df['SeniorCitizen'].mean(), inplace=True) # float인 경우 평균값
    # df['gender'].fillna(df['gender'].mode()[0], inplace=True) # object인 경우 최빈값으로 교체

    # 나머지는 drop (혹은 평균값, 최빈값 등)
    df.dropna(inplace=True)

    print(df.isnull().sum())
    #
    # # 정답 분포 확인
    # # sns.countplot(data=df, x='Churn')
    # # plt.show()
    #
    # # 지나치게 correlation이 있다고 생각되면 해소해줘야 되긴 하지만 삭제가 정답은 아님
    # # corr = df.select_dtypes('number').corr()
    # # sns.heatmap(corr, annot=True)
    # # plt.show()
    #

    # 4.encoding
    cal_cols = df.select_dtypes('object').columns.values # 범주형 데이터를 뽑아서 one-hot encoding을 하던 label-encoding을 하던
    print(cal_cols)
    df = pd.get_dummies(data = df, columns= cal_cols, drop_first=True)
    print('df after one-hot encoding')
    print(df.head())

    return df

def split(df):
    answer = 'Churn_Yes'

    # train test split
    from sklearn.model_selection import train_test_split
    x = df.drop(answer, axis=1) # y를 제외한 나머지 x들. df1에서 y만 빼라.
    y = df[answer]
    print('x.shape, y.shape', x.shape, y.shape)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, stratify=y, random_state=42) # stratify : (classification 일때) y비율 그대로 나눠라 (y의 클래스 비율에 맞춰서 나눠라)
    #
    print('sklearn.model_selection.train_test_split -> shape')
    print('x_train', x_train.shape)
    print('y_train', y_train.shape)
    print('x_test', x_test.shape)
    print('y_test', y_test.shape)

    print('x_train', x_train[:3])

    # scaling / normalization 시각화 해서 봐야하긴 함. 그냥해
    # df.info()

    from sklearn.preprocessing import MinMaxScaler # 0~1 normalization
    from sklearn.preprocessing import StandardScaler # average = 0, standard deviation = 1 으로 scaling
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train) # fit transform임
    x_test = scaler.transform(x_test) # transform임

    print('sklearn.preprocessing.StandardScaler')
    print('x_train StandardScaler', x_train[:3])
    print()
    return x_train, y_train, x_test, y_test

def report(y_test, y_pred):
    print('precision', accuracy_score(y_test, y_pred))
    print('recall', recall_score(y_test, y_pred))
    print()


if __name__ == '__main__':
    df = pd.read_csv('telco_churn_data_v1.csv')
    # df = df.dropna()
    # cond = (df['TotalCharges'] == ' ')  # 공백이 있어서 float이 아닌 문자열로 읽은거. 공백이 아니라 숫자나 평균값 등 조건은 다양하겠지
    # df = df[~cond]  # ~ : (공백인 애들 조건) 반대. 채우거나, 제거하던가 하는 방법을 쓰면 되지
    # df.to_csv('telco_churn_data_v1_infer.csv')

    df = preprocessing(df)
    df_infer = df.copy()
    x_train, y_train, x_test, y_test = split(df)
    x_train_over, y_train_over = smote_split(x_train, y_train)

    print('dnn4 over')
    y_pred, best_model = dnn4(x_train_over, y_train_over, x_test, y_test)
    report(y_test, y_pred) # 0.75, 0.76

    # inference
    df_infer = pd.read_csv('telco_churn_data_v1_infer.csv')

    df_infer_ori = df_infer.copy()
    df = preprocessing(df_infer)
    x = df.drop('Churn_Yes', axis=1).values # y를 제외한 나머지 x들. df1에서 y만 빼라.
    y = df['Churn_Yes']

    from sklearn.preprocessing import StandardScaler # average = 0, standard deviation = 1 으로 scaling
    scaler = StandardScaler()
    x = scaler.fit_transform(x) # fit transform임
    # x_test = scaler.transform(x_test) # transform임

    pred_best_model = best_model.predict(x)
    y_pred = np.argmax(pred_best_model, axis = 1)

    answer = []
    cnt = 0
    for i, yt in enumerate(y):
        if y_pred[i] == 1:
            answer.append('Yes')
        else:
            answer.append('No')
        if yt == y_pred[i]:
            cnt += 1

    print('acc', float(cnt)/float(len(y)))
    print(df_infer_ori.head())
    df_infer_ori['predicted'] = answer
    df_infer_ori.to_csv('predicted_results.csv', index=False, encoding='cp949')  # encoding for korean

