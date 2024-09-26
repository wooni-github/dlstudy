# windows
# conda create -n AICE2 python==3.8 -y
# conda activate AICE2
# pip install tensorflow
# pip install pillow
# pip install scikit-learn
# pip install matplotlib
# pip install pandas
# pip install seaborn

import os
from glob import glob
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import tensorflow as tf
import keras
import pathlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D


# score : train 09996, valid 0.8837

def main():
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
    data_dir = pathlib.Path(data_dir)
    print('data dir : ', data_dir)
    # data_dir = 'C:/Users/dwkim/.keras/datasets/flower_photos/'

    classes = os.listdir(data_dir)
    classes = [cls for cls in classes if '.' not in cls]
    print(f'{len(classes)} classes : {classes}')

    cls_info = {}
    for i, cls in enumerate(classes):
        cls_info[cls] = i
        cls_info[i] = cls

    all_img_dirs = []
    all_img_class = []
    all_imgs = []


    for cls in classes:
        dir = os.path.join(data_dir, cls)
        image_names_only = os.listdir(dir)
        print('dir', dir, '-> loads : ', len(image_names_only), 'images')

        cnt = 0
        for image_name_only in image_names_only:
            image_dir = (os.path.join(dir, image_name_only))
            img = Image.open(image_dir).resize((224, 224))
            img = np.array(img) / 255.  # 이미지 스케일링
            all_img_dirs.append(image_dir)
            all_img_class.append(cls_info[cls])
            all_imgs.append(img)
            cnt += 1
            # if cnt == 11:
            #     break

    all_img_dirs = np.array(all_img_dirs)
    all_img_class = np.array(all_img_class)
    all_imgs = np.array(all_imgs)
    print(f'total {len(all_imgs)} images')

    print('img dirs, classes, images shape', all_img_dirs.shape, all_img_class.shape, all_imgs.shape)

    encoder = OneHotEncoder(sparse=False)
    all_img_class_onehot = encoder.fit_transform(all_img_class.reshape(-1, 1))
    print('One-hot encoded classes shape', all_img_class_onehot.shape)


    X_train, X_test, y_train, y_test = train_test_split(all_imgs, all_img_class_onehot, test_size=0.3, stratify=all_img_class_onehot, random_state=41)
    print('x train, test shape : ', X_train.shape, X_test.shape)
    print('y train, test shape : ', y_train.shape, y_test.shape)

    num_epochs = 2
    batch_size = 1280

    base_model = tf.keras.applications.ResNet50V2(include_top=False,
                      input_shape=(224,224,3),
                       weights = 'imagenet',
                        pooling='avg'
                      )
    base_model.trainable = False

    inputs = base_model.input
    l1 = Dense(128,activation='relu')(base_model.output)
    outputs = Dense(len(classes),activation='softmax')(l1)
    model = keras.Model(inputs=inputs,outputs=outputs)

    #compile
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    history = model.fit(
        X_train, y_train ,
        validation_data=(X_test, y_test),
        epochs=num_epochs,
        batch_size=batch_size,
        # callbacks=[es, checkpoint]
    )

    print(history.history.keys())

    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Model Accuracy')
    # plt.show()


    i=1
    plt.figure(figsize=(16, 8))
    for img, label in zip(X_test[:8], y_test[:8]):
          # 모델 예측(predict)
          pred = model.predict(img.reshape(-1,224, 224, 3))
          pred_t = np.argmax(pred)
          plt.subplot(2, 4, i)
          plt.title(f'GT:{cls_info[np.argmax(label)]}, Pred: {cls_info[pred_t]}')
          plt.imshow(img)
          plt.axis('off')
          i = i + 1

    plt.show()
    
    import pandas as pd
    pd = pd.read_csv('test.csv')
    answer = []
    predicted_classes = []
    for i, img_path in enumerate(pd['data']):
        img = Image.open(img_path).resize((224, 224))
        img = np.array(img) / 255.0  # normalize to [0, 1]
        img_batch = np.expand_dims(img, axis=0)
        pred = model.predict(img_batch)
        pred_class = np.argmax(pred)
        answer.append([i, img_path, cls_info[pred_class]])
        predicted_classes.append(cls_info[pred_class])
    
    pd['predicted'] = predicted_classes
    pd.to_csv('predicted_results.csv', index=False, encoding='cp949')  # encoding for korean

main()

