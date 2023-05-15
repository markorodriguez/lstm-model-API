import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from src.service.balance_df import generate_balaced_dataset
from src.service.df_to_dataset import df_to_dataset

custom_encoder = tf.keras.layers.TextVectorization(max_tokens=2000)


def train_model():
    
    global df_balanced

    balanced_file = os.listdir('src/dataset/balanced')
    if len(balanced_file) == 0:
        print('No existe un dataset balanceado, se generará uno')
        df_balanced = generate_balaced_dataset()
    else:
        print('Existe un dataset balanceado, leyendo...')
        df_balanced = pd.read_csv('src/dataset/balanced/balanced_out.csv')

    train, val, test = np.split(df_balanced.sample(frac=1), [int(0.8*len(df_balanced)), int(0.9*len(df_balanced))])

    print("Generando datasets...")
    train_data = df_to_dataset(train)
    valid_data = df_to_dataset(val)
    test_data = df_to_dataset(test)

    encoder = tf.keras.layers.TextVectorization(max_tokens=2000)
    encoder.adapt(train_data.map(lambda text, label:text))
    vocab = np.array(encoder.get_vocabulary())

    modelLSTM = tf.keras.Sequential([
    encoder,
    tf.keras.layers.Embedding(
        input_dim=len(encoder.get_vocabulary()),
        output_dim=32,
        mask_zero=True
    ),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    modelLSTM.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])
    
    modelLSTM.evaluate(train_data)
    modelLSTM.evaluate(valid_data)
    modelLSTM.evaluate(test_data)

    modelLSTM.fit(train_data, epochs=5, validation_data=valid_data)
    
    modelLSTM.save('src/model/model.keras', save_format="keras")

def use_model():
    global model

    model_list = os.listdir('src/models')

    if len(model_list) == 0:
        train_model()
        model = tf.keras.models.load_model('src/models/model.keras' )
    else:
        model = tf.keras.models.load_model('src/models/model.keras')

    """
    try:
        model = tf.keras.models.load_model('src/models/model.h5')
    except:
        print('No existe un modelo entrenado, se entrenará uno')
        train_model()
        model = tf.keras.models.load_model('src/models/model.h5')
    """

    return model