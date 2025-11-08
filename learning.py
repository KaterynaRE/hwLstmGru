import numpy as np
from tensorflow.keras.layers import Lambda, Dense, Dropout, BatchNormalization, Input, Bidirectional, LSTM
from tensorflow.keras.backend import ctc_batch_cost
from tensorflow.keras.models import Model

###LOAD DATA
import data

###Побудова моделі LSTM + CTC
# Input
input_data = Input(shape=(None, data.NUM_FEATURES), name='input')
labels = Input(shape=(None,), name='labels')
input_len = Input(shape=(1,), name='input_len')
label_len = Input(shape=(1,), name='label_len')
# Мережа
x = Bidirectional(LSTM(512, return_sequences=True))(input_data)
x = BatchNormalization()(x)
x = Bidirectional(LSTM(512, return_sequences=True))(x)
x = BatchNormalization()(x)
x = Bidirectional(LSTM(256, return_sequences=True))(x)
x = Dropout(0.3)(x)
x = Dense(data.num_classes, activation='softmax')(x)


# CTC
def ctc_loss_lambda_func(y_true, y_pred, input_length, label_length):
    return ctc_batch_cost(y_true, y_pred, input_length, label_length)
lost_out = Lambda(
    lambda args: ctc_loss_lambda_func(*args), output_shape=(1,), name='ctc'
)([labels, x, input_len, label_len])


# Model
model = Model(inputs=[input_data, labels, input_len, label_len], outputs=lost_out)
model.compile(optimizer='adam', loss=lambda y_true, y_pred: y_pred)
print(model.summary())
# Навчання
BATCH_SIZE = 8
history = model.fit(
    x=[data.X, data.Y_padded, data.input_lengths, data.label_lengths],
    y=np.zeros(len(data.X)),
    batch_size=BATCH_SIZE,
    epochs=3,  # 15-20, 30-50, 100-300
    validation_split=0.1,
)
# Створення моделі для передбачення
prediction_model = Model(inputs=input_data, outputs=x)
prediction_model.save('model.h5')
