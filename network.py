import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, Flatten, Multiply, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K



def get_matrix_a(echo_num, echo_space, t2_sample_num, t2_min, t2_max):
    t_list = []
    # 用秒的单位
    for i in range(echo_num):
        t_list.append((i + 1) * echo_space)
    t = np.array(t_list)
    t = np.expand_dims(t, axis=-1)
    log_min, log_max = math.log10(t2_min), math.log10(t2_max)
    t2 = np.logspace(log_min, log_max, t2_sample_num, base=10)
    t2 = np.expand_dims(t2, axis=0)
    t2 = 1 / t2
    matrix_a = np.dot(t, t2)
    matrix_a = np.exp(-1 * matrix_a)
    return matrix_a

def attention(att_dim,inputs,name):
    V = inputs
    QK = Dense(att_dim)(inputs)
    QK = Activation("softmax", name=name)(QK)
    MV = Multiply()([V, QK])
    return(MV)


def nmr_inversion_model_fcn():
    # 编码器解码器结构
    input_shape_echo = (1, 2500)
    input_shape_T2 = (1, 128)

    echo_noise = Input(shape=input_shape_echo)
    T2_AMPLITUDE = Input(shape=input_shape_T2)

    pure_echo = Dense(units=2500, name="pure_echo")(echo_noise)
    x = Dense(units=1024, activation='relu')(pure_echo)
    x = Dense(units=512, activation='relu')(x)
    t2 = Dense(units=128, name="t2")(x)

    model = Model([echo_noise, T2_AMPLITUDE], [pure_echo, t2])
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(optimizer=optimizer,
                  loss={'pure_echo': 'mse',
                        't2': 'mse'
                        },
                  loss_weights={'pure_echo': 1000000,
                                't2': 1.
                                },
                  metrics=["mse", "mae"])
    return model

def nmr_inversion_model_en_deocde():

    # 编码器解码器结构
    input_shape_echo = (1, 2500)
    input_shape_T2 = (1, 128)

    echo_noise = Input(shape=input_shape_echo)
    T2_AMPLITUDE = Input(shape=input_shape_T2)

    encoder = Dense(units=1024, name="encoder")(echo_noise)
    pure_echo = Dense(units=2500, name="pure_echo")(encoder)
    x = Dense(units=1024, activation='relu')(pure_echo)
    x = Dense(units=512, activation='relu')(x)
    t2 = Dense(units=128, name="t2")(x)

    model = Model([echo_noise, T2_AMPLITUDE], [pure_echo, t2])
    optimizer = Adam(learning_rate=0.0001, beta_1=0.5)
    model.compile(optimizer=optimizer,
                  loss={'pure_echo': 'mse',
                        't2': 'mse'
                        },
                  loss_weights={'pure_echo': 1000000,
                                't2': 1.
                                },
                  metrics=["mse", "mae"])
    return model


def nmr_inversion_model_cnn():

    # 编码器解码器结构
    input_shape_echo = (1, 2500)
    input_shape_T2 = (1, 128)

    echo_noise = Input(shape=input_shape_echo)
    T2_AMPLITUDE = Input(shape=input_shape_T2)

    pure_echo = Dense(units=2500, name="pure_echo")(echo_noise)
    cov1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same')(pure_echo)
    cov2 = Conv1D(filters=32, kernel_size=50, strides=10, padding='same')(pure_echo)
    flat1 = Flatten()(cov1)
    flat2 = Flatten()(cov2)
    flat = K.concatenate([flat1, flat2])
    flat = tf.expand_dims(flat, axis=1)
    x = Dense(units=1024, activation='relu')(flat)
    x = Dense(units=512, activation='relu')(x)
    t2 = Dense(units=128, name="t2")(x)

    model = Model([echo_noise, T2_AMPLITUDE], [pure_echo, t2])
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                  loss={'pure_echo': 'mse',
                        't2': 'mse'
                        },
                  loss_weights={'pure_echo': 1000000.,
                                't2': 1.
                                },
                  metrics=["mse", "mae"])
    return model


def nmr_inversion_model_cnn_attention():
    # 编码器解码器结构
    input_shape_echo = (1, 2500)
    input_shape_T2 = (1, 128)

    echo_noise = Input(shape=input_shape_echo)
    T2_AMPLITUDE = Input(shape=input_shape_T2)
    pure_echo = Dense(units=2500, name="pure_echo")(echo_noise)
    attention_layer = attention(2500, pure_echo, "attention_vec1")
    cov1 = Conv1D(filters=32, kernel_size=5, strides=1, padding='same', name="cov1")(attention_layer)
    cov2 = Conv1D(filters=32, kernel_size=50, strides=10, padding='same', name="cov2")(attention_layer)
    flat1 = Flatten()(cov1)
    flat2 = Flatten()(cov2)
    flat = K.concatenate([flat1, flat2])
    flat = tf.expand_dims(flat, axis=1)
    x = Dense(units=1024, activation='relu', name="dense1")(flat)
    x = Dense(units=512, activation='relu', name="dense2")(x)
    t2 = Dense(units=128, name="t2")(x)

    model = Model([echo_noise, T2_AMPLITUDE], [pure_echo, t2])
    optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer,
                  loss={'pure_echo': 'mse',
                        't2': 'mse'
                        },
                  loss_weights={'pure_echo': 1000000.,
                                't2': 1.
                                },
                  metrics=["mse", "mae"])
    return model


