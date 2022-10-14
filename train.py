import math
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from network import nmr_inversion_model_en_deocde
from network import nmr_inversion_model_cnn
from network import nmr_inversion_model_fcn
from network import nmr_inversion_model_cnn_attention
from tensorflow.keras.models import Model
from network import get_matrix_a
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from config import EPOCHS, BATCH_SIZE, PROJ_DIR, model_dir, model_SNR, model_name, model_path




# 删除生成数据中非稀疏T2谱，使得生成数据接近于真实数据
def data_filter(echo_noise, pure_echo, T2_AMPLITUDE):
    echo_noise = pd.DataFrame(echo_noise)
    pure_echo = pd.DataFrame(pure_echo)
    T2_AMPLITUDE = pd.DataFrame(T2_AMPLITUDE)
    T2_min = T2_AMPLITUDE.min(axis = 1)
    # b = a < 0.00001
    T2_AMPLITUDE = np.array(T2_AMPLITUDE[T2_min <= 0.00001])
    echo_noise = np.array(echo_noise[T2_min <= 0.00001])
    pure_echo = np.array(pure_echo[T2_min <= 0.00001])
    return echo_noise, pure_echo, T2_AMPLITUDE



def data_load(SNR, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path):
    echo_noise = np.load(
        save_echo_noise_path + 'echo_noise'+'_SNR' + str(SNR) + '.npy',
        allow_pickle=True)
    # print(np.argwhere(np.isnan(echo_noise)))
    pure_echo = np.load(
        save_pure_echo_path + 'pure_echo'+'_SNR' + str(SNR) + '.npy',
        allow_pickle=True)
    # print(np.argwhere(np.isnan(pure_echo)))
    T2_AMPLITUDE = np.load(
        save_T2_AMPLITUDE_path + 'T2_AMPLITUDE'+'_SNR' + str(SNR) + '.npy',
        allow_pickle=True)
    # 过滤异常回波串（非稀疏T2）
    echo_noise, pure_echo, T2_AMPLITUDE = data_filter(echo_noise, pure_echo, T2_AMPLITUDE)

    echo_noise = np.expand_dims(echo_noise, axis=1)
    pure_echo = np.expand_dims(pure_echo, axis=1)
    T2_AMPLITUDE = np.expand_dims(T2_AMPLITUDE, axis=1)

    # print(np.argwhere(np.isnan(T2_AMPLITUDE)))
    T2_AMPLITUDE = T2_AMPLITUDE * 10000
    echo_noise_train, echo_noise_test, pure_echo_train, pure_echo_test, T2_AMPLITUDE_train, T2_AMPLITUDE_test = train_test_split(echo_noise, pure_echo, T2_AMPLITUDE, test_size=0.01, random_state=42)

    return echo_noise_train, echo_noise_test, pure_echo_train, pure_echo_test, T2_AMPLITUDE_train, T2_AMPLITUDE_test


def training(SNR, network_model, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path):
    start_time = time.time()
    save_model_path = os.path.join(PROJ_DIR, 'out_result', 'trained_models',
                                   datetime.now().strftime("%Y%m%d-%H%M%S") +
                                   ' NMR_T2_Analysis')
    echo_noise_train, echo_noise_test, pure_echo_train, pure_echo_test, T2_AMPLITUDE_train, T2_AMPLITUDE_test = data_load(SNR, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path)


    nmr_model = network_model()
    callbacks_list = [
        EarlyStopping(monitor='val_loss', patience=10),
        ModelCheckpoint(os.path.join(save_model_path, 'NMR_T2_Analysis.h5'), monitor='val_loss',mode='auto', save_best_only='True')
    ]
    print('training...')
    history = nmr_model.fit([echo_noise_train, T2_AMPLITUDE_train], [pure_echo_train, T2_AMPLITUDE_train],
                            batch_size=BATCH_SIZE,
                            epochs=EPOCHS,
                            validation_split=0.1,
                            shuffle=True,
                            callbacks=callbacks_list
                            )
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')

    plt.legend()
    plt.show()
    # os.mkdir(save_model_path)
    nmr_model.save(os.path.join(save_model_path, 'NMR_T2_Analysis1.h5'))
    end_time = time.time()
    print("Training time:{:0.2f}s".format(end_time - start_time))


def predict(model_path, SNR, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path):

    model = load_model(model_path + "\\NMR_T2_Analysis.h5")
    echo_noise_train, echo_noise_test, pure_echo_train, pure_echo_test, T2_AMPLITUDE_train, T2_AMPLITUDE_test = data_load(SNR, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path)
    print(model.summary())
    result = model.predict([echo_noise_test, T2_AMPLITUDE_test])
    t2_prediction = result[1]
    inversion_echo = result[0]
    # t2_layer_model = Model(inputs=model.input, outputs=model.get_layer('t2_layer').output)
    # t2 = t2_layer_model.predict(echo_noise)
    # t2 = pd.DataFrame(t2)
    # 根据预测T2计算回波串
    t2_prediction = np.array(t2_prediction)
    t2_prediction = t2_prediction / 10000
    t2_prediction[t2_prediction < 0] = 0

    t2_prediction = t2_prediction.reshape(-1, 128)
    inversion_echo = inversion_echo.reshape(-1, 2500)
    T2_AMPLITUDE_test = T2_AMPLITUDE_test.reshape(-1, 128)
    pure_echo_test = pure_echo_test.reshape(-1, 2500)
    echo_noise_test = echo_noise_test.reshape(-1, 2500)
    # 转化为dateframe
    t2_prediction = pd.DataFrame(t2_prediction)
    inversion_echo = pd.DataFrame(inversion_echo)
    echo_noise_test = pd.DataFrame(echo_noise_test)
    pure_echo_test = pd.DataFrame(pure_echo_test)
    T2_AMPLITUDE_test = pd.DataFrame(T2_AMPLITUDE_test) / 10000
    # 计算 RMSE
    # 计算T2
    t2_mse = mean_squared_error(T2_AMPLITUDE_test, t2_prediction)
    t2_mae = mean_absolute_error(T2_AMPLITUDE_test, t2_prediction)
    t2_rmse = np.sqrt(mean_squared_error(T2_AMPLITUDE_test, t2_prediction))
    t2_r2 = r2_score(T2_AMPLITUDE_test, t2_prediction)
    # 计算回波串
    echo_mse = mean_squared_error(pure_echo_test, inversion_echo)
    echo_mae = mean_absolute_error(pure_echo_test, inversion_echo)
    echo_rmse = np.sqrt(mean_squared_error(pure_echo_test, inversion_echo))
    echo_r2 = r2_score(pure_echo_test, inversion_echo)
    # 保存
    t2_prediction.to_csv(model_path + "\\t2_prediction.csv", header=False, index=False)
    inversion_echo.to_csv(model_path + "\\inversion_echo.csv", header=False, index=False)
    T2_AMPLITUDE_test.to_csv(model_path + "\\T2_AMPLITUDE.csv", header=False, index=False)
    pure_echo_test.to_csv(model_path + "\\pure_echo_test.csv", header=False, index=False)
    echo_noise_test.to_csv(model_path + "\\echo_noise_test.csv", header=False, index=False)
    # 输出评价指标
    print('t2 mse: ' + str(t2_mse) + '  t2 mae: ' + str(t2_mae) + ' t2 rmse: ' + str(t2_rmse) + ' t2 r2: ' + str(t2_r2))
    print('echo mse: ' + str(echo_mse) + '  echo mae: ' + str(echo_mae) + ' echo rmse: ' + str(echo_rmse) + ' echo r2: ' + str(echo_r2))


def true_data_evalute(true_data_path, model_SNR):
    TE = 0.2
    # 回波个数
    NE = 2500
    true_echo_test = pd.read_csv(true_data_path + '\\core_normal_data.csv', header=None)
    T2_AMPLITUDE_test = pd.read_csv(true_data_path + '\\BRD_normal_result.csv', header=None)

    T2_AMPLITUDE_test = T2_AMPLITUDE_test * 10000
    true_echo_test = np.expand_dims(true_echo_test, axis=1)
    T2_AMPLITUDE_test = np.expand_dims(T2_AMPLITUDE_test, axis=1)


    # 预测
    model = load_model(model_path + "\\NMR_T2_Analysis.h5")
    result = model.predict([true_echo_test, T2_AMPLITUDE_test])
    t2_prediction = result[1]
    inversion_echo = result[0]


    t2_prediction = t2_prediction.reshape(-1, 128)
    t2_prediction = t2_prediction / 10000
    t2_prediction = pd.DataFrame(t2_prediction)

    T2_AMPLITUDE_test = T2_AMPLITUDE_test.reshape(-1, 128)
    T2_AMPLITUDE_test = T2_AMPLITUDE_test / 10000
    T2_AMPLITUDE_test = pd.DataFrame(T2_AMPLITUDE_test)

    # 计算回波串
    inversion_echo = inversion_echo.reshape(-1, 2500)
    inversion_echo = pd.DataFrame(inversion_echo)

    t2_prediction[t2_prediction < 0] = 0

    t2_prediction.to_csv(true_data_path + "\\t2_network_prediction_modelSNR" + str(model_SNR) + ".csv", header=False, index=False)
    inversion_echo.to_csv(true_data_path + "\\inversion_echo_network_prediction_modelSNR" + str(model_SNR) + ".csv", header=False, index=False)
