import os
from config import model_dir, save_pure_echo_path, save_echo_noise_path, save_T2_AMPLITUDE_path, generated_echo_noise_path, generated_T2_AMPLITUDE_path, generated_pure_echo_path, true_data_path

def makedir():
    if os.path.exists(model_dir) == False:
        os.makedirs(model_dir)
    if  os.path.exists(save_T2_AMPLITUDE_path) == False:
        os.makedirs(save_T2_AMPLITUDE_path)
    if os.path.exists(true_data_path) == False:
        os.makedirs(true_data_path)
    if os.path.exists(generated_pure_echo_path) == False:
        os.makedirs(generated_pure_echo_path)
    if os.path.exists(generated_echo_noise_path) == False:
        os.makedirs(generated_echo_noise_path)
    if os.path.exists(generated_T2_AMPLITUDE_path) == False:
        os.makedirs(generated_T2_AMPLITUDE_path)
    return "make dir finish"