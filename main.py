from forward_data import data_merge
from forward_data import data_generated
from config import SNR,T2_number
from config import generated_pure_echo_path, generated_echo_noise_path, generated_T2_AMPLITUDE_path, save_pure_echo_path, save_echo_noise_path, save_T2_AMPLITUDE_path
from config import model_path
from network import nmr_inversion_model_fcn, nmr_inversion_model_cnn, nmr_inversion_model_cnn_attention, nmr_inversion_model_en_deocde
from train import training, predict

if __name__ == '__main__':
    data_generated(T2_number, SNR, generated_pure_echo_path, generated_echo_noise_path, generated_T2_AMPLITUDE_path)
    data_merge(SNR, generated_pure_echo_path, generated_echo_noise_path, generated_T2_AMPLITUDE_path, save_pure_echo_path, save_echo_noise_path, save_T2_AMPLITUDE_path)
    training(SNR, nmr_inversion_model_fcn, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path)
    predict(model_path, SNR, save_echo_noise_path, save_pure_echo_path, save_T2_AMPLITUDE_path)
