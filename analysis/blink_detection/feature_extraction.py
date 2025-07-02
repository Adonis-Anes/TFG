from scipy import stats as scipy_stats
import numpy as np
import pywt
import pandas as pd
from sklearn.decomposition import PCA
import mne
from scipy.stats import kurtosis, skew


def extract_time_domain_features(epoch_data):
    """Calcula estadísticas básicas del dominio temporal para una época."""
    features = {}
    features['mean'] = np.mean(epoch_data)
    features['std'] = np.std(epoch_data)
    features['max'] = np.max(epoch_data)
    features['min'] = np.min(epoch_data)
    features['kurtosis'] = kurtosis(epoch_data)
    features['skew'] = skew(epoch_data)
    return features

def compute_wavelet_coeffs_from_array(data_array, wavelet='db1', level=3):
    # data_array: shape (n_epochs, n_times)
    coeffs_list = []
    for signal in data_array:
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        coeffs_list.append(coeffs)
    return coeffs_list


def wavelet_coeffs_to_dataframe(coeffs_list, ch_name='Fpz'):
    # 1) Aplanar cada lista de coeficientes
    flat_data = []
    for coeffs in coeffs_list:
        row = []
        for arr in coeffs:
            row.extend(arr.flatten())
        flat_data.append(row)

    # 2) Longitud máxima dentro de este archivo
    max_len = max(len(r) for r in flat_data) if flat_data else 0

    # 3) Rellenar con ceros
    padded = [r + [0]*(max_len - len(r)) for r in flat_data]

    # 4) DataFrame homogéneo
    cols = [f"{ch_name}_coeff_{i}" for i in range(max_len)]
    return pd.DataFrame(padded, columns=cols)

