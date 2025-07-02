import os
import sys
import warnings

import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.decomposition import PCA

BASE_DIR = r'C:\Users\adoni\Documents\CurrentStudy'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'eeg')

# === OPCIONES DEL USUARIO ===
USE_WAVELET = True  # <--- True para wavelet, False para estadísticos básicos
USE_DATA_AUGMENTATION = True  # <--- True o False

if USE_WAVELET:
    SAVE_PATH = os.path.join(BASE_DIR, 'data', 'blink_wavelet.csv')
else:
    SAVE_PATH = os.path.join(BASE_DIR, 'data', 'blink_stats.csv')

sys.path.append(rf'{BASE_DIR}\src')
from preprocessing import preprocess_raw

sys.path.append(rf'{BASE_DIR}\analysis\blink_detection')
from data_augmentation import jitter_blink_array
from ML_models import find_best_knn_model, find_best_svm_model, evaluate_model, split_data_into_train_test_validation

if USE_WAVELET:
    from feature_extraction import compute_wavelet_coeffs_from_array, wavelet_coeffs_to_dataframe
else:
    from feature_extraction import extract_time_domain_features

warnings.filterwarnings("ignore", message="The events passed to the Epochs constructor are not chronologically ordered")
warnings.filterwarnings("ignore", message="Concatenation of Annotations within Epochs is not supported yet. All annotations will be dropped.")

# --- Paso opcional: recarga df_all si ya existe ---
if os.path.exists(SAVE_PATH):
    resp = input(f"Encontrado '{SAVE_PATH}'. ¿Quieres recargarlo en lugar de reprocesar todo? (y/N): ").strip().lower()
    if resp == 'y':
        df_all = pd.read_csv(SAVE_PATH)
        print(f"✔️  `df_all` recargado desde {SAVE_PATH}.")
    else:
        print("🔄  Reprocesando los datos desde cero...")
        df_all = None
else:
    df_all = None



# --- Si no recargamos, procesamos todos los archivos ---
if df_all is None:
    fnames, paths = [], []
    for sub_dir in os.listdir(DATA_DIR):
        for n in range(1, 3):
            base_path = rf"{BASE_DIR}\data\eeg\{sub_dir}\raw\blink-manual-annot"
            fname = f"{sub_dir}_ses-S001_task-eyes_open_run-00{n}_raw.fif"
            path = os.path.join(base_path, fname)
            if os.path.exists(path):
                fnames.append(fname)
                paths.append(path)

    all_feats_dfs = []

    for i, path in enumerate(paths):
        try:
            print(f"Processing file {i+1}/{len(paths)}: {fnames[i]}")
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            preprocess_raw(raw, h_freq=None)
            events, events_dict = mne.events_from_annotations(raw, regexp='bad', verbose=False)
            blink_events = mne.pick_events(events, events_dict['bad_blink'])
            blink_epochs = mne.Epochs(raw, events=blink_events, event_id=1, tmin=-0.01, tmax=0.1, baseline=(None, 0), reject_by_annotation=False, verbose=False, preload=True)
            non_blink_epochs = mne.make_fixed_length_epochs(raw, duration=0.7, reject_by_annotation=True, id=2, verbose=False, preload=True)
            print(f"  - Found {len(blink_epochs)} blink, {len(non_blink_epochs)} non-blink epochs")

            # --- EXTRACCIÓN DE FEATURES ---
            feats_list = []
            labels = []

            if USE_WAVELET:
                blink_coeffs = compute_wavelet_coeffs_from_array(blink_epochs.get_data()[:, 0, :])
                non_blink_coeffs = compute_wavelet_coeffs_from_array(non_blink_epochs.get_data()[:, 0, :])
                all_coeffs = blink_coeffs + non_blink_coeffs
                df_feats = wavelet_coeffs_to_dataframe(all_coeffs)
                labels = np.concatenate([np.ones(len(blink_coeffs)), np.zeros(len(non_blink_coeffs))])
                df_feats['class'] = labels.astype(int)
                # 6. Apply PCA
                pca = PCA(n_components=13)
                X_pca = pca.fit_transform(df_feats.drop('class', axis=1))
                # 7. Create DataFrame with PCA features
                df_pca = pd.DataFrame(X_pca, columns=[f"wavelet_pca_{j}" for j in range(13)])
                df_pca['class'] = df_feats['class']
                df_feats = df_pca
            else:
                # Estadísticos básicos
                for ep in blink_epochs.get_data()[:, 0, :]:  # solo canal 0 (puedes cambiarlo)
                    feats = extract_time_domain_features(ep)
                    feats_list.append(feats)
                    labels.append(1)
                for ep in non_blink_epochs.get_data()[:, 0, :]:
                    feats = extract_time_domain_features(ep)
                    feats_list.append(feats)
                    labels.append(0)
                df_feats = pd.DataFrame(feats_list)
                df_feats['class'] = labels

            all_feats_dfs.append(df_feats)
        except Exception as e:
            print(f"Error processing {fnames[i]}: {e}")
            continue

    df_all = pd.concat(all_feats_dfs, ignore_index=True)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    df_all.to_csv(SAVE_PATH, index=False)
    print(f"✔️  `df_all` guardado en {SAVE_PATH}.")

# --- A partir de aquí, df_all ya existe ---
X = df_all.drop('class', axis=1)
y = df_all['class']

X_train, X_test, y_train, y_test = split_data_into_train_test_validation(X, y)

if USE_DATA_AUGMENTATION:
    X_blink     = X_train[y_train == 1].values
    n_non_blink = (y_train == 0).sum()
    X_blink_aug = jitter_blink_array(X_blink,
                                     target_count=n_non_blink,
                                     noise_std_factor=0.01,
                                     scale_range=(0.9, 1.1),
                                     random_state=42)
    y_blink_aug = np.ones(len(X_blink_aug), dtype=int)
    X_non_blink = X_train[y_train == 0].values
    y_non_blink = np.zeros(len(X_non_blink), dtype=int)
    X_train_bal = np.vstack([X_blink_aug, X_non_blink])
    y_train_bal = np.concatenate([y_blink_aug, y_non_blink])
    print("\n[INFO] Data augmentation applied.")
else:
    X_train_bal = X_train.values
    y_train_bal = y_train.values
    print("\n[INFO] Data augmentation not applied.")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_bal)
X_test_sc  = scaler.transform(X_test.to_numpy())

print(f"Total samples: {len(X)}")
print(f"Train set (original): {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%} of data)")
print(f"Train set (usado para entrenar): {X_train_bal.shape[0]} samples ({X_train_bal.shape[0]/len(X):.1%} of data)")
print(f"Test set: {X_test_sc.shape[0]} samples ({X_test_sc.shape[0]/len(X):.1%} of data)")
print(f"Class distribution in original train set: {np.bincount(y_train)}")
print(f"Class distribution en set usado para entrenar: {np.bincount(y_train_bal)}")

# Info para el usuario
print("MODEL USING WAVELET FEATURES" if USE_WAVELET else "MODEL USING TIME DOMAIN STATISTICS")
print("=================================================================")

knn_model, knn_params = find_best_knn_model(X_train_sc, y_train_bal)
evaluate_model(knn_model, X_test_sc, y_test)

svm_model, svm_params = find_best_svm_model(X_train_sc, y_train_bal)
evaluate_model(svm_model, X_test_sc, y_test)

os.makedirs(os.path.join(BASE_DIR, 'data', 'models'), exist_ok=True)
joblib.dump(knn_model, rf"{BASE_DIR}\data\models\knn_blink_detection_model.pkl")
joblib.dump(svm_model, rf"{BASE_DIR}\data\models\svm_blink_detection_model.pkl")
print("✔️  Modelos guardados en 'data/models'.")
