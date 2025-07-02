"""
peak_detection.py

Blink detection and peak analysis utilities for EEG data, con menú interactivo.
"""
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

# -----------------------------------------------------------------------------
# Project paths and imports
# -----------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\adoni\Documents\CurrentStudy")
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import preprocess_raw
from Subjects import Subjects

# -----------------------------------------------------------------------------
# Configuration constants
# -----------------------------------------------------------------------------
DATA_ROOT = BASE_DIR / "data" / "eeg"
SESSION_ID = "S001"
FILE_SUBDIR = {
    "raw":       "raw",
    "manual":    "raw-blink-manual-annot",
    "auto":      "raw-blink-auto-annot",
    "ica":       "raw-ICA",
    "ica_annot": "raw-ICA-annot",
}
PARTICIPANTS_FILE = BASE_DIR / "data" / "participants.json"
SUBJECTS = Subjects(str(PARTICIPANTS_FILE))
DEFAULT_FILE_TYPE = "raw"

# -----------------------------------------------------------------------------
# File path utilities
# -----------------------------------------------------------------------------
def make_eeg_filename(subject: str, task: str, run: str, suffix: str, ext: str = "fif") -> str:
    return f"sub-{subject}_ses-{SESSION_ID}_task-{task}_run-{run}_{suffix}.{ext}"


def get_eeg_path(subject: str, task: str, run: str, file_type: str,
                 mode: str = "read") -> Path:
    suffix = FILE_SUBDIR.get(file_type, FILE_SUBDIR[DEFAULT_FILE_TYPE])
    fname = make_eeg_filename(subject, task, run, suffix)
    folder = DATA_ROOT / f"sub-{subject}" / f"ses-{SESSION_ID}" / suffix
    if mode == "write":
        folder.mkdir(parents=True, exist_ok=True)
    return folder / fname

# -----------------------------------------------------------------------------
# Core functions
# -----------------------------------------------------------------------------
def read_raw(subject: str, task: str, run: str, file_type: str = DEFAULT_FILE_TYPE) -> mne.io.Raw:
    path = get_eeg_path(subject, task, run, file_type, mode="read")
    raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
    preprocess_raw(raw)
    return raw


def extract_frontal_channel_data(raw: mne.io.Raw,
                                  preferred_channel: str = 'Fpz') -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        data, times = raw.get_data(picks=[preferred_channel], return_times=True)
        channel = preferred_channel
    except ValueError:
        frontal = [ch for ch in raw.ch_names if ch.startswith('F')]
        if frontal:
            data, times = raw.get_data(picks=[frontal[0]], return_times=True)
            channel = frontal[0]
            print(f"Using alternate channel {channel}")
        else:
            data, times = raw.get_data(return_times=True)
            channel = raw.ch_names[0]
            print(f"Using first channel {channel}")
    return data.ravel(), times, channel


def detect_peaks(data: np.ndarray, times: np.ndarray, sfreq: float,
                 height_factor: float = 2.0, min_distance: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    dist_samples = int(min_distance * sfreq)
    threshold = np.std(data) * height_factor
    peaks, _ = find_peaks(data, height=threshold, distance=dist_samples)
    prominences = peak_prominences(data, peaks)[0]
    return peaks, prominences


def create_blink_annotations(times: np.ndarray, peaks: np.ndarray,
                             annot_duration: float = 0.3,
                             pre_peak_offset: float = 0.15) -> mne.Annotations:
    onsets = times[peaks] - pre_peak_offset
    durations = [annot_duration] * len(peaks)
    desc = ['blink'] * len(peaks)
    return mne.Annotations(onset=onsets, duration=durations, description=desc)


def plot_signal_with_peaks(times: np.ndarray, data: np.ndarray,
                           peaks: np.ndarray, channel: str,
                           prominences: np.ndarray = None) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(times, data, label=f'{channel} signal')
    if prominences is not None:
        sizes = 5 + (prominences / prominences.max()) * 15
        plt.scatter(times[peaks], data[peaks], s=sizes, label='peaks')
    else:
        plt.scatter(times[peaks], data[peaks], c='r', label='peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Interactive menu
# -----------------------------------------------------------------------------
def display_menu(current_type: str) -> None:
    print("\n" + "=" * 40)
    print("EEG PEAK DETECTION MENU")
    print("=" * 40)
    print("1. View raw signal")
    print("2. Detect peaks and plot")
    print("3. Detect, annotate into Raw and save annotations")
    print("0. Exit")
    print("=" * 40)
    print(f"Current file type: {current_type}")
    print("=" * 40)


def get_subject_info() -> Tuple[str, str, str]:
    sub = input(f"Subject ID (default: mario): ").strip() or 'mario'
    task = input(f"Task (default: eyes_open): ").strip() or 'eyes_open'
    run = input(f"Run (default: 001): ").strip() or '001'
    return sub, task, run


def main() -> None:
    file_type = DEFAULT_FILE_TYPE
    while True:
        display_menu(file_type)
        choice = input("Enter your choice (0-3): ").strip()
        if choice == '0':
            print("Exiting...")
            break

        sub, task, run = get_subject_info()
        raw = read_raw(sub, task, run, file_type)
        data, times, channel = extract_frontal_channel_data(raw)
        sfreq = raw.info['sfreq']

        if choice == '1':
            plot_signal_with_peaks(times, data, np.array([]), channel)

        elif choice == '2':
            peaks, prom = detect_peaks(data, times, sfreq)
            print(f"Detected {len(peaks)} peaks on channel {channel}")
            plot_signal_with_peaks(times, data, peaks, channel, prom)

        elif choice == '3':
            peaks, prom = detect_peaks(data, times, sfreq)
            annotations = create_blink_annotations(times, peaks)
            raw.set_annotations(annotations)
            save_path = get_eeg_path(sub, task, run, file_type, mode="write")
            raw.save(str(save_path), overwrite=True, verbose=False)
            print(f"Annotations saved to {save_path}")

        else:
            print("Invalid choice, please select 0-3.")


if __name__ == '__main__':
    main()
