"""
Unified EEG Data Labeling and Peak Detection Tool.

Interactive menu for:
  - EEG data visualization & blink annotation (manual/auto)
  - Peak detection and annotation
"""
import sys
from pathlib import Path
from typing import Tuple, List

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences

# -----------------------------------------------------------------------------
# Project setup
# -----------------------------------------------------------------------------
BASE_DIR = Path(r"C:\Users\adoni\Documents\CurrentStudy")
SRC_DIR = BASE_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocessing import preprocess_raw
from Subjects import Subjects

# -----------------------------------------------------------------------------
# Configuration
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

# Defaults
DEFAULT_SUBJECT = "mario"
DEFAULT_TASK    = "eyes_open"
DEFAULT_RUN     = "001"

# -----------------------------------------------------------------------------
# File & path utilities
# -----------------------------------------------------------------------------
def make_eeg_filename(subject: str, task: str, run: str, ext: str = "fif") -> str:
    return f"sub-{subject}_ses-{SESSION_ID}_task-{task}_run-{run}_raw.{ext}"


def get_eeg_path(subject: str,
                 task: str,
                 run: str,
                 file_type: str,
                 mode: str = "read") -> Path:
    suffix = FILE_SUBDIR.get(file_type, FILE_SUBDIR["raw"])
    fname  = make_eeg_filename(subject, task, run)
    folder = DATA_ROOT / f"sub-{subject}" / f"ses-{SESSION_ID}" / suffix
    if mode == "write":
        folder.mkdir(parents=True, exist_ok=True)
    return folder / fname

# -----------------------------------------------------------------------------
# Core EEG I/O & preprocessing
# -----------------------------------------------------------------------------
def read_raw(subject: str,
             task: str,
             run: str,
             file_type: str = "raw") -> mne.io.Raw:
    path = get_eeg_path(subject, task, run, file_type, mode="read")
    raw = mne.io.read_raw_fif(str(path), preload=True, verbose=False)
    preprocess_raw(raw)
    return raw


def save_raw(raw: mne.io.Raw,
             subject: str,
             task: str,
             run: str,
             file_type: str) -> None:
    path = get_eeg_path(subject, task, run, file_type, mode="write")
    raw.save(str(path), overwrite=True, verbose=False)
    print(f"Saved file: {path}")

# -----------------------------------------------------------------------------
# Blink annotation (using MNE)
# -----------------------------------------------------------------------------
def detect_blinks(raw: mne.io.Raw,
                  channel: str = 'Fpz',
                  prominence: float = 0.01,
                  distance: float = 0.2) -> mne.Annotations:
    data, times = raw.copy().pick_channels([channel]).get_data(return_times=True)
    arr    = data.ravel()
    sfreq  = raw.info['sfreq']
    peaks, _    = find_peaks(arr, prominence=prominence, distance=int(distance * sfreq))
    onsets      = times[peaks] - 0.15
    durations   = [0.3] * len(peaks)
    desc        = ['blink'] * len(peaks)
    return mne.Annotations(onset=onsets, duration=durations, description=desc)

# -----------------------------------------------------------------------------
# Peak detection separate utilities
# -----------------------------------------------------------------------------
def extract_frontal_channel_data(raw: mne.io.Raw,
                                 preferred_channel: str = 'Fpz') -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        data, times = raw.get_data(picks=[preferred_channel], return_times=True)
        ch = preferred_channel
    except ValueError:
        frontal = [c for c in raw.ch_names if c.startswith('F')]
        if frontal:
            data, times = raw.get_data(picks=[frontal[0]], return_times=True)
            ch = frontal[0]
            print(f"Using channel {ch}")
        else:
            data, times = raw.get_data(return_times=True)
            ch = raw.ch_names[0]
            print(f"Using channel {ch}")
    return data.ravel(), times, ch


def detect_peaks(data: np.ndarray,
                 times: np.ndarray,
                 sfreq: float,
                 height_factor: float = 2.0,
                 min_distance: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    dist = int(min_distance * sfreq)
    threshold = np.std(data) * height_factor
    peaks, _ = find_peaks(data, height=threshold, distance=dist)
    prom = peak_prominences(data, peaks)[0]
    return peaks, prom


def plot_signal_with_peaks(times: np.ndarray,
                           data: np.ndarray,
                           peaks: np.ndarray,
                           channel: str,
                           prominences: np.ndarray = None) -> None:
    plt.figure(figsize=(10, 3))
    plt.plot(times, data, label=f'{channel}')
    if peaks.size:
        sizes = 5 + (prominences / prominences.max()) * 15 if prominences is not None else 10
        plt.scatter(times[peaks], data[peaks], s=sizes, label='peaks')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.tight_layout()
    plt.show()

# -----------------------------------------------------------------------------
# Menu & interaction
# -----------------------------------------------------------------------------
def display_menu() -> None:
    print("\n" + "="*40)
    print("UNIFIED EEG TOOL MENU")
    print("="*40)
    print("1. Label: View raw data")
    print("2. Label: Auto-detect blinks and save")
    print("3. Label: Manual annotation and save")
    print("4. Peaks: View raw signal")
    print("5. Peaks: Detect peaks and plot")
    print("6. Peaks: Detect, annotate & save")
    print("0. Exit")
    print("="*40)


def get_info() -> Tuple[str,str,str]:
    sub  = input(f"Subject [{DEFAULT_SUBJECT}]: ").strip() or DEFAULT_SUBJECT
    task = input(f"Task    [{DEFAULT_TASK}]: ").strip()   or DEFAULT_TASK
    run  = input(f"Run     [{DEFAULT_RUN}]: ").strip()   or DEFAULT_RUN
    return sub, task, run


def main() -> None:
    while True:
        display_menu()
        choice = input("Choice (0-6): ").strip()
        if choice == '0':
            print("Bye!"); break

        sub, task, run = get_info()

        if choice == '1':  # view raw
            raw = read_raw(sub, task, run, 'raw')
            raw.plot(picks=['Fpz','F7','F3','Fz','F4','F8']); plt.show()

        elif choice == '2':  # auto detect & save
            raw = read_raw(sub, task, run, 'auto')
            ann = detect_blinks(raw)
            raw.set_annotations(ann)
            raw.plot(picks=['Fpz','F7','F3','Fz','F4','F8']); plt.show()
            save_raw(raw, sub, task, run, 'auto')

        elif choice == '3':  # manual annotate & save
            raw = read_raw(sub, task, run, 'manual')
            fig = raw.plot(picks=['Fpz','F7','F3','Fz','F4','F8'])
            fig.fake_keypress('a'); plt.show()
            save_raw(raw, sub, task, run, 'manual')

        elif choice == '4':  # view signal
            raw = read_raw(sub, task, run, 'raw')
            data, times, ch = extract_frontal_channel_data(raw)
            plot_signal_with_peaks(times, data, np.array([]), ch)

        elif choice == '5':  # detect peaks & plot
            raw = read_raw(sub, task, run, 'raw')
            data, times, ch = extract_frontal_channel_data(raw)
            peaks, prom = detect_peaks(data, times, raw.info['sfreq'])
            print(f"{len(peaks)} peaks on {ch}")
            plot_signal_with_peaks(times, data, peaks, ch, prom)

        elif choice == '6':  # detect, annotate & save
            raw = read_raw(sub, task, run, 'auto')
            data, times, ch = extract_frontal_channel_data(raw)
            peaks, prom = detect_peaks(data, times, raw.info['sfreq'])
            ann = create_blink_annotations(times, peaks)
            raw.set_annotations(ann)
            save_raw(raw, sub, task, run, 'auto')

        else:
            print("Invalid. Choose 0-6.")

if __name__ == '__main__':
    main()
