import mne
import numpy as np
from scipy.ndimage import uniform_filter1d
from sklearn.preprocessing import StandardScaler


def preprocess_raw(raw,
                   l_freq=0.1,
                   h_freq=48,
                   notch_freqs=(50, 81, 100),
                   notch_widths=np.array([.3, 0.4, 0.2]),
                   interpolate_bads=False):
    """
    1) Notch filtering at specified freqs
    2) Band-pass filtering between l_freq and h_freq
    3) (Optional) Time-domain normalization per channel
    4) Average reference projection (only if not already present)
    5) (Optional) Bad channel interpolation
    """
    # Notch filters
    raw.notch_filter(freqs=notch_freqs,
                     method="spectrum_fit",
                     filter_length="40s",
                     notch_widths=notch_widths,
                     verbose=False)
    raw.notch_filter(freqs=(50,),
                     method="spectrum_fit",
                     filter_length="10s",
                     notch_widths=0.1,
                     verbose=False)

    # Band-pass filter
    raw.filter(l_freq=l_freq, 
               h_freq=h_freq, 
               filter_length='auto', #RuntimeWarning: filter_length (8449) is longer than the signal (8425), distortion is likely. Reduce filter length or filter a longer signal. raw.filter(l_freq=l_freq, h_freq=h_freq, filter_length='auto', verbose=False)
               verbose=False)


    # Average reference (if not already present)
    existing = raw.info.get('projs', [])
    if not any('average' in p.get('desc', '').lower() for p in existing):
        raw.set_eeg_reference('average', projection=True, verbose=False)

    # Interpolate bads
    if interpolate_bads and raw.info.get('bads'):
        raw.interpolate_bads(reset_bads=True, verbose=False)

    return raw



def extract_psd(epochs,
                fmin=1,
                fmax=45,
                n_fft=2048,
                smoothing_bins=5):
    """
    Compute epoch-wise PSD using Welch, smooth with moving average, and convert to dB.
    Parameters:
    - epochs: mne.Epochs object
    - fmin, fmax: frequency range
    - n_fft: FFT window length
    - smoothing_bins: size of moving average window in frequency bins

    Returns:
    - psds_db: array (n_epochs, n_channels, n_freqs)
    - freqs: array of frequency values
    """
    # Use Epochs compute_psd (Welch)
    psd_obj = epochs.compute_psd(method='welch', fmin=fmin, fmax=fmax,
                                 n_fft=n_fft, verbose=False)
    psds = psd_obj.get_data()  # shape: (n_epochs, n_channels, n_freqs)
    freqs = psd_obj.freqs

    # Smooth per frequency axis
    psds_smooth = uniform_filter1d(psds, size=smoothing_bins, axis=-1)

    # Convert to dB
    psds_db = 10 * np.log10(psds_smooth + np.finfo(float).eps)
    return psds_db, freqs


def extract_band_power(psds_db, freqs, bands=None):
    """
    Compute mean band power per channel from psds_db.
    Returns array shape (n_epochs, n_channels * n_bands).
    """
    if bands is None:
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta':  (12, 30),
            'gamma': (30, 45)
        }
    features = []
    for ep in psds_db:  # (n_channels, n_freqs)
        ep_feats = []
        for fmin, fmax in bands.values():
            mask = (freqs >= fmin) & (freqs < fmax)
            ep_feats.extend(ep[:, mask].mean(axis=1))
        features.append(ep_feats)
    return np.array(features)


def subject_normalize(feats):
    """
    Z-score normalization per subject (trials × features).
    """
    mu = feats.mean(axis=0, keepdims=True)
    sigma = feats.std(axis=0, keepdims=True) + 1e-12
    return (feats - mu) / sigma


def global_normalize(feats):
    """
    Z-score normalization globally across all subjects.
    Returns normalized feats and fitted scaler.
    """
    scaler = StandardScaler()
    feats_norm = scaler.fit_transform(feats)
    return feats_norm, scaler
