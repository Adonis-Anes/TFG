"""
Code based on: https://github.com/cbrnr/mnelab/blob/main/src/mnelab/io/xdf.py
"""

from datetime import datetime, timezone
from scipy import stats

import mne
import numpy as np
from mne.io import BaseRaw
import pyxdf
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), 'src')))

from Subjects import Subject


class MyRawXDF(BaseRaw):
    """Raw data from .xdf file."""

    def __init__(self, subject: Subject, task: str, run: int, session: int = 1):
        """Read raw data from my .xdf files.

        Parameters
        ----------
        fname : str
            File name to load.
        streams_ids : list[int]
            IDs of streams to load.
        marker_ids : list[int] | None
            IDs of marker streams to load. If `None`, load all marker streams. A marker
            stream is a stream with a nominal sampling frequency of 0 Hz.
        
        """
        data_dir = os.path.join(os.getcwd(), 'data', 'eeg')
        run = str(run).zfill(3)
        session = str(session).zfill(3)
        common_file_name = f"sub-{subject.his_id}_ses-S{session}_task-{task}_run-{run}"
        xdf_dir = os.path.join(data_dir, f"sub-{subject.his_id}", f"ses-S{session}", 'xdf')

        fname = os.path.join(xdf_dir, f"{common_file_name}.xdf")
        
        streams, header = pyxdf.load_xdf(fname)
        # Order streams by their IDs
        streams = {stream["info"]["stream_id"]: stream for stream in streams}
        
        streams_id_dict = {}
        for stream in pyxdf.resolve_streams(fname):
            streams_id_dict[stream['name']] = stream['stream_id']

        eeg_stream = streams[streams_id_dict['eeg_eeg']]

        labels = []
        for ch in eeg_stream['info']['desc'][0]['channels'][0]['channel']:
            labels.append(ch['label'][0])
        labels[-1] = subject.ref_channel  # Set the last channel as the reference channel

        types = ['eeg'] * 17
       
        first_time = eeg_stream["time_stamps"][0]
        data = eeg_stream["time_series"]

        fs = float(np.array(eeg_stream["info"]["effective_srate"]).item())

        info = mne.create_info(ch_names=labels, sfreq=fs, ch_types=types)

        scale = np.array([1e-6]) # convert to Volts
        data = (data * scale).T


        super().__init__(preload=data, info=info, filenames=[fname])

        # add recordig date and time info
        recording_datetime = header["info"].get("datetime", [None])[0]
        if recording_datetime is not None:
            recording_datetime = recording_datetime[:-2] + ":" + recording_datetime[-2:]
            meas_date = datetime.fromisoformat(recording_datetime)
            self.set_meas_date(meas_date.astimezone(timezone.utc))

        # add montage and set EEG reference
        self.set_montage('standard_1020')
        #self.set_eeg_reference(ref_channels=[subject.ref_channel], verbose=False)


        # add bad channels
        self.info['bads'] = _get_bad_channels(quality_stream=streams[streams_id_dict['eeg_quality']], channel_names=labels)

        # add subject info and experimenter info
        self.info['experimenter'] = 'Adonis'
        self.info['subject_info'] = subject.to_mne_dict()

        # convert marker streams to annotations if task is Stroop
        if 'stroop' in fname:
            stroop_stream = streams[streams_id_dict['PsychoPyMarkers']]
            annotations = _compact_annotations(stroop_stream, first_time)
            self.set_annotations(annotations)


def _delete_not_useful_markers(stroop_stream: dict, 
                              markers=['TRIAL_START', 'EXPERIMENT_START', 'INSTRUCTIONS_ONSET', 
                                       'THANKS_START', 'THANKS_END', 'TRIAL_END', '']) -> tuple[list[str], np.ndarray]:
    """Delete 'TRIAL_START' markers from the Stroop task stream."""
    stroop_tseries = stroop_stream["time_series"]
    stroop_tstamps = stroop_stream["time_stamps"]
    idx_to_del = [i for i, ts in enumerate(stroop_tseries) if ts[0] in markers]
    # Elimina primero de mayor a menor para no desordenar los índices
    for i in sorted(idx_to_del, reverse=True):
        stroop_tseries.pop(i)
        stroop_tstamps = np.delete(stroop_tstamps, i)
    assert len(stroop_tseries) == len(stroop_tstamps), "Error: Length mismatch after deletion."
    return stroop_tseries, stroop_tstamps


def _get_bad_channels(quality_stream: dict, channel_names: list[str]) -> list[str]:
    """Identify bad channels based on quality stream data. """
    mode = stats.mode(quality_stream['time_series'], axis=0, keepdims=False).mode
    bad_channels_idx = np.where(mode > 2)[0].tolist()
    return [channel_names[i] for i in bad_channels_idx]


def _calculate_durations(onsets: np.ndarray) -> np.ndarray:
    """Calculate durations for each marker based on the next marker's onset."""
    durations = np.zeros(len(onsets))
    for i in range(len(onsets) - 1):
        durations[i] = onsets[i + 1] - onsets[i]
    durations[-1] = 0  # Last marker has no duration
    return durations

def _compact_annotations(stroop_stream, first_time):
    stroop_tseries, stroop_tstamps = _delete_not_useful_markers(stroop_stream)
    onsets = stroop_tstamps - first_time
    descriptions = [marker[0] for marker in stroop_tseries]
    durations = _calculate_durations(onsets)

    compact_descriptions = descriptions

    replacements = [("verde", ""), ("rojo", ""), ("azul", ""), 
                ("green", ""), ("red", ""), ("blue", ""),
                ("down", ""), ("left", ""), ("right", ""),
                ("Not_Congruent", "I"), ("Congruent", "C"), 
                ("Incorrect", "I"), ("Correct", "C"), 
                ("WORD_ONSET", "W"), ("RESPONSE", "R"), 
                ("_", "")
            ]
    
    for old_new in replacements:
        compact_descriptions = _replace_description_name(compact_descriptions, old_new)

    return mne.Annotations(onset=onsets, duration=durations, description=compact_descriptions)

def _replace_description_name(descriptions: list[str], old_new: tuple[str, str]) -> list[str]:
    """Replace a specific description name in the list."""
    return [d.replace(old_new[0], old_new[1]) for d in descriptions]

def _calculate_durations(onsets: np.ndarray) -> np.ndarray:
    """Calculate durations for each marker based on the next marker's onset."""
    durations = np.zeros(len(onsets))
    for i in range(len(onsets) - 1):
        durations[i] = onsets[i + 1] - onsets[i]
    durations[-1] = 0  # Last marker has no duration
    return durations

def _create_events(annotations):
    pass


