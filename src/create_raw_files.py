import os
import sys
import mne
import pyxdf
import numpy as np
import os


# Add the src directory to the Python path
# This allows you to import modules from the src directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(''), 'src')))
from Subjects import Subjects
from xdf import MyRawXDF

EEG_DATA_DIR = os.path.join(os.getcwd(), 'data', 'eeg')

def create_and_save_raw_reference_projection_all_subjects():
    SUBJECTS = Subjects(os.path.join('data', 'participants.json'))
    for sub in SUBJECTS:
        print(f"👤 Subject info: first_name: {sub.first_name}, id: {sub.id}")
        for task in ['eyes_close', 'eyes_open', 'stroop']:
            for run in ['001', '002', '003', '004']:
                try:
                    fname = f"sub-{sub}_ses-001_task-{task}_run-{run}_raw.fif"  
                    path = os.path.join(EEG_DATA_DIR, f'sub-{sub}', 'ses-S001', 'raw', fname)
                    raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
                    raw.set_eeg_reference(ref_channels="average", verbose=False)
                    raw.set_eeg_reference("average", projection=True, verbose=False)
                    # Save
                    fname = f"sub-{sub}_ses-001_task-{task}_run-{run}_proj-raw.fif"  
                    path = os.path.join(EEG_DATA_DIR, f'sub-{sub}', 'ses-S001', 'raw-proj', fname)
                    raw.save(path, overwrite=True)
                except FileNotFoundError as e:
                    raise e
                except Exception as e:
                    raise e
        #print(f"Finished processing subject {sub.first_name}.\n")

def create_and_save_raw_all_subjects():
    subjects = Subjects(os.path.join('data', 'participants.json'))
    for sub in subjects:
        for task in ['eyes_close', 'eyes_open', 'stroop']:
            for run in ['001', '002', '003', '004']:
                xdf_name = f"sub-{sub.his_id}_ses-S001_task-{task}_run-{run}.xdf"
                path = os.path.join(EEG_DATA_DIR, f'sub-{sub.his_id}', 'ses-S001', 'xdf', xdf_name)
                if os.path.exists(path):
                    try:
                        # Create MNE raw object using my class
                        raw = MyRawXDF(sub, task, run)
                        # Save
                        fname = f"sub-{sub.his_id}_ses-S001_task-{task}_run-{run}_raw.fif"
                        output_dir = os.path.join(EEG_DATA_DIR, f'sub-{sub.his_id}', 'ses-S001', 'raw', fname)
                        raw.save(output_dir, overwrite=True, verbose=False)
                        print(f"✅ Raw saved for first_name: {sub.first_name}, id: {sub.id}, task: {task}, run: {run}")
                    except Exception as e:
                        raise e  
                    

def main():
    create_and_save_raw_all_subjects()
    #create_and_save_raw_reference_projection_all_subjects()
if __name__ == "__main__":
   main()
