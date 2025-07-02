import os

# Directorio base: la carpeta donde está este archivo (globals.py)
BASE_DIR = r'C:\Users\adoni\Documents\CurrentStudy'
DATA_DIR = os.path.join(BASE_DIR, 'data', 'eeg')

# Configuración de estructura de archivos
RAW_SUBDIR = 'raw'          # Carpeta para archivos raw
ICA_CLEAN_SUBDIR = 'ica-clean'  # Carpeta para archivos clean

def get_folder_path(sub, type):
    return os.path.join(DATA_DIR, f'sub-{sub}', 'raw', type)


def get_raw_file_path(sub, task, run, type='unprocessed'):
    """Get the path to the raw file for a given subject, task, and run.
    
    Args:
        sub: Subject ID (e.g., '01')
        task: Task name (e.g., 'rest')
        run: Run number (e.g., '01')
        raw_type: 'unprocessed' or 'clean' (default: 'raw')
    
    Returns:
        Full path to the .fif file.
    """
        
    # Nombre del archivo
    file_name = f'sub-{sub}_task-{task}_run-{run}_raw.fif'
    # Construir ruta completa
    return os.path.join(DATA_DIR, f'sub-{sub}', 'raw', type, file_name)