# -*- coding: utf-8 -*-
from pathlib import Path
import mne

# Carpeta raíz del proyecto
BASE_DIR = Path(r"C:\Users\adoni\Documents\CurrentStudy") / "data" / "eeg"

# Valores admitidos en “kind”
ALLOWED_KINDS = {
    "unprocessed",                      # señal bruta
    "blink-manual",   # cruda + anot blink manual
    "blink-auto",     # cruda + anot blink automática
    "ica-clean",                  # tras ICA
    "ica-clean-annot",            # tras ICA + anot
}

def load_raw(
    sub: str,
    task: str,
    run: str,
    kind: str = "unprocessed",
    *,
    preload: bool = True,
    verbose: str | bool = False
) -> mne.io.Raw:
    """
    Carga un archivo .fif organizado en la estructura:
        data/eeg/sub-<sub>/raw/<kind>/
            sub-<sub>_task-<task>_run-<run>_raw.fif
    
    Parámetros
    ----------
    sub : str
        Identificador del sujeto, sin ceros a la izquierda o con ellos (p. ej. "1" o "01").
    task : str
        Nombre de la tarea: stroop, rest-closed, rest-open.
    run : str
        Número de repetición/segmento (p. ej. "1" o "01").
    kind : str
        Variante del raw. Debe ser uno de `ALLOWED_KINDS`.
    preload : bool
        Se pasa directamente a `mne.io.read_raw_fif`.
    verbose : str | bool
        Se pasa directamente a `mne.io.read_raw_fif`.

    Devuelve
    --------
    mne.io.Raw
        Objeto Raw cargado y listo para usar.
    """
    # Normalizar entradas numéricas a 2 dígitos
    sub = f"{int(sub):03d}"
    run = f"{int(run):03d}"

    if kind not in ALLOWED_KINDS:
        raise ValueError(
            f"«kind» debe ser uno de {sorted(ALLOWED_KINDS)}, y recibí {kind!r}."
        )

    # Construir ruta
    fname = f"sub-{sub}_task-{task}_run-{run}_raw.fif"
    fpath = BASE_DIR / f"sub-{sub}" / 'raw' / kind / fname

    if not fpath.is_file():
        raise FileNotFoundError(f"No existe el archivo:\n{fpath}")

    return mne.io.read_raw_fif(fpath, preload=preload, verbose=verbose)
