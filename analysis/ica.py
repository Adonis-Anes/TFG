import json
import os
import mne
import sys
import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis
from mne.preprocessing import ICA
from matplotlib import pyplot as plt

BASE_DIR = os.getcwd()
EXCLUDE_JSON_PATH = os.path.join(BASE_DIR, 'data', 'exclude_ICA.json')
RAW_TYPE_SAVE = "ica-clean"

sys.path.append(rf'{BASE_DIR}\src')
from preprocessing import preprocess_raw
from globals import get_raw_file_path

# —————————————————————————————————————————————————————————
# Carga de exclude_dict desde JSON
# —————————————————————————————————————————————————————————
def load_exclude_dict():
    """
    Carga el diccionario de componentes excluidos desde el JSON.
    Si el archivo no existe o está vacío, crea un diccionario nuevo.
    """
    try:
        if not os.path.isfile(EXCLUDE_JSON_PATH):
            # Si no existe, creamos un directorio para él si es necesario
            os.makedirs(os.path.dirname(EXCLUDE_JSON_PATH), exist_ok=True)
            # Retornamos un diccionario vacío
            return {}
            
        # Si existe pero está vacío
        if os.path.getsize(EXCLUDE_JSON_PATH) == 0:
            return {}
            
        # Si existe y tiene contenido
        with open(EXCLUDE_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if not isinstance(data, dict):
                print("Advertencia: exclude_ICA.json no contiene un diccionario. Creando uno nuevo.")
                return {}
            return data
    except json.JSONDecodeError:
        # Si hay un error de formato JSON
        print("Error en el formato del archivo JSON. Creando un diccionario nuevo.")
        return {}

# —————————————————————————————————————————————————————————
# Guardar componentes excluidos en JSON
# —————————————————————————————————————————————————————————
def save_excluded_components(task, sub, run, excl, exclude_dict=None, json_path=None):
    """
    Guarda los componentes excluidos en el archivo JSON.
    
    Parameters:
    -----------
    task : str
        Tarea ('stroop', 'rest-open', etc.)
    sub : str
        ID del sujeto
    run : str
        Número de ejecución
    excl : list
        Lista de componentes a excluir
    exclude_dict : dict, optional
        Diccionario de exclusión. Si es None, usa el global
    json_path : str, optional
        Ruta al archivo JSON. Si es None, usa la ruta global
    """
    if exclude_dict is None:
        exclude_dict = globals()['exclude_dict']
    if json_path is None:
        json_path = EXCLUDE_JSON_PATH
        
    exclude_dict.setdefault(task, {}).setdefault(sub, {})[run] = excl
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(exclude_dict, f, indent=3, ensure_ascii=False, separators=(',', ':'))
    print(f"Guardado: {','.join(map(str,excl))}")

exclude_dict = load_exclude_dict()

# —————————————————————————————————————————————————————————
# Detección automática con razones detalladas
# —————————————————————————————————————————————————————————
def detect_auto_ica(ica, raw, hf_band=(35,100), hf_ratio_thresh=0.4, kurt_thresh=5.0):
    sources = ica.get_sources(raw).get_data()
    sfreq = raw.info['sfreq']
    exclude = set()
    reasons = {}
    # HF ratio
    for idx, comp_ts in enumerate(sources):
        f, Pxx = welch(comp_ts, sfreq, nperseg=int(sfreq*2))
        total_power = np.trapz(Pxx, f)
        hf_mask = (f>=hf_band[0]) & (f<=hf_band[1])
        hf_power = np.trapz(Pxx[hf_mask], f[hf_mask])
        if total_power>0:
            ratio=hf_power/total_power
            if ratio>hf_ratio_thresh:
                exclude.add(idx)
                reasons.setdefault(idx,[]).append(f"HF ratio={ratio:.2f}>{hf_ratio_thresh}")
    # Kurtosis
    for idx, comp_ts in enumerate(sources):
        k=kurtosis(comp_ts)
        if abs(k)>kurt_thresh:
            exclude.add(idx)
            reasons.setdefault(idx,[]).append(f"kurtosis={k:.2f}>{kurt_thresh}")
    return sorted(exclude), reasons

# —————————————————————————————————————————————————————————
# Funciones base
# —————————————————————————————————————————————————————————
# ------------- MODIFICACIONES CLAVE -----------------
# 1) load_raw(...)   ->  ahora siempre carga SIN filtrado
# 2) compute_ica(...) ->  crea raw_hp (1 Hz) para entrenar
#                         y devuelve también raw_final (0.1 Hz)
# ----------------------------------------------------

def load_raw(sub, task, run):
    """Carga sin aplicar ningún filtro; la decisión de filtrado
       se hace después según si es entrenamiento o señal final."""
    fname = get_raw_file_path(sub, task, run)
    raw = mne.io.read_raw_fif(fname, preload=True, verbose=False)

    # referencia definitiva para todo el flujo
    raw.set_eeg_reference('average', verbose=False)

    # recorta al primer-último marcador, como antes
    if task == 'stroop':
        ann = raw.annotations
        raw.crop(tmin=ann[0]['onset'], tmax=ann[-1]['onset'], verbose=False)
    elif task == 'rest-open' or task == 'rest-closed':
        # crop 5 seconds before the end
        raw.crop(tmin=5, tmax=raw.times[-1]-5, verbose=False)
    return raw


def compute_ica(raw):
    """ 1) Crea copia high-pass 1 Hz para entrenamiento
        2) Ajusta ICA y devuelve:
           - ica   (object entrenado)
           - raw_final (señal filtrada 0.1 Hz lista para análisis)
           - n     (nº de componentes)                                      """
    # ---------- 1. Copia para entrenar ICA ----------
    raw_hp = raw.copy().filter(l_freq=1., h_freq=None,
                               fir_design='firwin', phase='zero',
                               verbose=False)
    raw_hp.notch_filter(freqs=[50, 100], verbose=False)

    # ---------- 2. Ajuste de ICA ----------
    n = 15 - len(raw.info['bads'])
    ica = ICA(n_components=n, max_iter='auto', random_state=97)
    ica.fit(raw_hp, verbose=False)

    # ---------- 3. Señal final 0.1 Hz ----------
    raw_final = raw.copy().filter(l_freq=0.1, h_freq=40,
                                  fir_design='firwin', phase='zero',
                                  verbose=False)
    return ica, raw_final, n


# —————————————————————————————————————————————————————————
# Menú e interacción
# —————————————————————————————————————————————————————————
def batch_process_all_subjects(task='rest-closed', run='002'):
    """
    Procesa automáticamente todos los sujetos: detecta componentes ICA,
    los excluye y guarda los resultados para la tarea especificada.
    """
    import glob
    import os.path as op
    
    subjects = ["002", "003", "005", "006", "007", "008", "009", "010", "011", "012"]

    subjects = sorted(set(subjects))  # Eliminar duplicados y ordenar
    
    print(f"Procesando {len(subjects)} sujetos para tarea {task}, run {run}:")
    print(', '.join(subjects))
    
    # Confirmar procesamiento
    if input("¿Continuar con el procesamiento automático? (y/n): ").lower() != 'y':
        print("Procesamiento automático cancelado.")
        return
    
    # Procesar cada sujeto
    results = {"completados": 0, "errores": 0, "cortos": 0, "sin_componentes": 0}
    
    for sub in subjects:
        print(f"\nProcesando sujeto {sub}...")
        try:
            # Intentar cargar raw utilizando la función load_raw existente
            try:
                raw0 = load_raw(sub, task, run)
            except FileNotFoundError:
                print(f"  Error: Archivo no encontrado para sujeto {sub}")
                results["errores"] += 1
                continue
            except Exception as e:
                print(f"  Error cargando archivo para sujeto {sub}: {str(e)}")
                results["errores"] += 1
                continue
                
            # Verificar duración mínima
            if raw0.times[-1] < 30:
                print(f"  Advertencia: Grabación demasiado corta para sujeto {sub} ({raw0.times[-1]:.1f}s < 30s)")
                results["cortos"] += 1
                continue
                
            # Calcular ICA
            ica, raw, n = compute_ica(raw0)
            
            # Detectar componentes automáticamente
            idxs, reasons = detect_auto_ica(ica, raw)
            
            # Guardar resultados incluso si no hay componentes detectados
            save_excluded_components(task, sub, run, idxs)
            
            if not idxs:
                print(f"  No se detectaron componentes para excluir en sujeto {sub}")
                results["sin_componentes"] += 1
                # Guardar raw sin cambios
                raw.save(get_raw_file_path(sub, task, run, type=RAW_TYPE_SAVE), overwrite=True)
                print(f"  Guardado: {sub}_{task}_{run} (sin componentes excluidos)")
            else:
                # Mostrar componentes detectados
                s = ','.join(map(str, idxs))
                print(f"  Detectados: {s}")
                for i in idxs:
                    print(f"   -{i}: {';'.join(reasons[i])}")
                
                # Aplicar y guardar
                ica.exclude = idxs
                ica.apply(raw)
                raw.save(get_raw_file_path(sub, task, run, type=RAW_TYPE_SAVE), overwrite=True)
                print(f"  Guardado: {sub}_{task}_{run} ({len(idxs)} componentes excluidos)")
            
            results["completados"] += 1
            
        except Exception as e:
            print(f"  Error procesando sujeto {sub}: {str(e)}")
            results["errores"] += 1
    
    # Resumen final
    print("\n" + "="*40)
    print(f"RESUMEN DE PROCESAMIENTO ({task}, run {run}):")
    print(f"  Total sujetos: {len(subjects)}")
    print(f"  Completados: {results['completados']}")
    print(f"  Sin componentes detectados: {results['sin_componentes']}")
    print(f"  Grabaciones cortas (<30s): {results['cortos']}")
    print(f"  Errores: {results['errores']}")
    print("="*40)


def display_menu(with_subject=False):
    """Muestra el menú principal, con opciones adaptadas según si hay sujeto cargado o no."""
    print("\n"+"-"*40)
    print("ICA ANALYSIS MENU")
    
    if with_subject:
        print("1. Mostrar componentes ICA")
        print("2. Ver correcciones ICA")
        print("3. Aplicar correcciones ICA desde JSON/entrada")
        print("4. Detectar componentes automáticamente")
        print("5. Cambiar número de ejecución")
        print("6. Recargar raw original")
        print("7. Mostrar componentes excluidos actuales")
    
    print("8. Procesar todos los sujetos automáticamente (rest-closed)")
    print("9. Cambiar/cargar sujeto individual")
    print("0. Salir")
    print("-"*40)

def load_subject(task='rest-closed', run='001'):
    """Carga un sujeto específico por ID."""
    sub = input("Subject ID: ").strip().zfill(3)
    run = input(f"Run number (default {run}): ").strip() or run
    run = run.zfill(3)
    task = input(f"Task (default '{task}'): ").strip() or task
    
    try:
        raw0 = load_raw(sub, task, run)
        ica, raw, n = compute_ica(raw0)
        print(f"Sujeto {sub}, task {task}, run {run} cargado correctamente.")
        return raw, ica, n, sub, run, task, run  # El último run es para init
    except Exception as e:
        print(f"Error cargando sujeto: {str(e)}")
        return None, None, None, None, None, None, None

def handle_choice(state):
    raw, ica, n, sub, run, task, init, cont = state
    has_subject = raw is not None
    
    display_menu(with_subject=has_subject)
    options = "0,8,9" if not has_subject else "0-9"
    c = input(f"Opción ({options}): ").strip()
    
    if c == '0':
        cont = False
    elif not has_subject and c not in ['0', '8', '9']:
        print("Primero debe cargar un sujeto (opción 9).")
    elif c == '1' and has_subject:
        if input("Plot raw? (y/n): ").lower()=='y':
            raw.plot(); plt.show()
        ica.plot_sources(raw)
        ica.plot_components()
        if input("Mostrar propiedades? (y/n): ").lower()=='y':
            show_all = input("¿Mostrar todas las propiedades? (y/n, 'n' para seleccionar): ").lower() == 'y'
            if show_all:
                picks = list(range(n))
            else:
                comp_input = input("Componentes a mostrar (coma-sep): ")
                picks = [int(x) for x in comp_input.split(',') if x.strip().isdigit()]
                if not picks:
                    print("Componentes inválidos, mostrando todas.")
                    picks = list(range(n))
            ica.plot_properties(raw, picks=picks)
            plt.show()

    elif c == '2' and has_subject:
        try:
            excl = [int(x) for x in input("Componentes a excluir (coma-sep): ").split(',') if x.strip().isdigit()]
        except Exception:
            print("Entrada inválida de componentes.")
            return (raw, ica, n, sub, run, task, init, cont)  # Fixed return tuple
        ch_input = input("Canales para overlay (coma-sep, default 'eeg'): ").strip()
        picks = ch_input.split(',') if ch_input else ['eeg']
        if not any(picks):
            print("Canales inválidos. Inténtalo de nuevo.")
            return (raw, ica, n, sub, run, task, init, cont)  # Fixed return tuple
        ica.plot_overlay(raw, exclude=excl, picks=picks)

    elif c == '3' and has_subject:
        excl = [int(x) for x in input("Componentes a excluir (coma-sep): ").split(',') if x.strip().isdigit()]
        save_excluded_components(task, sub, run, excl)
        ica.exclude = excl
        ica.apply(raw)
        raw.plot(); plt.show()
        if input("Guardar raw corregido? (y/n): ").lower()=='y':
            raw.save(get_raw_file_path(sub,task,run,type=RAW_TYPE_SAVE), overwrite=True)

    elif c == '4' and has_subject:
        idxs, reasons = detect_auto_ica(ica, raw)
        s = ','.join(map(str, idxs))
        if s == '':
            print('No se han detectado componentes')
        else:
            print(f"Detectados: {s}")
            for i in idxs:
                print(f" -{i}: {';'.join(reasons[i])}")
            if input("Mostrar propiedades? (y/n): ").lower()=='y':
                show_all = input("¿Mostrar todas las propiedades detectadas? (y/n, 'n' para seleccionar): ").lower() == 'y'
                if show_all:
                    picks = idxs
                else:
                    comp_input = input("Componentes a mostrar (coma-sep): ")
                    picks = [int(x) for x in comp_input.split(',') if x.strip().isdigit()]
                    if not picks:
                        print("Componentes inválidos, mostrando todos los detectados.")
                        picks = idxs
                
                for i in picks:
                    ica.plot_properties(raw, picks=[i])
                    plt.show()
            if input("Excluir? (y/n): ").lower()=='y':
                ica.exclude = idxs
                ica.apply(raw)
                raw.plot(); plt.show()
                if input("Guardar? (y/n): ").lower()=='y':
                    save_excluded_components(task, sub, run, idxs)
                    raw.save(get_raw_file_path(sub,task,run,type=RAW_TYPE_SAVE), overwrite=True)

    elif c == '5' and has_subject:        # Cambiar número de ejecución
        nr  = input("Nuevo run: ").strip() or run
        nr  = nr.zfill(3)
        raw0 = load_raw(sub, task, nr)    # carga sin filtrar
        ica, raw, n = compute_ica(raw0)       # <- recibe los 3 valores
        run = nr                              # actualiza run

    elif c == '6' and has_subject:        # Recargar raw original
        raw0 = load_raw(sub, task, init)  # carga sin filtrar
        ica, raw, n = compute_ica(raw0)       # <- idem
        run = init


    elif c == '7' and has_subject:
        current = exclude_dict.get(task, {}).get(sub, {}).get(run, [])
        print(f"Excluidos actuales: {','.join(map(str, current))}")

    elif c == '8':
        batch_process_all_subjects(task='rest-closed', run='003')
    elif c == '9':
        # Cargar nuevo sujeto
        new_raw, new_ica, new_n, new_sub, new_run, new_task, new_init = load_subject(
            task=task if task else 'rest-closed', 
            run=run if run else '001'
        )
        if new_raw is not None:  # Solo actualiza si se cargó correctamente
            raw, ica, n, sub, run, task, init = new_raw, new_ica, new_n, new_sub, new_run, new_task, new_init
    else:
        print("Opción no válida.")

    return (raw, ica, n, sub, run, task, init, cont)  

def main():
    """Función principal del programa."""
    # Iniciar con estado vacío
    state = (None, None, None, None, None, None, None, True)
    
    while state[-1]:  # Mientras continuar sea True
        state = handle_choice(state)
    
    print("Fin de ICA Tool")

# Fix duplicate main call
if __name__ == '__main__':
    main()
    main()
