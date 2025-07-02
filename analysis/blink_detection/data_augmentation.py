import numpy as np

def jitter_blink_array(blink_arr: np.ndarray,
                       target_count: int,
                       noise_std_factor: float = 0.01,
                       scale_range: tuple[float, float] = (0.9, 1.1),
                       random_state: int | None = None
                       ) -> np.ndarray:
    """
    Aumenta por jittering tu array de blink-epochs hasta `target_count`.

    Parámetros
    ----------
    blink_arr : np.ndarray, shape (n_blink, n_times)
        Array con tus épocas de blink originales.
    target_count : int
        Número total de épocas blink que quieres tras el augment.
        Debe ser >= blink_arr.shape[0].
    noise_std_factor : float, optional
        Factor multiplicativo sobre la desviación estándar de cada señal
        para generar el ruido gaussiano. Por defecto 0.01 (1% de la señal).
    scale_range : tuple (min, max), optional
        Parámetros para multiplicar cada señal por un factor aleatorio ∈ [min, max].
    random_state : int or None, optional
        Semilla para reproducibilidad.

    Retorna
    -------
    np.ndarray, shape (target_count, n_times)
        Blink_arr ampliado con `target_count - n_blink` nuevas épocas jittered.
    """
    rng = np.random.default_rng(random_state)
    n_blink, n_times = blink_arr.shape
    if target_count < n_blink:
        raise ValueError(f"target_count ({target_count}) debe ser ≥ número de épocas blink ({n_blink})")

    n_to_gen = target_count - n_blink
    augmented = []

    for _ in range(n_to_gen):
        # 1) elige al azar una época original
        idx = rng.integers(0, n_blink)
        sig = blink_arr[idx].copy()

        # 2) escala aleatoria
        scale = rng.uniform(scale_range[0], scale_range[1])
        sig *= scale

        # 3) añade ruido gaussiano
        noise_std = noise_std_factor * np.std(sig)
        sig += rng.normal(0, noise_std, size=n_times)

        augmented.append(sig)

    # concatenar original + augment
    return np.vstack([blink_arr] + augmented)
