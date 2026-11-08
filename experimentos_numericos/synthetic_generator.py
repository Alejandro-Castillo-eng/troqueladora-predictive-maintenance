"""
synthetic_generator.py
======================
Generador de datos sinteticos CONFIGURABLE para los experimentos numericos.

Reutiliza el generador del dataset v3 (generar_dataset_v3.py): calendario real de 2024
(lunes a sabado, 08:00-17:59, 1 lectura/min), las 27 FALLAS REALES de log_fallas_2024.csv y la
firma de cada causa (cuadro causa -> variables). Sobre esa base permite variar, de forma
controlada y con semilla documentada, cuatro propiedades:

- magnitud : fraccion de la firma de falla (1.0 = firma del dataset v3).
               presion: caida 8-18 bar x magnitud · vibracion: subida hasta 2.8-4.5 mm/s x magnitud
               corriente: +10-25 % x magnitud · temperatura: +5-15 degC x magnitud
- ruido    : multiplicador de la desviacion del ruido de los sensores (1 = v3) y de la frecuencia
               de picos transitorios.
- desgaste : rango (min, max) de la duracion del desgaste previo, en dias de turno de 10 h.
               None = rangos del dataset v3 por causa (presion 0.5-4, desalineacion 1-6, sobrecarga 1-5).
- faltantes: fraccion de lecturas perdidas por sensor (en rachas de 1-30 min), imputadas con el
               ultimo valor valido, como haria un sistema de adquisicion.

Ground truth: cada lectura conserva el momento exacto de la siguiente falla y su causa real.
Distribuciones: ver docstring de generar(); todas con numpy.random.default_rng(semilla).
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import generar_dataset_v3 as g3  # noqa: E402

SENSORES = g3.SENSORES
SIGMA_RUIDO = {"temp_motor_c": 0.6, "presion_hidraulica_bar": 1.2, "vibracion_rms_mms": 0.10, "corriente_motor_a": 0.45}
FIRMA = {   # rangos de la firma en el dataset v3 (magnitud = 1)
    "caida_presion_bar": (8.0, 18.0),
    "vibracion_final_mm_s": (2.8, 4.5),
    "factor_corriente": (1.10, 1.25),
    "subida_temperatura_c": (5.0, 15.0),
}


def generar(semilla, magnitud=1.0, ruido=1.0, desgaste=None, faltantes=0.0):
    """
    Devuelve (df, bitacora, registro):
      df       : timestamp + 4 sensores, 1 lectura/min (lunes a sabado, 08:00-17:59, 2024).
      bitacora : las 27 fallas reales (dt_falla = hora de inicio).
      registro : parametros sorteados para cada falla (ground truth del generador).

    Distribuciones (por falla, sorteadas con la semilla):
      duracion del desgaste ~ Uniforme(desgaste) dias de turno (limitada por la reparacion anterior
      de la misma causa); forma ~ {lineal, exponencial, escalonada} equiprobable; magnitudes ~
      Uniforme(FIRMA) x magnitud.
    Operacion normal (identica a v3): carga por trabajo ~ {0.92 ... 1.08} cada Uniforme(60, 240)
      min; ruido AR(1) gaussiano por sensor (sigma = SIGMA_RUIDO x ruido); calentamiento diario;
      picos transitorios ~ Bernoulli(0.0015 x ruido).
    """
    rng = np.random.default_rng(semilla)
    bitacora = g3.cargar_bitacora()
    t = g3.calendario(bitacora)
    valores, _ = g3.operacion_normal(t, rng)
    n = len(t)

    # Ruido adicional: si ruido = k, la desviacion total del ruido pasa de sigma a k * sigma.
    if ruido > 1:
        for s in SENSORES:
            valores[s] = valores[s] + rng.normal(0, SIGMA_RUIDO[s] * np.sqrt(ruido ** 2 - 1), n)
        extra = rng.random(n) < g3.PROB_PICO * (ruido - 1)
        valores["vibracion_rms_mms"][extra] += rng.uniform(1.0, 2.5, extra.sum())
        extra_p = rng.random(n) < g3.PROB_PICO / 2 * (ruido - 1)
        valores["presion_hidraulica_bar"][extra_p] -= rng.uniform(8, 20, extra_p.sum())

    desv = {s: np.zeros(n) for s in SENSORES}
    ruido_falla = {s: np.zeros(n) for s in SENSORES}
    registro, ultima_reparacion = [], {}
    for _, f in bitacora.iterrows():
        fin = int(np.searchsorted(t.values, f["inicio"].to_datetime64(), side="left"))
        rango = desgaste if desgaste is not None else g3.CAUSAS[f["causa"]]["dias"]
        largo = max(1, int(rng.uniform(*rango) * g3.MIN_POR_DIA))
        tipo = rng.choice(["lineal", "exponencial", "escalonada"])
        ini = max(0, fin - largo, min(ultima_reparacion.get(f["causa"], 0), fin - 1))
        ultima_reparacion[f["causa"]] = int(np.searchsorted(t.values, f["fin"].to_datetime64(), side="left"))
        p = np.linspace(0, 1, fin - ini, endpoint=False) + 1.0 / max(fin - ini, 1)
        gp = g3.forma(p, tipo, rng)
        detalle = {}
        if f["causa"] == "Desalineación del troquel":
            delta = (rng.uniform(*FIRMA["vibracion_final_mm_s"]) - g3.NOMINAL["vibracion_rms_mms"]) * magnitud
            desv["vibracion_rms_mms"][ini:fin] = np.maximum(desv["vibracion_rms_mms"][ini:fin], delta * gp)
            ruido_falla["vibracion_rms_mms"][ini:fin] = np.maximum(ruido_falla["vibracion_rms_mms"][ini:fin], 0.25 * gp * magnitud)
            detalle["subida_vibracion_mm_s"] = round(delta, 3)
        elif f["causa"] == "Desajuste de presión de corte":
            caida = rng.uniform(*FIRMA["caida_presion_bar"]) * magnitud
            desv["presion_hidraulica_bar"][ini:fin] = np.minimum(desv["presion_hidraulica_bar"][ini:fin], -caida * gp)
            ruido_falla["presion_hidraulica_bar"][ini:fin] = np.maximum(ruido_falla["presion_hidraulica_bar"][ini:fin], 1.5 * gp * magnitud)
            detalle["caida_presion_bar"] = round(caida, 3)
        else:
            d_i = g3.NOMINAL["corriente_motor_a"] * (rng.uniform(*FIRMA["factor_corriente"]) - 1) * magnitud
            d_t = rng.uniform(*FIRMA["subida_temperatura_c"]) * magnitud
            desv["corriente_motor_a"][ini:fin] = np.maximum(desv["corriente_motor_a"][ini:fin], d_i * gp)
            desv["temp_motor_c"][ini:fin] = np.maximum(desv["temp_motor_c"][ini:fin], d_t * gp ** 1.5)
            ruido_falla["corriente_motor_a"][ini:fin] = np.maximum(ruido_falla["corriente_motor_a"][ini:fin], 0.4 * gp * magnitud)
            detalle.update({"subida_corriente_a": round(d_i, 3), "subida_temperatura_c": round(d_t, 3)})
        registro.append({"id_evento": f["id_evento"], "causa": f["causa"], "forma": tipo,
                         "minutos_de_desgaste": int(fin - ini), **detalle})
    for s in SENSORES:
        valores[s] = valores[s] + desv[s] + ruido_falla[s] * rng.normal(0, 1, n)

    df = pd.DataFrame({"timestamp": t, **{s: valores[s] for s in SENSORES}})
    if faltantes > 0:
        for s in SENSORES:
            perdido = np.zeros(n, dtype=bool)
            objetivo = int(faltantes * n)
            while perdido.sum() < objetivo:
                inicio = int(rng.integers(0, n))
                perdido[inicio:inicio + int(rng.integers(1, 31))] = True
            df.loc[perdido, s] = np.nan
        df[SENSORES] = df[SENSORES].ffill().bfill()

    bitacora = bitacora.rename(columns={"inicio": "dt_falla"})
    return df, bitacora, pd.DataFrame(registro)
