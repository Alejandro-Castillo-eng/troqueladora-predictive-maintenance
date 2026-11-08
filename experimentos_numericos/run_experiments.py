"""
run_experiments.py
==================
Ejecuta los experimentos numericos (validacion etapa 1) del sistema de mantenimiento predictivo.

Para cada escenario y cada replica (semilla 1..30):
1. Genera un dataset sintetico con synthetic_generator.generar() (27 fallas reales de 2024).
2. Calcula el RUL de referencia (ground truth) y las 29 variables de train_model.py.
3. Particion cronologica 80/20 (igual que train_model.py).
4. Entrena y evalua, sobre los MISMOS datos (numeros aleatorios comunes, CRN):
   - Modelo RUL propuesto: XGBRegressor (hiperparametros de train_model.py).
   - Linea base RUL: regresion lineal Ridge con las mismas 29 variables.
   - Referencia ingenua RUL: predecir siempre el RUL medio del entrenamiento.
   - Clasificador de tipo de falla propuesto: XGBClassifier (como clasificador_tipo_falla_v3.py).
   - Linea base de clasificacion: reglas de umbral del tablero (app.py): estado por umbrales y
     causa = sensor con mayor desviacion respecto al valor nominal.
5. Guarda una fila por (escenario, replica, metodo) en results/resultados_replicas.csv.

Uso:  python run_experiments.py            (todos los escenarios y el barrido de magnitud)
      python run_experiments.py --rapido   (2 replicas por escenario, para probar)
"""

from pathlib import Path
import argparse
import contextlib
import io
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier, XGBRegressor

AQUI = Path(__file__).resolve().parent
sys.path.insert(0, str(AQUI.parent))
sys.path.insert(0, str(AQUI))
import train_model as tm  # noqa: E402
import synthetic_generator as sg  # noqa: E402

RESULTADOS = AQUI / "results"
SALIDA = RESULTADOS / "resultados_replicas.csv"
REPLICAS = list(range(1, 31))
ESCENARIOS = {
    "tipico":     {"magnitud": 1.00, "ruido": 1, "desgaste": None,       "faltantes": 0.0},
    "borde":      {"magnitud": 0.50, "ruido": 2, "desgaste": (0.5, 2.0), "faltantes": 0.0},
    "extremo":    {"magnitud": 0.25, "ruido": 3, "desgaste": (0.5, 2.0), "faltantes": 0.0},
    "patologico": {"magnitud": 0.25, "ruido": 3, "desgaste": (0.2, 0.6), "faltantes": 0.3},
}
BARRIDO_MAGNITUD = [1.0, 0.75, 0.5, 0.35, 0.25, 0.15]     # curva de rendimiento (otros factores = tipico)
REPLICAS_BARRIDO = list(range(1, 11))
CLASES = ["normal", "Desajuste de presión de corte", "Desalineación del troquel", "Sobrecarga del motor principal"]
FILAS_DIA = tm.MUESTRAS_POR_TURNO_DIA
TOPE = int(tm.MAX_RUL_DIAS * FILAS_DIA)
XGB = dict(n_estimators=180, max_depth=6, learning_rate=0.08, subsample=0.85, colsample_bytree=0.85,
           n_jobs=-1, random_state=42)


def preparar(df, bitacora):
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"])
    with contextlib.redirect_stdout(io.StringIO()):
        df, idx = tm.cruzar_marcas_y_calcular_rul(df, bitacora)
        df, feats = tm.generar_variables_ventana_movil(df)
    n = len(df)
    filas = np.arange(n)
    pos = np.searchsorted(idx, filas, side="left")
    conocida = pos < len(idx)
    sin_tope = np.full(n, np.nan)
    sin_tope[conocida] = (idx[pos[conocida]] - filas[conocida]) / FILAS_DIA
    sin_tope[~conocida & (n - 1 - filas >= TOPE)] = np.inf          # al menos 10 dias sin falla
    # Etiqueta de clase: causa en las 600 tomas previas a la falla; normal a >= 10 dias.
    fila_ev = [int(np.argmin(np.abs(df["dt"].values - t))) for t in bitacora["dt_falla"].values]
    causa = {}
    for (_, f), fila in zip(bitacora.iterrows(), fila_ev):
        causa.setdefault(fila, f["causa"])
    etiqueta = np.full(n, None, dtype=object)
    cerca = conocida & (sin_tope * FILAS_DIA < FILAS_DIA)
    etiqueta[cerca] = [causa[int(idx[p])] for p in pos[cerca]]
    etiqueta[sin_tope >= tm.MAX_RUL_DIAS] = "normal"
    return df, feats, idx, sin_tope, etiqueta


def auc(pred, sin_tope, horizonte):
    cerca = sin_tope <= horizonte
    normal = sin_tope >= tm.MAX_RUL_DIAS
    sel = cerca | normal
    if cerca.sum() == 0 or normal.sum() == 0:
        return np.nan
    return roc_auc_score(cerca[sel].astype(int), -pred[sel])


def metricas_rul(pred, sin_tope, inicio_fallas):
    ok = ~np.isnan(sin_tope)
    real = np.minimum(sin_tope[ok], tm.MAX_RUL_DIAS)
    p = pred[ok]
    y, yh = real <= 1, p <= 1
    normal = sin_tope >= tm.MAX_RUL_DIAS
    alertas = []
    for k, f in enumerate(inicio_fallas):
        a = max(0, f - TOPE, inicio_fallas[k - 1] + 1 if k else 0)
        tramo = pd.Series(pred[a:f] <= 1).rolling(30).sum() >= 30
        alertas.append(bool(tramo.any()))
    return {
        "mae_h": mean_absolute_error(real, p) * 10, "rmse_h": np.sqrt(mean_squared_error(real, p)) * 10,
        "r2": r2_score(real, p),
        "auc_1d": auc(pred, sin_tope, 1), "auc_3d": auc(pred, sin_tope, 3),
        "precision_alerta": precision_score(y, yh, zero_division=0), "recall_alerta": recall_score(y, yh, zero_division=0),
        "f1_alerta": f1_score(y, yh, zero_division=0), "acc_bal_alerta": balanced_accuracy_score(y, yh),
        "fallas_con_alerta": int(np.sum(alertas)), "fallas_test": len(alertas),
        "pct_normal_en_critico": float((pred[normal] <= 1).mean() * 100) if normal.any() else np.nan,
    }


def reglas_tablero(d):
    """Linea base: logica de evaluar_estado_y_causa_raiz() de app.py sin el RUL predicho."""
    t, v, p, c, e = (d["temp_motor_c_mean_60"], d["vibracion_rms_mms_mean_60"], d["presion_hidraulica_bar_mean_60"],
                     d["corriente_motor_a_mean_60"], d["indice_estres_termomecanico"])
    pico = ((d["temp_motor_c"] > 60) & (t < 54)) | ((d["vibracion_rms_mms"] > 2.5) & (v < 1.6)) | \
           ((d["corriente_motor_a"] > 25) & (c < 22.5))
    alerta = (t >= 57.5) | (v >= 2.2) | (p <= 204) | (c >= 23.5) | (e >= 1.5)
    z = np.column_stack([(t - 52.5) / 2.0, (c - 21.0) / 1.2, (v - 1.35) / 0.25, (214.0 - p) / 3.0])
    causa = np.array(["Sobrecarga del motor principal", "Sobrecarga del motor principal",
                      "Desalineación del troquel", "Desajuste de presión de corte"])[np.argmax(z, axis=1)]
    return np.where(alerta.to_numpy() & ~pico.to_numpy(), causa, "normal")


def causa_por_falla(pred):
    no_normal = pred[pred != "normal"]
    if len(no_normal) == 0:
        return "normal"
    v, c = np.unique(no_normal, return_counts=True)
    return v[np.argmax(c)]


def metricas_clasificacion(y, yp, ciclo):
    f1s = f1_score(y, yp, labels=CLASES, average=None, zero_division=0)
    eventos = []
    for c in np.unique(ciclo[y != "normal"]):
        m = (ciclo == c) & (y != "normal")
        eventos.append((y[m][0], causa_por_falla(yp[m])))
    return {
        "accuracy": accuracy_score(y, yp), "acc_bal": balanced_accuracy_score(y, yp),
        "f1_macro": f1_score(y, yp, labels=CLASES, average="macro", zero_division=0),
        **{f"f1_{n}": f for n, f in zip(["normal", "presion", "desalineacion", "sobrecarga"], f1s)},
        "fallas_causa_correcta": int(sum(r == p for r, p in eventos)), "fallas_test": len(eventos),
    }


def una_replica(escenario, params, semilla):
    t0 = time.time()
    df, bitacora, _ = sg.generar(semilla, **params)
    df, feats, idx, sin_tope, etiqueta = preparar(df, bitacora)
    n = len(df)
    sp = int(n * 0.8)
    X = df[feats].to_numpy()
    y_rul = df["rul_dias"].to_numpy()
    inicio_test = [int(f) - sp for f in idx if f >= sp]
    filas = []
    base = {"escenario": escenario, "semilla": semilla, **{k: str(v) for k, v in params.items()}}

    m = XGBRegressor(**XGB).fit(X[:sp], y_rul[:sp])
    pred_xgb = np.clip(m.predict(X[sp:]), 0, 10)
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0)).fit(X[:sp], y_rul[:sp])
    pred_ridge = np.clip(ridge.predict(X[sp:]), 0, 10)
    pred_media = np.full(n - sp, y_rul[:sp].mean())
    for metodo, pred in [("RUL · XGBoost (propuesto)", pred_xgb), ("RUL · Ridge (línea base)", pred_ridge),
                         ("RUL · promedio (ingenuo)", pred_media)]:
        filas.append({**base, "tarea": "rul", "metodo": metodo, **metricas_rul(pred, sin_tope[sp:], inicio_test)})

    con = etiqueta != None  # noqa: E711
    entrena, prueba = con & (np.arange(n) < sp), con & (np.arange(n) >= sp)
    cod = {c: i for i, c in enumerate(CLASES)}
    yc = np.array([cod[v] for v in etiqueta[entrena]])
    clf = XGBClassifier(**XGB, objective="multi:softprob").fit(X[entrena], yc, sample_weight=compute_sample_weight("balanced", yc))
    yp_xgb = np.array(CLASES)[clf.predict(X[prueba])]
    yp_reglas = reglas_tablero(df.loc[prueba])
    y_test = etiqueta[prueba].astype(str)
    pos = np.searchsorted(idx, np.flatnonzero(prueba), side="left")
    for metodo, yp in [("Tipo · XGBoost (propuesto)", yp_xgb), ("Tipo · reglas del tablero (línea base)", yp_reglas)]:
        filas.append({**base, "tarea": "tipo", "metodo": metodo, **metricas_clasificacion(y_test, yp, pos)})
    for f in filas:
        f["segundos"] = round(time.time() - t0, 1)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    RESULTADOS.mkdir(exist_ok=True)
    reps = REPLICAS[:2] if args.rapido else REPLICAS
    reps_b = REPLICAS_BARRIDO[:1] if args.rapido else REPLICAS_BARRIDO
    corridas = [(e, p, s) for e, p in ESCENARIOS.items() for s in reps]
    corridas += [(f"magnitud_{m:.2f}", {**ESCENARIOS["tipico"], "magnitud": m}, s) for m in BARRIDO_MAGNITUD for s in reps_b]
    salida = RESULTADOS / ("prueba_rapida.csv" if args.rapido else SALIDA.name)
    hechas = set()
    if salida.exists() and not args.rapido:
        prev = pd.read_csv(salida)
        hechas = set(zip(prev["escenario"], prev["semilla"]))
    import sklearn, xgboost, scipy
    (RESULTADOS / "entorno.json").write_text(json.dumps({
        "python": platform.python_version(), "so": platform.platform(), "cpu_nucleos": __import__("os").cpu_count(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "scipy": scipy.__version__, "semillas": reps, "semillas_barrido": reps_b,
        "escenarios": {k: {a: str(b) for a, b in v.items()} for k, v in ESCENARIOS.items()},
        "barrido_magnitud": BARRIDO_MAGNITUD}, indent=2, ensure_ascii=False), encoding="utf-8")
    pendientes = [c for c in corridas if (c[0], c[2]) not in hechas]
    print(f"Corridas: {len(corridas)} · pendientes: {len(pendientes)}", flush=True)
    t0 = time.time()
    for i, (esc, params, semilla) in enumerate(pendientes, 1):
        filas = una_replica(esc, params, semilla)
        pd.DataFrame(filas).to_csv(salida, mode="a", header=not salida.exists(), index=False, encoding="utf-8")
        r = {f["metodo"]: f for f in filas}
        print(f"[{i}/{len(pendientes)}] {esc} s={semilla} · AUC RUL {r['RUL · XGBoost (propuesto)']['auc_1d']:.3f} "
              f"· F1 tipo {r['Tipo · XGBoost (propuesto)']['f1_macro']:.3f} vs reglas {r['Tipo · reglas del tablero (línea base)']['f1_macro']:.3f} "
              f"· {filas[0]['segundos']} s · total {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
