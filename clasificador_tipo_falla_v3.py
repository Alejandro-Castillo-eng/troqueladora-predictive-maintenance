"""
clasificador_tipo_falla_v3.py
=============================
Segundo modelo del sistema: predice QUE TIPO de falla se aproxima (el modelo RUL predice CUANDO).

Datos: telemetria_troqueladora_2024_v3.csv + log_fallas_2024.csv (27 fallas reales de 2024).
Variables: las mismas 29 del modelo RUL (train_model.generar_variables_ventana_movil).
Clases:  normal · Desajuste de presión de corte · Desalineación del troquel · Sobrecarga del motor principal
Etiquetas:
- Las ultimas 600 tomas (10 h de turno) antes de cada falla -> causa de esa falla (bitacora real).
- Tomas a >= 10 dias de turno de cualquier falla -> normal.
- Zona intermedia (entre 10 h y 10 dias antes de una falla) -> sin etiqueta (no se usa).
Modelo: XGBClassifier con los mismos hiperparametros que el modelo RUL y pesos balanceados por clase.

Evaluacion:
1. Validacion cruzada dejando fuera un ciclo de falla cada vez (cada falla real se evalua con un
   modelo que no la vio). Metricas por toma (accuracy, precision, recall, F1, matriz de confusion) y
   por falla (causa mayoritaria predicha en las 10 h previas frente a la causa real).
2. Benchmark con las 5 maquinas CNC de Industrial Machine PdM (syn), sin reentrenar, con equivalencia
   por sensor: bearing -> desalineacion; motor_overheat y electrical -> sobrecarga; hydraulic -> presion.
   Variante principal 'trasladada' (los 4 sensores a la escala de la troqueladora). Con 'directa'
   (temperatura y vibracion originales) la vibracion normal de las CNC (~1,8 mm/s) supera la de la
   troqueladora (~1,3 mm/s) y el clasificador marca como desalineacion la mitad del tiempo normal;
   se reportan ambas. La variante principal se eligio despues de observar ese efecto.

Guarda 'modelo_tipo_falla_v3.joblib' (entrenado con todos los datos) y
'output/clasificador_tipo_falla_v3/resultados.json'. No modifica otros modelos ni datos.
Uso:  python clasificador_tipo_falla_v3.py
"""

from pathlib import Path
import contextlib
import io
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import train_model as tm
import entrenar_modelo_v3 as e3
import probar_modelo_industrial_syn as pi


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "clasificador_tipo_falla_v3"
MODELO = ROOT / "modelo_tipo_falla_v3.joblib"
CLASES = ["normal", "Desajuste de presión de corte", "Desalineación del troquel", "Sobrecarga del motor principal"]
VENTANA_FALLA = 600          # 10 h de turno antes de la falla
EQUIV_INDUSTRIAL = {"bearing": "Desalineación del troquel", "motor_overheat": "Sobrecarga del motor principal",
                    "electrical": "Sobrecarga del motor principal", "hydraulic": "Desajuste de presión de corte"}


def nuevo_modelo():
    return XGBClassifier(n_estimators=180, max_depth=6, learning_rate=0.08, subsample=0.85,
                         colsample_bytree=0.85, n_jobs=-1, random_state=42, objective="multi:softprob")


def preparar_v3():
    df, fallas = e3.cargar_v3()
    with contextlib.redirect_stdout(io.StringIO()):
        df, idx = tm.cruzar_marcas_y_calcular_rul(df, fallas)
        df, features = tm.generar_variables_ventana_movil(df)
    n = len(df)
    filas = np.arange(n)
    # Causa de cada punto de falla: la del primer evento de la bitacora que cae en ese punto.
    fila_evento = [int(np.argmin(np.abs(df["dt"].values - t))) for t in fallas["dt_falla"].values]
    causa_por_fila, evento_por_fila = {}, {}
    for (_, f), fila in zip(fallas.iterrows(), fila_evento):
        causa_por_fila.setdefault(fila, f["causa"])
        evento_por_fila.setdefault(fila, f["id_evento"])
    pos = np.searchsorted(idx, filas, side="left")
    conocida = pos < len(idx)
    filas_hasta = np.full(n, np.inf)
    filas_hasta[conocida] = idx[pos[conocida]] - filas[conocida]
    etiqueta = np.full(n, None, dtype=object)
    cerca = filas_hasta < VENTANA_FALLA
    etiqueta[cerca] = [causa_por_fila[int(idx[p])] for p in pos[cerca]]
    lejos = filas_hasta >= tm.MAX_RUL_DIAS * tm.MUESTRAS_POR_TURNO_DIA
    # Tras la ultima falla solo es "normal" si quedan >= 6000 tomas sin falla conocida.
    lejos &= conocida | (n - 1 - filas >= tm.MAX_RUL_DIAS * tm.MUESTRAS_POR_TURNO_DIA)
    etiqueta[lejos] = "normal"
    ciclo = np.where(conocida, pos, len(idx))      # ciclo = indice de la siguiente falla
    falla_del_ciclo = np.array([evento_por_fila.get(int(idx[p]), "") if p < len(idx) else "" for p in pos])
    return df, features, etiqueta, ciclo, falla_del_ciclo, fallas


def metricas_por_toma(y, yp):
    p, r, f, s = precision_recall_fscore_support(y, yp, labels=CLASES, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y, yp)),
        "accuracy_balanceada": float(balanced_accuracy_score(y, yp)),
        "f1_macro": float(np.mean(f)),
        "por_clase": {c: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "tomas": int(s[i])}
                      for i, c in enumerate(CLASES)},
        "matriz_confusion": {"filas_real_columnas_predicho": CLASES,
                             "valores": confusion_matrix(y, yp, labels=CLASES).tolist()},
    }


def causa_por_falla(pred_codigos, clases_modelo):
    """Causa mayoritaria entre las tomas que el modelo NO clasifica como normal."""
    nombres = np.array(clases_modelo)[pred_codigos]
    no_normal = nombres[nombres != "normal"]
    if len(no_normal) == 0:
        return "normal (no detecta falla)"
    v, c = np.unique(no_normal, return_counts=True)
    return str(v[np.argmax(c)])


def validacion_cruzada(df, features, etiqueta, ciclo, falla_del_ciclo):
    usar = etiqueta != None  # noqa: E711
    X = df.loc[usar, features].to_numpy()
    y = etiqueta[usar].astype(str)
    g = ciclo[usar]
    cod = {c: i for i, c in enumerate(CLASES)}
    yc = np.array([cod[v] for v in y])
    pred = np.empty(len(y), dtype=int)
    eventos = []
    ciclos_con_falla = sorted(set(g[y != "normal"]))
    for c in sorted(set(g)):
        test = g == c
        modelo = nuevo_modelo()
        modelo.fit(X[~test], yc[~test], sample_weight=compute_sample_weight("balanced", yc[~test]))
        pred[test] = modelo.predict(X[test])
        if c in ciclos_con_falla:
            m = test & (y != "normal")
            real = y[m][0]
            predicha = causa_por_falla(pred[m], CLASES)
            eventos.append({"falla": falla_del_ciclo[usar][m][0], "causa_real": real, "causa_predicha": predicha,
                            "acierto": real == predicha,
                            "pct_tomas_10h_con_causa_correcta": float((pred[m] == cod[real]).mean() * 100)})
    yp = np.array(CLASES)[pred]
    return metricas_por_toma(y, yp), pd.DataFrame(eventos)


def benchmark_cnc(modelo, ref, variante="trasladada"):
    lec = pd.read_csv(pi.descargar_si_falta(), parse_dates=["timestamp"])
    lec = lec[lec["machine_type"] == "CNC"]
    d = pi.por_minuto(lec).sort_values(["machine_id", "dt"]).reset_index(drop=True)
    base = pi.base_por_tipo(lec)
    sensores = pi.trasladar(d, base, ref, variante)
    pred = np.empty(len(d), dtype=int)
    for _, ix in d.groupby("machine_id").groups.items():
        s = sensores.loc[ix].copy()
        s.insert(0, "dt", d.loc[ix, "dt"])
        with contextlib.redirect_stdout(io.StringIO()):
            s, feats = tm.generar_variables_ventana_movil(s.reset_index(drop=True))
        pred[d.index.get_indexer(ix)] = modelo.predict(s[feats].to_numpy())
    real_rul = d["rul_hours"].to_numpy()
    ev = d["evaluable"].to_numpy()
    al = pi.alertas_por_falla(d, np.zeros(len(d)), lec)          # solo para obtener la lista de fallas
    y, yp, eventos = [], [], []
    for _, f in al.iterrows():
        t = pd.Timestamp(f["falla"])
        x = lec[(lec["machine_id"] == f["maquina"]) & (lec["timestamp"] <= t) & (lec["timestamp"] >= t - pd.Timedelta("24h"))]
        tipos = x.loc[x["failure_type"] != "none", "failure_type"]
        if not len(tipos):
            continue
        real = EQUIV_INDUSTRIAL[tipos.mode().iloc[0]]
        m = ev & (d["machine_id"] == f["maquina"]).to_numpy() & (d["dt"] < t).to_numpy() & (d["dt"] >= t - pd.Timedelta("10h")).to_numpy()
        if m.sum() == 0:
            continue
        y += [real] * int(m.sum())
        yp += list(np.array(CLASES)[pred[m]])
        eventos.append({"maquina": int(f["maquina"]), "falla": str(t), "tipo_original": tipos.mode().iloc[0],
                        "causa_equivalente": real, "causa_predicha": causa_por_falla(pred[m], CLASES)})
    lejos = ev & (real_rul >= 48)
    y += ["normal"] * int(lejos.sum())
    yp += list(np.array(CLASES)[pred[lejos]])
    ev_df = pd.DataFrame(eventos)
    ev_df["acierto"] = ev_df["causa_equivalente"] == ev_df["causa_predicha"]
    return metricas_por_toma(np.array(y), np.array(yp)), ev_df


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df, features, etiqueta, ciclo, falla_del_ciclo, fallas = preparar_v3()
    print("Tomas etiquetadas:", pd.Series(etiqueta[etiqueta != None]).value_counts().to_dict())  # noqa: E711
    print("-> Validación cruzada dejando fuera un ciclo de falla cada vez...")
    cv, ev_cv = validacion_cruzada(df, features, etiqueta, ciclo, falla_del_ciclo)

    usar = etiqueta != None  # noqa: E711
    cod = {c: i for i, c in enumerate(CLASES)}
    yc = np.array([cod[v] for v in etiqueta[usar].astype(str)])
    modelo = nuevo_modelo()
    modelo.fit(df.loc[usar, features].to_numpy(), yc, sample_weight=compute_sample_weight("balanced", yc))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sano = df[df["rul_dias"] >= tm.MAX_RUL_DIAS]
    ref = {s: (float(sano[s].mean()), float(sano[s].std())) for s in tm.SENSORES}
    joblib.dump({"modelo": modelo, "clases": CLASES, "feature_cols": features, "ventana_falla_tomas": VENTANA_FALLA,
                 "referencia_normal": ref}, MODELO)

    print("-> Benchmark Industrial syn (CNC)...")
    bench, ev_b = benchmark_cnc(modelo, ref, "trasladada")
    bench_d, ev_bd = benchmark_cnc(modelo, ref, "directa")

    resultados = {"validacion_cruzada": {**cv, "fallas_con_causa_correcta": f"{int(ev_cv['acierto'].sum())} de {len(ev_cv)}"},
                  "benchmark_industrial_syn_cnc_trasladada": {**bench, "fallas_con_causa_correcta": f"{int(ev_b['acierto'].sum())} de {len(ev_b)}"},
                  "benchmark_industrial_syn_cnc_directa": {**bench_d, "fallas_con_causa_correcta": f"{int(ev_bd['acierto'].sum())} de {len(ev_bd)}"}}
    (OUT / "resultados.json").write_text(json.dumps(resultados, ensure_ascii=False, indent=2), encoding="utf-8")
    ev_cv.to_csv(OUT / "causa_por_falla_validacion.csv", index=False, encoding="utf-8-sig")
    ev_b.to_csv(OUT / "causa_por_falla_benchmark_cnc.csv", index=False, encoding="utf-8-sig")

    for nombre, r, ev in [("VALIDACIÓN CRUZADA (fallas reales 2024)", cv, ev_cv),
                          ("BENCHMARK INDUSTRIAL SYN (CNC) · sensores trasladados", bench, ev_b),
                          ("BENCHMARK INDUSTRIAL SYN (CNC) · temperatura y vibración originales", bench_d, ev_bd)]:
        print(f"\n===== {nombre}")
        print(f"Accuracy {r['accuracy']:.3f} · Accuracy balanceada {r['accuracy_balanceada']:.3f} · F1 macro {r['f1_macro']:.3f}")
        print(pd.DataFrame(r["por_clase"]).T.round(3).to_string())
        print("Matriz de confusión (filas = real, columnas = predicho):")
        print(pd.DataFrame(r["matriz_confusion"]["valores"], index=CLASES, columns=["normal", "presión", "desalineación", "sobrecarga"]).to_string())
        print(f"Fallas con causa correcta: {int(ev['acierto'].sum())} de {len(ev)}")
        print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
