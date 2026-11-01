"""
probar_modelo_industrial_syn.py
===============================
Prueba el modelo guardado (modelo_rul.joblib) con el dataset publico
"Industrial Machine Predictive Maintenance (syn)" (Kaggle, Tatheer Abbas, CC0).

- NO reentrena el modelo ni modifica los datos del proyecto.
- Descarga el dataset en 'datos_externos/industrial_syn/' si no existe (no se sube a git).
- 20 maquinas simuladas (CNC, bomba, compresor, brazo robotico), 14 dias, con las cuatro
  variables del modelo y un RUL en horas.
- Guarda resultados en 'output/prueba_industrial_syn/'.

Uso:  python probar_modelo_industrial_syn.py
"""

from pathlib import Path
import contextlib
import io
import json
import urllib.request
import warnings
import zipfile

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_model as tm
from probar_modelo_metropt3 import (
    capacidad_de_distinguir, magnitud_fallas_troqueladora, metricas, por_intervalo,
    referencia_troqueladora,
)


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datos_externos" / "industrial_syn"
OUT = ROOT / "output" / "prueba_industrial_syn"
URL = "https://www.kaggle.com/api/v1/datasets/download/tatheerabbas/industrial-machine-predictive-maintenance"
CSV_NAME = "predictive_maintenance_v3.csv"

# Sensor de la troqueladora <- columna del dataset
EQUIVALENCIAS = {
    "temp_motor_c": "temperature_motor",
    "presion_hidraulica_bar": "pressure_level",
    "vibracion_rms_mms": "vibration_rms",
    "corriente_motor_a": "current_phase_avg",
}
# En la variante 'directa' estos sensores se usan con su valor original (ya estan en rangos parecidos).
SIN_TRASLADAR_EN_DIRECTA = ["temp_motor_c", "vibracion_rms_mms"]
HORAS_BASE_NORMAL = 48      # lecturas con RUL > 48 h = operacion lejos de la falla (base de cada tipo)
MAX_HUECO_MIN = 60          # no se interpola entre lecturas separadas mas de 60 min
RUL_RELLENO = 0.5           # el dataset deja el RUL fijo en 0.5 h despues de la falla
HORAS_POR_UNIDAD = 10.0     # 1 unidad de RUL del modelo = 600 tomas de 1 min = 10 h
LEJOS = 4.8                 # >= 48 h de la falla: zona lejana para AUC y falsas alarmas
UMBRAL_ADVERTENCIA = 4.0
UMBRAL_CRITICO = 1.0
MIN_ALERTA_SOSTENIDA = 30
VARIANTES = {"directa": "temperatura y vibración originales", "trasladada": "4 sensores trasladados"}
TIPOS = {"CNC": "CNC", "Compressor": "Compresor", "Pump": "Bomba", "Robotic Arm": "Brazo robótico"}


def descargar_si_falta():
    csv_path = DATA_DIR / CSV_NAME
    if csv_path.exists():
        return csv_path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "dataset.zip"
    print(f"-> Descargando dataset desde Kaggle: {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        z.extract(CSV_NAME, DATA_DIR)
    zip_path.unlink()
    return csv_path


def por_minuto(lecturas):
    """
    Interpola cada maquina a una toma por minuto. Marca:
    - 'evaluable': falso entre lecturas con RUL de relleno (0.5 h) o con un salto de RUL hacia arriba.
    Los minutos entre lecturas separadas mas de MAX_HUECO_MIN se descartan.
    """
    cols = list(EQUIVALENCIAS.values()) + ["rul_hours"]
    partes = []
    for mid, m in lecturas.groupby("machine_id"):
        m = m.sort_values("timestamp").drop_duplicates("timestamp")
        t = m["timestamp"].to_numpy()
        rul = m["rul_hours"].to_numpy()
        hueco = np.diff(t).astype("timedelta64[s]").astype(float) / 60.0
        tramo_valido = hueco <= MAX_HUECO_MIN
        tramo_evaluable = tramo_valido & (rul[:-1] > RUL_RELLENO) & (rul[1:] > RUL_RELLENO) & (rul[1:] <= rul[:-1])

        grid = pd.date_range(m["timestamp"].min().ceil("min"), m["timestamp"].max().floor("min"), freq="1min")
        serie = m.set_index("timestamp")[cols]
        interp = serie.reindex(serie.index.union(grid)).interpolate(method="time", limit_area="inside").loc[grid]
        tramo = np.clip(np.searchsorted(t, grid.to_numpy(), side="right") - 1, 0, len(t) - 2)
        df = interp.rename_axis("dt").reset_index()
        df["machine_id"] = mid
        df["machine_type"] = m["machine_type"].iloc[0]
        df["modo"] = m["operating_mode"].to_numpy()[tramo]
        df["con_datos"] = tramo_valido[tramo]
        df["evaluable"] = tramo_evaluable[tramo]
        partes.append(df[df["con_datos"]].dropna(subset=cols))
    return pd.concat(partes, ignore_index=True)


def base_por_tipo(lecturas):
    sano = lecturas[(lecturas["rul_hours"] > HORAS_BASE_NORMAL)]
    return {tipo: {c: (float(g[c].mean()), float(g[c].std())) for c in EQUIVALENCIAS.values()}
            for tipo, g in sano.groupby("machine_type")}


def trasladar(df, base, ref, variante):
    out = pd.DataFrame(index=df.index)
    for destino, origen in EQUIVALENCIAS.items():
        if variante == "directa" and destino in SIN_TRASLADAR_EN_DIRECTA:
            out[destino] = df[origen]
            continue
        mu = df["machine_type"].map(lambda t: base[t][origen][0])
        sd = df["machine_type"].map(lambda t: base[t][origen][1])
        out[destino] = ref[destino][0] + (df[origen] - mu) / sd * ref[destino][1]
    return out[tm.SENSORES]


def predecir(df, sensores, bundle):
    pred = np.empty(len(df))
    for _, idx in df.groupby("machine_id").groups.items():
        d = sensores.loc[idx].copy()
        d.insert(0, "dt", df.loc[idx, "dt"])
        with contextlib.redirect_stdout(io.StringIO()):
            d, features = tm.generar_variables_ventana_movil(d.reset_index(drop=True))
        if features != bundle["feature_cols"]:
            raise ValueError("Las variables no coinciden con las del modelo guardado.")
        pred[df.index.get_indexer(idx)] = bundle["modelo"].predict(d[features])
    return np.clip(pred, 0, bundle["max_rul_dias"])


def spearman(a, b):
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def auc(pred, real, horizonte):
    lejos = (real >= LEJOS) & (real < 10)
    return capacidad_de_distinguir(pred, real, lejos, horizonte)


def alertas_por_falla(df, pred, lecturas):
    """Fallas = primera lectura en que el RUL llega al valor de relleno (0.5 h)."""
    g = lecturas.sort_values(["machine_id", "timestamp"]).groupby("machine_id")["rul_hours"]
    inicio = (lecturas.sort_values(["machine_id", "timestamp"])["rul_hours"] <= RUL_RELLENO) & \
             ~(g.shift() <= RUL_RELLENO)
    eventos = lecturas.sort_values(["machine_id", "timestamp"])[inicio].copy()
    # La ventana de cada falla empieza despues de la falla anterior de la misma maquina, para no
    # atribuirle alertas que corresponden al ciclo anterior.
    eventos["falla_anterior"] = eventos.groupby("machine_id")["timestamp"].shift()
    filas = []
    for _, ev in eventos.iterrows():
        desde = ev["timestamp"] - pd.Timedelta(hours=100)
        if pd.notna(ev["falla_anterior"]):
            desde = max(desde, ev["falla_anterior"])
        m = (df["machine_id"] == ev["machine_id"]) & df["evaluable"] & (df["dt"] < ev["timestamp"])
        m &= df["dt"] > desde
        p = pd.Series(pred[m.to_numpy()], index=df.loc[m, "dt"])
        if len(p) == 0:
            continue
        ult10 = p[p.index >= ev["timestamp"] - pd.Timedelta(hours=10)]
        ult40 = p[p.index >= ev["timestamp"] - pd.Timedelta(hours=40)]
        racha = (p <= UMBRAL_CRITICO).astype(int).rolling(MIN_ALERTA_SOSTENIDA).sum() >= MIN_ALERTA_SOSTENIDA
        anticipacion = None
        if racha.any():
            primera = racha.idxmax() - pd.Timedelta(minutes=MIN_ALERTA_SOSTENIDA - 1)
            anticipacion = (ev["timestamp"] - primera).total_seconds() / 3600
        filas.append({
            "maquina": int(ev["machine_id"]), "tipo": TIPOS[ev["machine_type"]],
            "falla": str(ev["timestamp"]), "tipo_falla_previa": ev["failure_type"],
            "horas_con_datos_antes": round(len(p) / 60, 1),
            "pct_critico_ultimas_10h": round(float((ult10 <= UMBRAL_CRITICO).mean() * 100), 1) if len(ult10) else None,
            "pct_advertencia_o_critico_ultimas_40h": round(float((ult40 <= UMBRAL_ADVERTENCIA).mean() * 100), 1) if len(ult40) else None,
            "rul_predicho_medio_ultimas_10h": round(float(ult10.mean()), 2) if len(ult10) else None,
            "anticipacion_alerta_critica_horas": None if anticipacion is None else round(anticipacion, 1),
        })
    return pd.DataFrame(filas)


def episodios_criticos(df, pred):
    """
    Cuenta alertas criticas sostenidas (RUL predicho <= 1 durante 30 min seguidos) segun lo lejos
    que estaba la falla real cuando empezaron. Sirve para saber si las alertas previas son aciertos.
    """
    real = df["rul_hours"].to_numpy() / HORAS_POR_UNIDAD
    conteo = {"cerca_rul_hasta_1": 0, "entre_1_y_4.8": 0, "lejos_rul_desde_4.8": 0}
    horas_evaluables = {"cerca_rul_hasta_1": 0.0, "entre_1_y_4.8": 0.0, "lejos_rul_desde_4.8": 0.0}
    for _, idx in df.groupby("machine_id").groups.items():
        pos = df.index.get_indexer(idx)
        ok = df["evaluable"].to_numpy()[pos]
        crit = pd.Series((pred[pos] <= UMBRAL_CRITICO) & ok)
        racha = crit.rolling(MIN_ALERTA_SOSTENIDA).sum().to_numpy() >= MIN_ALERTA_SOSTENIDA
        inicio = np.flatnonzero(racha & ~np.r_[False, racha[:-1]]) - (MIN_ALERTA_SOSTENIDA - 1)
        r = real[pos]
        for zona, mask in [("cerca_rul_hasta_1", r <= 1), ("entre_1_y_4.8", (r > 1) & (r < LEJOS)),
                           ("lejos_rul_desde_4.8", r >= LEJOS)]:
            horas_evaluables[zona] += float((mask & ok).sum() / 60)
        for i in inicio:
            zona = "cerca_rul_hasta_1" if r[i] <= 1 else ("lejos_rul_desde_4.8" if r[i] >= LEJOS else "entre_1_y_4.8")
            conteo[zona] += 1
    return {z: {"alertas": conteo[z], "horas_evaluables": round(horas_evaluables[z], 1),
                "alertas_por_100_horas": round(conteo[z] / horas_evaluables[z] * 100, 3) if horas_evaluables[z] else None}
            for z in conteo}


def comparacion_2025():
    """Mismas metricas en la validacion de 2025 de la troqueladora (filas con RUL conocido < 10)."""
    path = ROOT / "output" / "evaluacion_intervalos_rul" / "predicciones_validacion_2025.csv"
    if not path.exists():
        return None, None
    v = pd.read_csv(path)
    v = v[v["rul_sin_tope_dias"] < 10]
    real, pred = v["rul_sin_tope_dias"].to_numpy(), v["rul_predicho_evaluacion_dias"].to_numpy()
    lejos = real >= LEJOS
    res = {
        "global_rul_menor_10": metricas(real, pred),
        "spearman": spearman(real, pred),
        "auc_ultimas_10h_vs_lejos": auc(pred, real, 1.0),
        "auc_ultimas_24h_vs_lejos": auc(pred, real, 2.4),
        "pct_critico_lejos": float((pred[lejos] <= UMBRAL_CRITICO).mean() * 100),
        "pct_advertencia_o_critico_lejos": float((pred[lejos] <= UMBRAL_ADVERTENCIA).mean() * 100),
    }
    return res, (real, pred)


def curva(real, pred, bins):
    s = pd.DataFrame({"r": real, "p": pred})
    s["b"] = pd.cut(s["r"], bins)
    q = s.groupby("b", observed=True)["p"].quantile([0.25, 0.5, 0.75]).unstack()
    centros = np.array([iv.mid for iv in q.index])
    return centros, q


def graficar(df, preds, intervalos, ref_2025_int, val_2025, lecturas):
    colores = {"directa": "#ea580c", "trasladada": "#7c3aed"}
    ev = df["evaluable"].to_numpy()
    real = df["rul_hours"].to_numpy() / HORAS_POR_UNIDAD
    bins = np.arange(0, 10.01, 0.5)

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.plot([0, 10], [0, 10], color="black", lw=1, ls="--", label="predicción perfecta")
    if val_2025 is not None:
        c, q = curva(*val_2025, bins)
        ax.plot(c, q[0.5], color="#0d9488", lw=2.2, label="Troqueladora · validación 2025")
        ax.fill_between(c, q[0.25], q[0.75], color="#0d9488", alpha=0.15)
    for k, col in colores.items():
        c, q = curva(real[ev], preds[k][ev], bins)
        ax.plot(c, q[0.5], color=col, lw=2.2, label=f"Industrial syn · {VARIANTES[k]}")
        ax.fill_between(c, q[0.25], q[0.75], color=col, alpha=0.15)
    ax.set_xlabel("RUL real (unidades de 10 h)")
    ax.set_ylabel("RUL predicho (mediana y rango 25–75 %)")
    ax.set_title("¿Baja el RUL predicho cuando se acerca la falla?")
    ax.set_xlim(0, 10); ax.set_ylim(0, 10.3)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "rul_predicho_vs_real.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    for ax, tipo in zip(axes, TIPOS):
        mid = int(df.loc[df["machine_type"] == tipo, "machine_id"].min())
        m = (df["machine_id"] == mid).to_numpy()
        t = df.loc[m, "dt"]
        ax.plot(t, np.where(ev[m], real[m], np.nan), color="#2563eb", lw=1.6, label="RUL real")
        for k, col in colores.items():
            p = pd.Series(preds[k][m], index=t).rolling(30, min_periods=1).mean()
            ax.plot(t, p, color=col, lw=0.9, label=f"predicho · {VARIANTES[k]} (media 30 min)")
        relleno = lecturas[(lecturas["machine_id"] == mid) & (lecturas["rul_hours"] <= RUL_RELLENO)]
        ax.scatter(relleno["timestamp"], np.zeros(len(relleno)) - 0.3, s=4, color="red", label="RUL de relleno (excluido)")
        ax.axhline(UMBRAL_CRITICO, color="gray", ls="--", lw=0.8)
        ax.axhline(UMBRAL_ADVERTENCIA, color="gray", ls=":", lw=0.8)
        ax.set_title(f"Máquina {mid} · {TIPOS[tipo]}", fontsize=11)
        ax.set_ylim(-0.6, 10.5)
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper right", ncol=2)
    axes[-1].set_xlabel("Fecha (enero de 2024)")
    fig.suptitle("Una máquina de cada tipo · RUL en unidades de 10 h · líneas grises: advertencia (4) y crítico (1)")
    fig.tight_layout()
    fig.savefig(OUT / "ejemplos_por_maquina.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(10)
    w = 0.27
    if ref_2025_int is not None:
        ax.bar(x - w, ref_2025_int["mae_horas_operativas"], w, color="#0d9488", label="Troqueladora · validación 2025")
    for i, (k, col) in enumerate(colores.items()):
        ax.bar(x + i * w, intervalos[k]["mae_horas"], w, color=col, label=f"Industrial syn · {VARIANTES[k]}")
    ax.set_xticks(x, intervalos["directa"]["intervalo_dias_operativos"])
    ax.set_xlabel("RUL real (unidades de 10 h)")
    ax.set_ylabel("Error absoluto medio (horas)")
    ax.set_title("Error por distancia a la falla")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "error_por_intervalo_comparado.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(ROOT / "modelo_rul.joblib")
    lecturas = pd.read_csv(descargar_si_falta(), parse_dates=["timestamp"])
    print("-> Interpolando a una toma por minuto...")
    df = por_minuto(lecturas)
    ref = referencia_troqueladora()
    base = base_por_tipo(lecturas)
    ev = df["evaluable"].to_numpy()
    real = df["rul_hours"].to_numpy() / HORAS_POR_UNIDAD

    val_2025, val_2025_series = comparacion_2025()
    ref_2025_path = ROOT / "output" / "evaluacion_intervalos_rul" / "errores_por_intervalo.csv"
    ref_2025_int = pd.read_csv(ref_2025_path) if ref_2025_path.exists() else None

    preds, resultados, intervalos, alertas = {}, {}, {}, {}
    for variante in VARIANTES:
        sensores = trasladar(df, base, ref, variante)
        pred = predecir(df, sensores, bundle)
        preds[variante] = pred
        r, p = real[ev], pred[ev]
        lejos = ev & (real >= LEJOS)
        por_tipo = {}
        for tipo, nombre in TIPOS.items():
            mt = ev & (df["machine_type"] == tipo).to_numpy()
            por_tipo[nombre] = {**metricas(real[mt], pred[mt]), "spearman": spearman(real[mt], pred[mt]),
                                "auc_ultimas_10h_vs_lejos": auc(pred[mt], real[mt], 1.0)}
        modo_lejos = pd.Series(pred[lejos]).groupby(df.loc[lejos, "modo"].to_numpy()).mean()
        al = alertas_por_falla(df, pred, lecturas)
        alertas[variante] = al
        intervalos[variante] = por_intervalo(real, real, pred, ev)
        resultados[variante] = {
            "global": metricas(r, p),
            "spearman": spearman(r, p),
            "auc_ultimas_10h_vs_lejos": auc(p, r, 1.0),
            "auc_ultimas_24h_vs_lejos": auc(p, r, 2.4),
            "referencia_promedio_constante": metricas(r, np.full_like(r, r.mean())),
            "pct_critico_lejos": float((pred[lejos] <= UMBRAL_CRITICO).mean() * 100),
            "pct_advertencia_o_critico_lejos": float((pred[lejos] <= UMBRAL_ADVERTENCIA).mean() * 100),
            "rul_predicho_medio_lejos_por_modo": {k: float(v) for k, v in modo_lejos.items()},
            "fallas_analizadas": int(len(al)),
            "fallas_con_alerta_critica_previa": int(al["anticipacion_alerta_critica_horas"].notna().sum()),
            "anticipacion_mediana_horas": (float(al["anticipacion_alerta_critica_horas"].median())
                                           if al["anticipacion_alerta_critica_horas"].notna().any() else None),
            "alertas_criticas_sostenidas_por_zona": episodios_criticos(df, pred),
            "por_tipo_de_maquina": por_tipo,
        }

    graficar(df, preds, intervalos, ref_2025_int, val_2025_series, lecturas)

    cerca = ev & (real <= 1)
    magn = {}
    for destino, origen in EQUIVALENCIAS.items():
        mu = df["machine_type"].map(lambda t: base[t][origen][0])
        sd = df["machine_type"].map(lambda t: base[t][origen][1])
        magn[destino] = float(((df[origen] - mu) / sd)[cerca].mean())
    rangos = {destino: {"min": float(lecturas[o].min()), "mediana": float(lecturas[o].median()),
                        "max": float(lecturas[o].max())} for destino, o in EQUIVALENCIAS.items()}

    resumen = {
        "modelo": bundle["nombre_modelo"],
        "dataset": "Industrial Machine Predictive Maintenance (syn), Kaggle, Tatheer Abbas, CC0",
        "lecturas_originales": int(len(lecturas)),
        "maquinas": int(lecturas["machine_id"].nunique()),
        "periodo": [str(lecturas["timestamp"].min()), str(lecturas["timestamp"].max())],
        "minutos_interpolados": int(len(df)),
        "minutos_evaluables": int(ev.sum()),
        "equivalencias": EQUIVALENCIAS,
        "rangos_originales_dataset": rangos,
        "referencia_troqueladora_operacion_normal": {s: {"media": m, "desv": sd} for s, (m, sd) in ref.items()},
        "base_normal_por_tipo": {t: {c: {"media": m, "desv": s} for c, (m, s) in v.items()} for t, v in base.items()},
        "magnitud_ultimas_10h_en_desviaciones_normales": {
            "industrial_syn": magn, "troqueladora_2025_ultimo_dia": magnitud_fallas_troqueladora(ref)},
        "resultados": resultados,
        "troqueladora_validacion_2025_mismas_metricas": val_2025,
        "metodo": (
            "Sin reentrenar. Cada maquina se interpola a 1 toma por minuto (sin cruzar huecos > 60 min). "
            "Variante 'directa': temperatura y vibracion originales; corriente y presion trasladadas. "
            "Variante 'trasladada': los 4 sensores trasladados con media y desviacion de las lecturas con "
            "RUL > 48 h de cada tipo de maquina. RUL real = rul_hours / 10 (1 unidad = 600 min = 10 h). "
            "Se excluyen de la evaluacion los minutos con RUL de relleno (0.5 h) y los tramos con salto de "
            "RUL hacia arriba. 'Lejos' = RUL real >= 48 h."
        ),
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    for v in VARIANTES:
        intervalos[v].to_csv(OUT / f"errores_por_intervalo_{v}.csv", index=False, encoding="utf-8-sig")
        alertas[v].to_csv(OUT / f"alertas_por_falla_{v}.csv", index=False, encoding="utf-8-sig")
    filas_tipo = [{"variante": v, "tipo": t, **m} for v in VARIANTES for t, m in resultados[v]["por_tipo_de_maquina"].items()]
    pd.DataFrame(filas_tipo).to_csv(OUT / "metricas_por_tipo.csv", index=False, encoding="utf-8-sig")

    print(json.dumps({k: resumen[k] for k in ["minutos_interpolados", "minutos_evaluables",
                                              "magnitud_ultimas_10h_en_desviaciones_normales"]}, ensure_ascii=False, indent=1))
    print("2025:", json.dumps(val_2025, ensure_ascii=False))
    for v in VARIANTES:
        x = dict(resultados[v]); tipos = x.pop("por_tipo_de_maquina")
        print(f"\n===== {v} =====")
        print(json.dumps(x, ensure_ascii=False))
        print(pd.DataFrame(tipos).T[["filas", "mae_horas", "sesgo_horas", "spearman", "auc_ultimas_10h_vs_lejos"]].round(3))
        print(intervalos[v][["intervalo_dias_operativos", "filas", "mae_horas", "sesgo_horas"]].round(2).to_string(index=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
