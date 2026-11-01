"""
probar_modelo_metropt3.py
=========================
Prueba el modelo guardado (modelo_rul.joblib) con el dataset publico MetroPT-3
(compresor de aire de un tren del metro de Oporto, UCI, licencia CC BY 4.0).

- NO reentrena el modelo ni modifica los datos del proyecto.
- Descarga MetroPT-3 en 'datos_externos/metropt3/' si no existe (no se sube a git).
- Traslada cada sensor de MetroPT-3 a la escala de la troqueladora y mide el error.
- Guarda resultados en 'output/prueba_metropt3/'.

Uso:  python probar_modelo_metropt3.py
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


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datos_externos" / "metropt3"
OUT = ROOT / "output" / "prueba_metropt3"
URL = "https://archive.ics.uci.edu/static/public/791/metropt+3+dataset.zip"
CSV_NAME = "MetroPT3(AirCompressor).csv"

# Fallas del documento oficial "Data Description_Metro.pdf" que acompana al dataset.
# El momento de falla es el inicio de cada ventana; las filas dentro de la ventana se excluyen.
FALLAS = [
    ("Falla 1 (fuga de aire)", "2020-04-18 00:00", "2020-04-18 23:59"),
    ("Falla 2 (fuga de aire)", "2020-05-29 23:30", "2020-05-30 06:00"),
    ("Falla 3 (fuga de aire)", "2020-06-05 10:00", "2020-06-07 14:30"),
    ("Falla 4 (fuga de aire)", "2020-07-15 14:30", "2020-07-15 19:00"),
]

# Sensor de la troqueladora <- sensor de MetroPT-3. La vibracion no existe en MetroPT-3.
EQUIVALENCIAS = {
    "temp_motor_c": "Oil_temperature",        # temperatura del aceite del compresor (°C)
    "presion_hidraulica_bar": "TP3",          # presion neumatica del panel (bar)
    "corriente_motor_a": "Motor_current",     # corriente de una fase del motor (A)
}
SIN_SENSOR = "vibracion_rms_mms"

PERIODO_NORMAL = ("2020-02-01", "2020-03-01")   # primer mes, antes de la primera falla
VENTANA_MOVIL = "7D"
UMBRAL_ADVERTENCIA = 4.0   # mismos umbrales de RUL que app.py
UMBRAL_CRITICO = 1.0
MIN_ALERTA_SOSTENIDA = 30  # minutos seguidos para contar una alerta critica


def descargar_si_falta():
    csv_path = DATA_DIR / CSV_NAME
    if csv_path.exists():
        return csv_path
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATA_DIR / "metropt3.zip"
    print(f"-> Descargando MetroPT-3 desde UCI (~210 MB): {URL}")
    urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if name.endswith(".csv") or name.endswith(".pdf"):
                (DATA_DIR / Path(name).name).write_bytes(z.read(name))
    zip_path.unlink()
    return csv_path


def cargar_por_minuto(csv_path):
    """Promedia las lecturas (cada 10 s) por minuto y descarta los minutos sin datos."""
    cols = ["timestamp"] + list(EQUIVALENCIAS.values())
    raw = pd.read_csv(csv_path, usecols=cols, parse_dates=["timestamp"])
    return raw.set_index("timestamp").sort_index().resample("1min").mean().dropna()


def referencia_troqueladora():
    """Media y desviacion de cada sensor en operacion normal (RUL = 10) de 2025."""
    with contextlib.redirect_stdout(io.StringIO()):
        df, fallas = tm.cargar_datos(str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS))
        df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas)
    sano = df[df["rul_dias"] >= tm.MAX_RUL_DIAS]
    return {s: (float(sano[s].mean()), float(sano[s].std())) for s in tm.SENSORES}


def mascara_fallas(index):
    mask = np.zeros(len(index), dtype=bool)
    for _, ini, fin in FALLAS:
        mask |= (index >= pd.Timestamp(ini)) & (index <= pd.Timestamp(fin))
    return mask


def trasladar(minutos, ref, variante):
    """
    Traslada cada sensor a la escala de la troqueladora:
        x' = media_troqueladora + (x - media_base) / desv_base * desv_troqueladora
    'fija': base = primer mes (febrero). 'movil': base = 7 dias previos (sin ventanas de falla).
    """
    out = pd.DataFrame(index=minutos.index)
    normal = minutos.loc[PERIODO_NORMAL[0]:PERIODO_NORMAL[1]]
    sin_fallas = minutos[~mascara_fallas(minutos.index)]
    for destino, origen in EQUIVALENCIAS.items():
        mu_fija, sd_fija = normal[origen].mean(), normal[origen].std()
        if variante == "fija":
            mu, sd = mu_fija, sd_fija
        else:
            rolling = sin_fallas[origen].rolling(VENTANA_MOVIL, closed="left", min_periods=1440)
            # Dentro de una falla se conserva la ultima base calculada antes de ella.
            mu = rolling.mean().reindex(minutos.index).ffill().fillna(mu_fija)
            sd = rolling.std().reindex(minutos.index).ffill().fillna(sd_fija)
        ref_mu, ref_sd = ref[destino]
        out[destino] = ref_mu + (minutos[origen] - mu) / sd * ref_sd
    out[SIN_SENSOR] = ref[SIN_SENSOR][0]   # constante en su valor normal
    return out[tm.SENSORES]


def rul_referencia(index):
    """RUL objetivo con la misma regla de train_model.py: filas hasta la falla / 600, tope 10."""
    n = len(index)
    filas = np.arange(n)
    inicio_fallas = np.array([index.searchsorted(pd.Timestamp(ini)) for _, ini, _ in FALLAS])
    pos = np.searchsorted(inicio_fallas, filas, side="left")
    conocida = pos < len(inicio_fallas)
    rul = np.full(n, np.nan)
    rul_sin_tope = np.full(n, np.nan)
    rul_sin_tope[conocida] = (inicio_fallas[pos[conocida]] - filas[conocida]) / tm.MUESTRAS_POR_TURNO_DIA
    rul[conocida] = np.clip(rul_sin_tope[conocida], 0, tm.MAX_RUL_DIAS)
    grupo = np.where(conocida, "con_siguiente_falla", "")
    # Despues de la ultima falla: si quedan >= 6000 filas sin falla reportada, el RUL es al menos 10.
    restantes = n - 1 - filas
    tras_ultima = ~conocida
    seguro = tras_ultima & (restantes >= tm.MAX_RUL_DIAS * tm.MUESTRAS_POR_TURNO_DIA)
    rul[seguro] = tm.MAX_RUL_DIAS
    grupo = np.where(seguro, "sin_falla_al_menos_10", grupo)
    grupo = np.where(tras_ultima & ~seguro, "desconocido_final", grupo)
    durante = mascara_fallas(index)
    grupo = np.where(durante, "durante_falla", grupo)
    rul[durante] = np.nan
    return rul, rul_sin_tope, grupo, inicio_fallas


def metricas(real, pred):
    err = pred - real
    if len(err) == 0:
        return {"filas": 0}
    ss_tot = ((real - real.mean()) ** 2).sum()
    return {
        "filas": int(len(err)),
        "mae_dias": float(np.abs(err).mean()),
        "mae_horas": float(np.abs(err).mean() * 10),
        "rmse_dias": float(np.sqrt((err ** 2).mean())),
        "sesgo_horas": float(err.mean() * 10),
        "r2": float(1 - (err ** 2).sum() / ss_tot) if ss_tot > 0 else None,
        "dentro_1_dia_pct": float((np.abs(err) <= 1).mean() * 100),
    }


def alertas(pred, rul_sin_tope, grupo, inicio_fallas, index):
    filas = []
    ventana = int(tm.MAX_RUL_DIAS * tm.MUESTRAS_POR_TURNO_DIA)
    # La ventana de cada falla empieza despues del fin de la falla anterior.
    fin_anterior = [0] + [int(index.searchsorted(pd.Timestamp(fin), side="right")) for _, _, fin in FALLAS[:-1]]
    for (nombre, ini, _), pos, desde in zip(FALLAS, inicio_fallas, fin_anterior):
        a = max(0, pos - ventana, desde)
        tramo = pred[a:pos]
        critico = tramo <= UMBRAL_CRITICO
        racha = pd.Series(critico).rolling(MIN_ALERTA_SOSTENIDA).sum().to_numpy() >= MIN_ALERTA_SOSTENIDA
        primera = np.flatnonzero(racha)
        anticipacion = None
        if len(primera):
            inicio_racha = a + primera[0] - MIN_ALERTA_SOSTENIDA + 1
            anticipacion = (pos - inicio_racha) / 60.0
        filas.append({
            "falla": nombre,
            "inicio_falla": ini,
            "horas_analizadas_antes": round((pos - a) / 60.0, 1),
            "pct_tiempo_advertencia_o_critico": round(float((tramo <= UMBRAL_ADVERTENCIA).mean() * 100), 2),
            "pct_tiempo_critico": round(float(critico.mean() * 100), 2),
            "rul_predicho_medio_ultimas_10h": round(float(pred[max(0, pos - 600):pos].mean()), 2),
            "anticipacion_alerta_critica_horas": None if anticipacion is None else round(anticipacion, 1),
        })
    normal = (grupo == "sin_falla_al_menos_10") | ((grupo == "con_siguiente_falla") & (rul_sin_tope >= 10))
    falsas = {
        "filas_normales": int(normal.sum()),
        "pct_tiempo_advertencia_o_critico": float((pred[normal] <= UMBRAL_ADVERTENCIA).mean() * 100),
        "pct_tiempo_critico": float((pred[normal] <= UMBRAL_CRITICO).mean() * 100),
    }
    return pd.DataFrame(filas), falsas


def capacidad_de_distinguir(pred, rul_sin_tope, normal, horizonte_dias):
    """
    AUC: probabilidad de que una toma a <= horizonte de la falla reciba un RUL predicho
    menor que una toma de operacion normal. 0.5 = no distingue; 1.0 = distingue siempre.
    """
    from sklearn.metrics import roc_auc_score
    cerca = (rul_sin_tope >= 0) & (rul_sin_tope <= horizonte_dias)
    sel = cerca | normal
    if cerca.sum() == 0 or normal.sum() == 0:
        return None
    return float(roc_auc_score(cerca[sel].astype(int), -pred[sel]))


def magnitud_fallas_troqueladora(ref):
    """Desviacion media (en desviaciones de operacion normal) el ultimo dia antes de cada falla de 2025."""
    with contextlib.redirect_stdout(io.StringIO()):
        df, fallas = tm.cargar_datos(str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS))
        df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas)
    ultimo = df[df["rul_dias"] <= 1]
    return {s: float(((ultimo[s] - ref[s][0]) / ref[s][1]).mean()) for s in EQUIVALENCIAS}


def magnitud_fallas_metropt(minutos, inicio_fallas):
    """Lo mismo para MetroPT-3: ultimas 600 tomas antes de cada falla y tomas dentro de la falla."""
    base = minutos.loc[PERIODO_NORMAL[0]:PERIODO_NORMAL[1]]
    previas = np.zeros(len(minutos), dtype=bool)
    for pos in inicio_fallas:
        previas[max(0, pos - tm.MUESTRAS_POR_TURNO_DIA):pos] = True
    durante = mascara_fallas(minutos.index)
    out = {}
    for destino, origen in EQUIVALENCIAS.items():
        z = (minutos[origen] - base[origen].mean()) / base[origen].std()
        out[destino] = {"ultimo_dia_antes": float(z[previas].mean()), "durante_falla": float(z[durante].mean())}
    return out


def efecto_sin_vibracion_2025(bundle, ref):
    """RUL predicho en 2025 con la vibracion real y con la vibracion constante (como en MetroPT-3)."""
    with contextlib.redirect_stdout(io.StringIO()):
        df, fallas = tm.cargar_datos(str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS))
        df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas)
    sano, ultimo = df["rul_dias"] >= tm.MAX_RUL_DIAS, df["rul_dias"] <= 1
    out = {}
    for nombre, constante in [("vibracion_real", False), ("vibracion_constante", True)]:
        d = df[["dt"] + tm.SENSORES].copy()
        if constante:
            d[SIN_SENSOR] = ref[SIN_SENSOR][0]
        with contextlib.redirect_stdout(io.StringIO()):
            d, features = tm.generar_variables_ventana_movil(d)
        p = np.clip(bundle["modelo"].predict(d[features]), 0, tm.MAX_RUL_DIAS)
        out[nombre] = {"rul_medio_operacion_normal": float(p[sano].mean()),
                       "rul_medio_ultimo_dia": float(p[ultimo].mean())}
    return out


def auc_troqueladora_2025():
    path = ROOT / "output" / "evaluacion_intervalos_rul" / "predicciones_validacion_2025.csv"
    if not path.exists():
        return None
    v = pd.read_csv(path)
    sin_tope = v["rul_sin_tope_dias"].to_numpy()
    pred = v["rul_predicho_evaluacion_dias"].to_numpy()
    normal = ~np.isnan(sin_tope) & (sin_tope >= 10)
    sin_tope = np.nan_to_num(sin_tope, nan=-1)
    return {f"hasta_{h}_dias": capacidad_de_distinguir(pred, sin_tope, normal, h) for h in (1, 3)}


def por_intervalo(real_sin_tope, real, pred, mask):
    filas = []
    for d in range(10):
        m = mask & (real_sin_tope <= d + 1) & ((real_sin_tope >= 0) if d == 0 else (real_sin_tope > d))
        if d == 9:
            m &= real_sin_tope < 10
        etiqueta = "0 a 1" if d == 0 else (">9 a <10" if d == 9 else f">{d} a {d + 1}")
        filas.append({"intervalo_dias_operativos": etiqueta, **metricas(real[m], pred[m])})
    return pd.DataFrame(filas)


def graficar(index, real, preds, minutos_trasladados, interval_tables, ref_2025):
    colores = {"fija": "#ea580c", "movil": "#7c3aed"}
    etiquetas = {"fija": "base fija (febrero)", "movil": "base móvil (7 días previos)"}
    h = pd.DataFrame({"real": real, **{k: v for k, v in preds.items()}}, index=index).resample("1h").mean()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(h.index, h["real"], color="#2563eb", lw=1.6, label="RUL de referencia")
    for k, c in colores.items():
        ax.plot(h.index, h[k], color=c, lw=0.9, alpha=0.85, label=f"RUL predicho · {etiquetas[k]}")
    for _, ini, fin in FALLAS:
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color="red", alpha=0.25)
    ax.set_ylabel("RUL (días operativos de 10 h)")
    ax.set_title("MetroPT-3 · RUL de referencia y predicho (promedio por hora) · franjas rojas: fallas")
    ax.legend(loc="lower left", fontsize=9)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "rul_real_vs_predicho.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (nombre, ini, fin) in zip(axes.ravel(), FALLAS):
        a, b = pd.Timestamp(ini) - pd.Timedelta(days=6), pd.Timestamp(fin) + pd.Timedelta(days=1)
        z = pd.DataFrame({"real": real, **preds}, index=index).loc[a:b].resample("10min").mean()
        ax.plot(z.index, z["real"], color="#2563eb", lw=1.6, label="referencia")
        for k, c in colores.items():
            ax.plot(z.index, z[k], color=c, lw=0.9, label=etiquetas[k])
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color="red", alpha=0.25)
        ax.axhline(UMBRAL_ADVERTENCIA, color="gray", ls=":", lw=1)
        ax.axhline(UMBRAL_CRITICO, color="gray", ls="--", lw=1)
        ax.set_title(nombre, fontsize=11)
        ax.set_ylim(-0.3, 10.5)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("Seis días antes de cada falla · líneas grises: umbrales de advertencia (4) y crítico (1)")
    fig.tight_layout()
    fig.savefig(OUT / "zoom_fallas.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(10)
    ancho = 0.27
    if ref_2025 is not None:
        ax.bar(x - ancho, ref_2025["mae_horas_operativas"], ancho, color="#0d9488", label="Troqueladora · validación 2025")
    for i, (k, c) in enumerate(colores.items()):
        ax.bar(x + i * ancho, interval_tables[k]["mae_horas"], ancho, color=c, label=f"MetroPT-3 · {etiquetas[k]}")
    ax.set_xticks(x, interval_tables["fija"]["intervalo_dias_operativos"])
    ax.set_xlabel("Tiempo restante de referencia (días operativos de 10 h)")
    ax.set_ylabel("Error absoluto medio (horas)")
    ax.set_title("Error por distancia a la falla: troqueladora 2025 frente a MetroPT-3")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "error_por_intervalo_comparado.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    d = minutos_trasladados.resample("D").mean()
    for ax, s in zip(axes, EQUIVALENCIAS):
        ax.plot(d.index, d[s], lw=1.2)
        ax.axhline(tm.RANGOS_NOMINALES[s]["nominal_mean"], color="gray", ls="--", lw=1)
        for _, ini, fin in FALLAS:
            ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color="red", alpha=0.25)
        ax.set_ylabel(f"{s}\n← {EQUIVALENCIAS[s]}", fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].set_title("Sensores de MetroPT-3 trasladados a la escala de la troqueladora (base fija, promedio diario)")
    fig.tight_layout()
    fig.savefig(OUT / "sensores_trasladados.png", dpi=150)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(ROOT / "modelo_rul.joblib")
    csv_path = descargar_si_falta()
    print("-> Promediando MetroPT-3 por minuto...")
    minutos = cargar_por_minuto(csv_path)
    ref = referencia_troqueladora()
    rul, rul_sin_tope, grupo, inicio_fallas = rul_referencia(minutos.index)
    evaluable = ~np.isnan(rul)

    preds, resultados, intervalos, tablas_alerta, trasladados = {}, {}, {}, {}, {}
    for variante in ["fija", "movil"]:
        t = trasladar(minutos, ref, variante)
        trasladados[variante] = t
        df = t.reset_index().rename(columns={"timestamp": "dt"})
        with contextlib.redirect_stdout(io.StringIO()):
            df, features = tm.generar_variables_ventana_movil(df)
        if features != bundle["feature_cols"]:
            raise ValueError("Las variables no coinciden con las del modelo guardado.")
        pred = np.clip(bundle["modelo"].predict(df[features]), 0, bundle["max_rul_dias"])
        preds[variante] = pred
        r, p = rul[evaluable], pred[evaluable]
        cercana = evaluable & (rul_sin_tope <= 3)
        lejana = evaluable & (rul_sin_tope > 3) & (rul_sin_tope < 10)
        tope = evaluable & ~cercana & ~lejana
        al, falsas = alertas(pred, rul_sin_tope, grupo, inicio_fallas, minutos.index)
        tablas_alerta[variante] = al
        intervalos[variante] = por_intervalo(rul_sin_tope, rul, pred, evaluable & (grupo == "con_siguiente_falla"))
        normal = evaluable & (rul >= tm.MAX_RUL_DIAS) & ~(rul_sin_tope < 10)
        sin_tope = np.nan_to_num(np.where(evaluable, rul_sin_tope, np.nan), nan=-1)
        en_febrero = (minutos.index >= PERIODO_NORMAL[0]) & (minutos.index < PERIODO_NORMAL[1])
        resultados[variante] = {
            "auc_distingue_cercania_falla": {
                f"hasta_{h}_dias": capacidad_de_distinguir(pred, sin_tope, normal, h) for h in (1, 3)
            },
            "rul_predicho_medio_operacion_normal": float(pred[normal].mean()),
            "rul_predicho_medio_febrero_periodo_base": float(pred[en_febrero].mean()),
            "rul_predicho_medio_ultimo_dia_antes_de_falla": float(pred[evaluable & (rul_sin_tope <= 1)].mean()),
            "rul_predicho_medio_durante_falla": float(pred[grupo == "durante_falla"].mean()),
            "global": metricas(r, p),
            "cercanas_hasta_3_dias": metricas(rul[cercana], pred[cercana]),
            "alejadas_mas_3_menos_10_dias": metricas(rul[lejana], pred[lejana]),
            "objetivo_10_sin_falla_cercana": metricas(rul[tope], pred[tope]),
            "referencia_siempre_10": metricas(r, np.full_like(r, 10.0)),
            "referencia_promedio_constante": metricas(r, np.full_like(r, r.mean())),
            "falsas_alarmas_en_operacion_normal": falsas,
        }

    ref_2025_path = ROOT / "output" / "evaluacion_intervalos_rul" / "errores_por_intervalo.csv"
    ref_2025 = pd.read_csv(ref_2025_path) if ref_2025_path.exists() else None
    graficar(minutos.index, np.where(evaluable, rul, np.nan), preds, trasladados["fija"], intervalos, ref_2025)

    base = minutos.loc[PERIODO_NORMAL[0]:PERIODO_NORMAL[1]]
    resumen = {
        "modelo": bundle["nombre_modelo"],
        "dataset": "MetroPT-3 (UCI, DOI 10.24432/C5VW3R, CC BY 4.0)",
        "periodo": [str(minutos.index.min()), str(minutos.index.max())],
        "minutos_con_datos": int(len(minutos)),
        "fallas": [{"nombre": n, "inicio": i, "fin": f} for n, i, f in FALLAS],
        "equivalencias": {**EQUIVALENCIAS, SIN_SENSOR: f"sin sensor: constante {ref[SIN_SENSOR][0]:.3f}"},
        "referencia_troqueladora_operacion_normal": {s: {"media": m, "desv": sd} for s, (m, sd) in ref.items()},
        "base_fija_metropt3_febrero": {o: {"media": float(base[o].mean()), "desv": float(base[o].std())}
                                       for o in EQUIVALENCIAS.values()},
        "filas_por_grupo": {g: int((grupo == g).sum()) for g in np.unique(grupo)},
        "filas_evaluables": int(evaluable.sum()),
        "resultados": resultados,
        "troqueladora_validacion_2025": {**bundle["metricas"], "auc_distingue_cercania_falla": auc_troqueladora_2025()},
        "magnitud_de_las_fallas_en_desviaciones_normales": {
            "troqueladora_2025_ultimo_dia": magnitud_fallas_troqueladora(ref),
            "metropt3": magnitud_fallas_metropt(minutos, inicio_fallas),
        },
        "efecto_sin_vibracion_en_troqueladora_2025": efecto_sin_vibracion_2025(bundle, ref),
        "metodo": (
            "Sin reentrenar. Lecturas de 10 s promediadas por minuto; minutos sin datos descartados. "
            "Cada sensor se traslada a la escala de la troqueladora con su media y desviacion en operacion "
            "normal (base fija: febrero de 2020; base movil: 7 dias previos sin ventanas de falla). "
            "Vibracion constante en su valor normal. RUL de referencia = filas hasta el inicio de la "
            "siguiente falla / 600, tope 10 (1 unidad = 10 h de datos). Filas dentro de una falla y "
            "las ultimas filas sin certeza se excluyen. Tras la ultima falla, si quedan >= 6000 filas "
            "sin falla reportada, el RUL de referencia es 10."
        ),
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    for variante in ["fija", "movil"]:
        intervalos[variante].to_csv(OUT / f"errores_por_intervalo_{variante}.csv", index=False, encoding="utf-8-sig")
        tablas_alerta[variante].to_csv(OUT / f"alertas_por_falla_{variante}.csv", index=False, encoding="utf-8-sig")
    horas = pd.DataFrame({"rul_referencia": np.where(evaluable, rul, np.nan),
                          "rul_predicho_base_fija": preds["fija"],
                          "rul_predicho_base_movil": preds["movil"],
                          **{f"{s}_trasladado": trasladados["fija"][s].to_numpy() for s in EQUIVALENCIAS}},
                         index=minutos.index).resample("1h").mean().dropna(how="all")
    horas.round(3).to_csv(OUT / "predicciones_por_hora.csv", encoding="utf-8-sig")

    for variante in ["fija", "movil"]:
        print(f"\n===== Variante: base {variante} =====")
        for k, v in resultados[variante].items():
            print(f"{k}: {json.dumps(v, ensure_ascii=False)}")
        print(intervalos[variante][["intervalo_dias_operativos", "filas", "mae_horas", "sesgo_horas"]].to_string(index=False))
        print(tablas_alerta[variante].to_string(index=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
