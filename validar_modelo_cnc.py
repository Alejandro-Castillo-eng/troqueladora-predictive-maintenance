"""
validar_modelo_cnc.py
=====================
Validacion externa del modelo original (modelo_rul.joblib) SOLO con las 5 maquinas CNC del
dataset publico "Industrial Machine Predictive Maintenance (syn)" (Kaggle, CC0).

- Sin reentrenar. Misma preparacion que probar_modelo_industrial_syn.py (1 toma por minuto,
  base normal de las CNC con RUL > 48 h, minutos con RUL de relleno excluidos).
- Variante principal 'directa': temperatura y vibracion con sus valores originales; corriente y
  presion trasladadas a la escala de la troqueladora.
- Incluye intervalo de confianza del 95 % (bootstrap por ciclo de falla) y detalle por maquina y
  por tipo de falla.

Resultados en 'output/validacion_cnc/'.  Uso:  python validar_modelo_cnc.py
"""

from pathlib import Path
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import probar_modelo_industrial_syn as pi
from probar_modelo_metropt3 import metricas, referencia_troqueladora


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "validacion_cnc"
TIPO = "CNC"
VARIANTE = "directa"
BOOTSTRAP = 2000
SEMILLA = 42
TIPOS_FALLA = {"bearing": "Rodamiento", "electrical": "Eléctrica", "hydraulic": "Hidráulica",
               "motor_overheat": "Sobrecalentamiento", "none": "Sin tipo indicado"}


def ciclos(df):
    """Identificador de cada cuenta regresiva (ciclo hasta una falla) dentro de cada maquina."""
    nuevo = (df["machine_id"] != df["machine_id"].shift()) | (df["rul_hours"].diff() > 0)
    return nuevo.cumsum().to_numpy()


def auc_cerca_lejos(pred, real, cerca=1.0):
    malo, bueno = real <= cerca, (real >= pi.LEJOS) & (real < 10)
    sel = malo | bueno
    if malo.sum() == 0 or bueno.sum() == 0:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(malo[sel].astype(int), -pred[sel]))


def bootstrap_auc(pred, real, ciclo, rng, cerca=1.0):
    ids = np.unique(ciclo)
    pos = {c: np.flatnonzero(ciclo == c) for c in ids}
    valores = []
    for _ in range(BOOTSTRAP):
        idx = np.concatenate([pos[c] for c in rng.choice(ids, len(ids), replace=True)])
        v = auc_cerca_lejos(pred[idx], real[idx], cerca)
        if v is not None:
            valores.append(v)
    return [float(np.percentile(valores, 2.5)), float(np.percentile(valores, 97.5))]


def wilson(k, n, z=1.96):
    p = k / n
    centro = (p + z * z / (2 * n)) / (1 + z * z / n)
    margen = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [float(centro - margen), float(centro + margen)]


def tipo_de_cada_falla(lecturas, eventos):
    tipos = []
    for _, ev in eventos.iterrows():
        x = lecturas[(lecturas["machine_id"] == ev["maquina"]) &
                     (lecturas["timestamp"] <= pd.Timestamp(ev["falla"])) &
                     (lecturas["timestamp"] >= pd.Timestamp(ev["falla"]) - pd.Timedelta("24h"))]
        etiquetas = x.loc[x["failure_type"] != "none", "failure_type"]
        tipos.append(etiquetas.mode().iloc[0] if len(etiquetas) else "none")
    return tipos


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(ROOT / "modelo_rul.joblib")
    todas = pd.read_csv(pi.descargar_si_falta(), parse_dates=["timestamp"])
    lecturas = todas[todas["machine_type"] == TIPO].copy()
    df = pi.por_minuto(lecturas).sort_values(["machine_id", "dt"]).reset_index(drop=True)
    ref = referencia_troqueladora()
    base = pi.base_por_tipo(lecturas)
    pred = pi.predecir(df, pi.trasladar(df, base, ref, VARIANTE), bundle)

    ev = df["evaluable"].to_numpy()
    real = df["rul_hours"].to_numpy() / pi.HORAS_POR_UNIDAD
    r, p, c = real[ev], pred[ev], ciclos(df)[ev]

    alertas = pi.alertas_por_falla(df, pred, lecturas)
    alertas["tipo_falla"] = [TIPOS_FALLA[t] for t in tipo_de_cada_falla(lecturas, alertas)]
    alertas["alerta_critica_previa"] = alertas["anticipacion_alerta_critica_horas"].notna()
    con_alerta, n_fallas = int(alertas["alerta_critica_previa"].sum()), len(alertas)
    zonas = pi.episodios_criticos(df, pred)

    por_maquina = []
    for mid in sorted(df["machine_id"].unique()):
        m = ev & (df["machine_id"] == mid).to_numpy()
        a = alertas[alertas["maquina"] == mid]
        por_maquina.append({"maquina": int(mid), "fallas": len(a), "con_alerta_critica": int(a["alerta_critica_previa"].sum()),
                            "auc_ultimas_10h_vs_lejos": auc_cerca_lejos(pred[m], real[m]),
                            "mae_horas": metricas(real[m], pred[m])["mae_horas"]})
    por_maquina = pd.DataFrame(por_maquina)
    por_tipo_falla = alertas.groupby("tipo_falla").agg(fallas=("falla", "size"),
                                                       con_alerta_critica=("alerta_critica_previa", "sum"),
                                                       rul_predicho_medio_ultimas_10h=("rul_predicho_medio_ultimas_10h", "mean"))

    resultados = {
        "maquinas": [int(x) for x in sorted(df["machine_id"].unique())],
        "fallas": n_fallas,
        "minutos_evaluables": int(ev.sum()),
        "auc_ultimas_10h_vs_lejos": auc_cerca_lejos(p, r, 1.0),
        "auc_ultimas_10h_ic95": bootstrap_auc(p, r, c, rng, 1.0),
        "auc_ultimas_24h_vs_lejos": auc_cerca_lejos(p, r, 2.4),
        "auc_ultimas_24h_ic95": bootstrap_auc(p, r, c, rng, 2.4),
        "fallas_con_alerta_critica": f"{con_alerta} de {n_fallas}",
        "pct_fallas_con_alerta_critica": con_alerta / n_fallas * 100,
        "pct_fallas_con_alerta_ic95": [x * 100 for x in wilson(con_alerta, n_fallas)],
        "anticipacion_alerta_horas": alertas["anticipacion_alerta_critica_horas"].describe()[["min", "50%", "max"]].round(1).to_dict(),
        "alertas_criticas_por_zona": zonas,
        "pct_lejos_en_critico": float((p[r >= pi.LEJOS] <= pi.UMBRAL_CRITICO).mean() * 100),
        "pct_lejos_en_advertencia": float((p[r >= pi.LEJOS] <= pi.UMBRAL_ADVERTENCIA).mean() * 100),
        "spearman_rul": pi.spearman(r, p),
        "mae": metricas(r, p),
        "mae_referencia_promedio_constante": metricas(r, np.full_like(r, r.mean())),
        "rul_predicho_medio_ultimas_10h": float(p[r <= 1].mean()),
        "rul_predicho_medio_lejos": float(p[r >= pi.LEJOS].mean()),
    }
    resumen = {
        "modelo": "modelo_rul.joblib (original)", "dataset": "Industrial Machine Predictive Maintenance (syn), Kaggle, CC0",
        "tipo_de_maquina": TIPO, "variante": "temperatura y vibracion originales; corriente y presion trasladadas",
        "base_normal_cnc": {k: {"media": v[0], "desv": v[1]} for k, v in base[TIPO].items()},
        "resultados": resultados,
        "seleccion": ("Se eligio CNC por ser la maquina de mecanizado industrial mas parecida a una prensa. La "
                      "eleccion se hizo despues de ver los resultados de los cuatro tipos (todas las maquinas: "
                      "AUC 0.64, 20 de 63 fallas con alerta)."),
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    alertas.to_csv(OUT / "alertas_por_falla.csv", index=False, encoding="utf-8-sig")
    por_maquina.to_csv(OUT / "metricas_por_maquina.csv", index=False, encoding="utf-8-sig")
    por_tipo_falla.to_csv(OUT / "alertas_por_tipo_de_falla.csv", encoding="utf-8-sig")

    # --- Grafico 1: las 5 CNC a lo largo del tiempo
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True)
    for ax, mid in zip(axes, sorted(df["machine_id"].unique())):
        m = (df["machine_id"] == mid).to_numpy()
        t = df.loc[m, "dt"]
        ax.plot(t, np.where(ev[m], real[m], np.nan), color="#2563eb", lw=1.6, label="RUL real")
        ax.plot(t, pd.Series(pred[m]).rolling(30, min_periods=1).mean(), color="#ea580c", lw=1.0,
                label="RUL predicho (media 30 min)")
        a = alertas[alertas["maquina"] == mid]
        for _, f in a.iterrows():
            ax.axvline(pd.Timestamp(f["falla"]), color="red", ls=":", lw=1)
            ax.text(pd.Timestamp(f["falla"]), 10.1, "✓" if f["alerta_critica_previa"] else "✗",
                    color="green" if f["alerta_critica_previa"] else "red", ha="center", fontsize=12, fontweight="bold")
        ax.axhline(pi.UMBRAL_CRITICO, color="gray", ls="--", lw=0.8)
        ax.axhline(pi.UMBRAL_ADVERTENCIA, color="gray", ls=":", lw=0.8)
        ax.set_ylim(-0.3, 10.8)
        ax.set_title(f"CNC · máquina {mid}", fontsize=11, loc="left")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")
    axes[-1].set_xlabel("Fecha (enero de 2024)")
    fig.suptitle("Las 5 máquinas CNC · líneas rojas: fallas · ✓ = alerta crítica antes de la falla · ✗ = sin alerta\n"
                 "RUL en unidades de 10 h · líneas grises: advertencia (4) y crítico (1)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "maquinas_cnc.png", dpi=140)
    plt.close(fig)

    # --- Grafico 2: RUL predicho frente a real
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.plot([0, 10], [0, 10], color="black", ls="--", lw=1, label="predicción perfecta")
    c_, q = pi.curva(r, p, np.arange(0, 10.01, 0.5))
    ax.plot(c_, q[0.5], color="#ea580c", lw=2.2, label="CNC: mediana del RUL predicho")
    ax.fill_between(c_, q[0.25], q[0.75], color="#ea580c", alpha=0.18, label="rango 25–75 %")
    ax.set_xlabel("RUL real (unidades de 10 h)")
    ax.set_ylabel("RUL predicho")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10.3)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left")
    ax.set_title("CNC · el RUL predicho baja un poco al acercarse la falla, pero no sigue la diagonal")
    fig.tight_layout()
    fig.savefig(OUT / "rul_predicho_vs_real_cnc.png", dpi=150)
    plt.close(fig)

    # --- Grafico 3: comportamiento de los sensores de las CNC antes de la falla
    S = ["temperature_motor", "vibration_rms", "current_phase_avg", "pressure_level"]
    nombres = {"temperature_motor": "Temperatura (°C)", "vibration_rms": "Vibración (mm/s)",
               "current_phase_avg": "Corriente (A)", "pressure_level": "Presión"}
    v = lecturas[lecturas["rul_hours"] > pi.RUL_RELLENO].copy()
    v["hb"] = (v["rul_hours"] // 6) * 6 + 3
    modo = lecturas[lecturas["rul_hours"] > pi.HORAS_BASE_NORMAL].groupby("operating_mode")[S].mean()
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
    for ax, s in zip(axes, S):
        ax.axhspan(modo.loc["idle", s], modo.loc["peak", s], color="gray", alpha=0.12, label="reposo–pico lejos de la falla")
        mm = v.groupby("hb")[s].agg(["mean", lambda x: x.quantile(.25), lambda x: x.quantile(.75)])
        mm.columns = ["media", "q25", "q75"]
        ax.plot(mm.index, mm["media"], color="#2563eb", lw=2, label="media")
        ax.fill_between(mm.index, mm["q25"], mm["q75"], color="#2563eb", alpha=0.15, label="rango 25–75 %")
        ax.set_xlim(98, 0)
        ax.set_title(nombres[s])
        ax.set_xlabel("Horas hasta la falla")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, loc="upper left")
    fig.suptitle("CNC · cómo cambian los sensores antes de la falla", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "sensores_cnc_antes_de_falla.png", dpi=150)
    plt.close(fig)

    print(json.dumps(resultados, ensure_ascii=False, indent=1, default=float))
    print(por_maquina.round(3).to_string(index=False))
    print(por_tipo_falla.round(2).to_string())
    print(alertas[["maquina", "falla", "tipo_falla", "rul_predicho_medio_ultimas_10h", "anticipacion_alerta_critica_horas"]].to_string(index=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
