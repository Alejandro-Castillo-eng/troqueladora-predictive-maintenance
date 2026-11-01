"""
comparar_modelos_v1_v2.py
=========================
Compara el modelo original (modelo_rul.joblib, entrenado con telemetria_troqueladora.csv) y el
modelo v2 (modelo_rul_v2.joblib, entrenado con telemetria_troqueladora_v2.csv) con las mismas
metricas en:

1. Su propia validacion (ultimo 20 % de 2025).
2. La validacion del otro dataset (prueba cruzada).
3. MetroPT-3: datos REALES que no se usaron para construir ni entrenar ninguno de los dos.

No entrena ni modifica modelos o datos. Guarda resultados en 'output/comparacion_v1_v2/'.
Uso:  python comparar_modelos_v1_v2.py
"""

from pathlib import Path
import contextlib
import io
import json
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_model as tm
import probar_modelo_metropt3 as pm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "comparacion_v1_v2"
MODELOS = {"original": "modelo_rul.joblib", "v2": "modelo_rul_v2.joblib"}
DATOS = {"original": "telemetria_troqueladora.csv", "v2": "telemetria_troqueladora_v2.csv"}
NOMBRES = {"original": "Modelo original", "v2": "Modelo v2 (calibrado con Industrial syn)"}
COLORES = {"original": "#0d9488", "v2": "#7c3aed"}
FILAS_TOPE = int(tm.MAX_RUL_DIAS * tm.MUESTRAS_POR_TURNO_DIA)


def cargar_modelo(nombre):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(ROOT / MODELOS[nombre])


def preparar_troqueladora(nombre):
    with contextlib.redirect_stdout(io.StringIO()):
        df, fallas = tm.cargar_datos(str(ROOT / DATOS[nombre]), str(ROOT / tm.FILE_FALLAS))
        df, idx_fallas = tm.cruzar_marcas_y_calcular_rul(df, fallas)
        df, features = tm.generar_variables_ventana_movil(df)
    return df, idx_fallas, features


def referencia_normal(df):
    sano = df[df["rul_dias"] >= tm.MAX_RUL_DIAS]
    return {s: (float(sano[s].mean()), float(sano[s].std())) for s in tm.SENSORES}


def rul_sin_tope(n, idx_fallas):
    filas = np.arange(n)
    pos = np.searchsorted(idx_fallas, filas, side="left")
    conocida = pos < len(idx_fallas)
    r = np.full(n, np.nan)
    r[conocida] = (idx_fallas[pos[conocida]] - filas[conocida]) / tm.MUESTRAS_POR_TURNO_DIA
    # Tras la ultima falla: si quedan >= 6000 filas, el RUL es al menos 10
    r[~conocida & (n - 1 - filas >= FILAS_TOPE)] = np.inf
    return r


def metricas_completas(pred, sin_tope, inicio_fallas):
    """Metricas comunes. sin_tope: RUL real sin tope (inf = al menos 10, nan = desconocido)."""
    ok = ~np.isnan(sin_tope)
    real = np.minimum(sin_tope, tm.MAX_RUL_DIAS)
    normal = ok & (sin_tope >= tm.MAX_RUL_DIAS)
    st = np.where(ok, np.where(np.isinf(sin_tope), 99.0, sin_tope), -1.0)
    m = pm.metricas(real[ok], pred[ok])
    cerca3 = ok & (sin_tope <= 3)
    menor10 = ok & (sin_tope < 10)
    alertas = []
    for k, pos in enumerate(inicio_fallas):
        # La ventana de cada falla empieza despues de la falla anterior.
        a = max(0, pos - FILAS_TOPE, inicio_fallas[k - 1] + 1 if k > 0 else 0)
        tramo = pred[a:pos]
        racha = pd.Series(tramo <= pm.UMBRAL_CRITICO).rolling(pm.MIN_ALERTA_SOSTENIDA).sum() >= pm.MIN_ALERTA_SOSTENIDA
        alertas.append(bool(racha.any()))
    return {
        "filas": m["filas"],
        "mae_horas": m["mae_horas"],
        "r2": m["r2"],
        "mae_horas_rul_hasta_3_dias": pm.metricas(real[cerca3], pred[cerca3]).get("mae_horas"),
        "spearman_rul_menor_10": float(pd.Series(real[menor10]).corr(pd.Series(pred[menor10]), method="spearman")),
        "auc_ultimo_dia_vs_normal": pm.capacidad_de_distinguir(pred, st, normal, 1),
        "auc_ultimos_3_dias_vs_normal": pm.capacidad_de_distinguir(pred, st, normal, 3),
        "pct_normal_en_advertencia": float((pred[normal] <= pm.UMBRAL_ADVERTENCIA).mean() * 100),
        "pct_normal_en_critico": float((pred[normal] <= pm.UMBRAL_CRITICO).mean() * 100),
        "fallas_con_alerta_critica_previa": f"{sum(alertas)} de {len(alertas)}",
    }


def predecir(bundle, df):
    return np.clip(bundle["modelo"].predict(df[bundle["feature_cols"]]), 0, bundle["max_rul_dias"])


def evaluar_validacion(bundle, df, idx_fallas):
    split = int(len(df) * 0.8)
    pred = predecir(bundle, df)[split:]
    st = rul_sin_tope(len(df), idx_fallas)[split:]
    inicio = [p - split for p in idx_fallas if p >= split]
    return metricas_completas(pred, st, inicio)


def evaluar_metropt(bundle, ref, minutos, variante):
    t = pm.trasladar(minutos, ref, variante)
    df = t.reset_index().rename(columns={"timestamp": "dt"})
    with contextlib.redirect_stdout(io.StringIO()):
        df, features = tm.generar_variables_ventana_movil(df)
    pred = np.clip(bundle["modelo"].predict(df[features]), 0, bundle["max_rul_dias"])
    rul, sin_tope, grupo, inicio = pm.rul_referencia(minutos.index)
    st = np.where(np.isnan(rul), np.nan, np.where(grupo == "sin_falla_al_menos_10", np.inf, sin_tope))
    return metricas_completas(pred, st, list(inicio)), pred


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    bundles = {k: cargar_modelo(k) for k in MODELOS}
    datos = {k: preparar_troqueladora(k) for k in DATOS}
    refs = {k: referencia_normal(datos[k][0]) for k in DATOS}

    resultados = {}
    for modelo in MODELOS:
        for dataset in DATOS:
            df, idx, _ = datos[dataset]
            resultados[(modelo, f"validacion_{dataset}")] = evaluar_validacion(bundles[modelo], df, idx)

    print("-> MetroPT-3...")
    minutos = pm.cargar_por_minuto(pm.descargar_si_falta())
    preds_metro = {}
    for modelo in MODELOS:
        for variante in ["fija", "movil"]:
            res, pred = evaluar_metropt(bundles[modelo], refs[modelo], minutos, variante)
            resultados[(modelo, f"metropt3_base_{variante}")] = res
            preds_metro[(modelo, variante)] = pred

    tabla = pd.DataFrame([{"modelo": m, "evaluacion": e, **r} for (m, e), r in resultados.items()])
    tabla.to_csv(OUT / "tabla_comparativa.csv", index=False, encoding="utf-8-sig")
    resumen = {
        "modelos": MODELOS, "datos_entrenamiento": DATOS,
        "metricas_guardadas_en_cada_modelo": {k: b["metricas"] for k, b in bundles.items()},
        "importancia_top5": {k: dict(pd.Series(b["modelo"].feature_importances_, index=b["feature_cols"])
                                     .sort_values(ascending=False).head(5).round(4).astype(float))
                             for k, b in bundles.items()},
        "resultados": {f"{m}__{e}": r for (m, e), r in resultados.items()},
        "notas": ("Metricas calculadas igual para todos: filas con RUL desconocido (tras la ultima falla sin "
                  "6000 filas restantes, o dentro de una falla en MetroPT-3) excluidas. 'Normal' = RUL real >= 10. "
                  "AUC: 0.5 = azar, 1.0 = distingue siempre. Alerta critica = RUL predicho <= 1 durante 30 "
                  "tomas seguidas en las 6000 tomas previas a la falla."),
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")

    # Grafico 1: AUC y MAE por evaluacion
    evals = ["validacion_original", "validacion_v2", "metropt3_base_fija", "metropt3_base_movil"]
    etiquetas = ["Validación\ndatos originales", "Validación\ndatos v2", "MetroPT-3 (real)\nbase fija", "MetroPT-3 (real)\nbase móvil"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(evals))
    for i, m in enumerate(MODELOS):
        auc = [resultados[(m, e)]["auc_ultimo_dia_vs_normal"] for e in evals]
        mae = [resultados[(m, e)]["mae_horas"] for e in evals]
        axes[0].bar(x + (i - 0.5) * 0.38, auc, 0.38, color=COLORES[m], label=NOMBRES[m])
        axes[1].bar(x + (i - 0.5) * 0.38, mae, 0.38, color=COLORES[m], label=NOMBRES[m])
        for xi, v in zip(x, auc):
            axes[0].text(xi + (i - 0.5) * 0.38, v + 0.01, f"{v:.2f}", ha="center", fontsize=9)
        for xi, v in zip(x, mae):
            axes[1].text(xi + (i - 0.5) * 0.38, v + 0.5, f"{v:.0f}", ha="center", fontsize=9)
    axes[0].axhline(0.5, color="gray", ls="--", lw=1)
    axes[0].text(len(evals) - 0.5, 0.51, "azar", color="gray", ha="right", fontsize=9)
    axes[0].set_ylim(0, 1.08)
    axes[0].set_title("AUC: ¿distingue el último día antes de la falla?\n(más alto = mejor)")
    axes[1].set_title("Error medio absoluto (horas)\n(más bajo = mejor)")
    for ax in axes:
        ax.set_xticks(x, etiquetas, fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "comparacion_metricas.png", dpi=150)
    plt.close(fig)

    # Grafico 2: MetroPT-3, dias previos a cada falla, ambos modelos (base movil)
    rul, _, grupo, _ = pm.rul_referencia(minutos.index)
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, (nombre, ini, fin) in zip(axes.ravel(), pm.FALLAS):
        a, b = pd.Timestamp(ini) - pd.Timedelta(days=6), pd.Timestamp(fin) + pd.Timedelta(days=1)
        z = pd.DataFrame({"real": rul, **{m: preds_metro[(m, "movil")] for m in MODELOS}},
                         index=minutos.index).loc[a:b].resample("10min").mean()
        ax.plot(z.index, z["real"], color="#2563eb", lw=1.6, label="RUL de referencia")
        for m in MODELOS:
            ax.plot(z.index, z[m], color=COLORES[m], lw=0.9, label=NOMBRES[m])
        ax.axvspan(pd.Timestamp(ini), pd.Timestamp(fin), color="red", alpha=0.25)
        ax.axhline(pm.UMBRAL_CRITICO, color="gray", ls="--", lw=0.8)
        ax.set_title(nombre, fontsize=11)
        ax.set_ylim(-0.3, 10.5)
        ax.tick_params(axis="x", labelsize=8, rotation=20)
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("MetroPT-3 (datos reales, base móvil): seis días antes de cada falla")
    fig.tight_layout()
    fig.savefig(OUT / "metropt3_zoom_fallas.png", dpi=150)
    plt.close(fig)

    cols = ["mae_horas", "r2", "mae_horas_rul_hasta_3_dias", "spearman_rul_menor_10", "auc_ultimo_dia_vs_normal",
            "auc_ultimos_3_dias_vs_normal", "pct_normal_en_advertencia", "pct_normal_en_critico",
            "fallas_con_alerta_critica_previa"]
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(tabla.set_index(["evaluacion", "modelo"]).sort_index()[cols].round(3))
    print(json.dumps(resumen["importancia_top5"], ensure_ascii=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
