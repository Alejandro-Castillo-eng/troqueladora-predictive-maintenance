"""
comparar_modelos_v3.py
======================
Compara, con las mismas metricas y sin reentrenar, los tres modelos:

- original: modelo_rul.joblib    (telemetria_troqueladora.csv, fallas simuladas 2025)
- v2:       modelo_rul_v2.joblib (telemetria v2 con comportamiento de Industrial syn)
- v3:       modelo_rul_v3.joblib (telemetria 2024 v3 sobre las 27 fallas reales de log_fallas_2024.csv)

Pruebas: validacion propia de cada modelo (ultimo 20 %), MetroPT-3 (real), Industrial syn (todas
las maquinas y solo CNC), SKAB (real) y Hydraulic Systems (real). Para el v2, Industrial syn NO es
independiente (se uso para construir sus datos).

Resultados en 'output/comparacion_v3/'.  Uso:  python comparar_modelos_v3.py
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
import comparar_modelos_v1_v2 as cmp
import probar_modelo_metropt3 as pm
import probar_modelo_industrial_syn as pi
import probar_modelos_otros_datasets as po
import entrenar_modelo_v3 as e3
from probar_modelo_metropt3 import metricas


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "comparacion_v3"
NOMBRES = {"original": "Modelo original", "v2": "Modelo v2", "v3": "Modelo v3 (fallas reales 2024)"}
COLORES = {"original": "#0d9488", "v2": "#7c3aed", "v3": "#ea580c"}


def preparar():
    modelos = {}
    for k in ["original", "v2"]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(ROOT / cmp.MODELOS[k])
        df, idx, _ = cmp.preparar_troqueladora(k)
        modelos[k] = {"bundle": bundle, "df": df, "idx": idx, "ref": cmp.referencia_normal(df)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bundle = joblib.load(e3.MODELO_V3)
    df, fallas = e3.cargar_v3()
    with contextlib.redirect_stdout(io.StringIO()):
        df, idx = tm.cruzar_marcas_y_calcular_rul(df, fallas)
        df, _ = tm.generar_variables_ventana_movil(df)
    modelos["v3"] = {"bundle": bundle, "df": df, "idx": idx, "ref": cmp.referencia_normal(df)}
    return modelos


def evaluar_industrial(bundle, ref, lecturas):
    df = pi.por_minuto(lecturas)
    base = pi.base_por_tipo(lecturas)
    pred = pi.predecir(df, pi.trasladar(df, base, ref, "directa"), bundle)
    ev = df["evaluable"].to_numpy()
    real = df["rul_hours"].to_numpy() / pi.HORAS_POR_UNIDAD
    al = pi.alertas_por_falla(df, pred, lecturas)
    r, p = real[ev], pred[ev]
    return {
        "auc_ultimas_10h_vs_lejos": pi.auc(p, r, 1.0),
        "auc_ultimas_24h_vs_lejos": pi.auc(p, r, 2.4),
        "fallas_con_alerta_critica_previa": f"{int(al['anticipacion_alerta_critica_horas'].notna().sum())} de {len(al)}",
        "mae_horas": metricas(r, p)["mae_horas"],
        "mae_horas_referencia_promedio": metricas(r, np.full_like(r, r.mean()))["mae_horas"],
        "pct_lejos_en_critico": float((p[r >= pi.LEJOS] <= pi.UMBRAL_CRITICO).mean() * 100),
        "pct_lejos_en_advertencia": float((p[r >= pi.LEJOS] <= pi.UMBRAL_ADVERTENCIA).mean() * 100),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    modelos = preparar()
    filas = []

    def agregar(prueba, modelo, auc, alertas, extra, independiente=True):
        filas.append({"prueba": prueba, "modelo": modelo, "auc": auc, "alertas_criticas": alertas,
                      "independiente": independiente, **extra})

    print("-> Validaciones propias...")
    for k, m in modelos.items():
        r = cmp.evaluar_validacion(m["bundle"], m["df"], m["idx"])
        agregar("Validación propia (último 20 %)", k, r["auc_ultimo_dia_vs_normal"], r["fallas_con_alerta_critica_previa"], r)

    print("-> MetroPT-3...")
    minutos = pm.cargar_por_minuto(pm.descargar_si_falta())
    for variante in ["fija", "movil"]:
        for k, m in modelos.items():
            r, _ = cmp.evaluar_metropt(m["bundle"], m["ref"], minutos, variante)
            agregar(f"MetroPT-3 (real) · base {variante}", k, r["auc_ultimo_dia_vs_normal"], r["fallas_con_alerta_critica_previa"], r)

    print("-> Industrial syn...")
    lecturas = pd.read_csv(pi.descargar_si_falta(), parse_dates=["timestamp"])
    for nombre, sub in [("Industrial syn · 20 máquinas", lecturas),
                        ("Industrial syn · 5 CNC", lecturas[lecturas["machine_type"] == "CNC"])]:
        for k, m in modelos.items():
            r = evaluar_industrial(m["bundle"], m["ref"], sub)
            agregar(nombre, k, r["auc_ultimas_10h_vs_lejos"], r["fallas_con_alerta_critica_previa"], r,
                    independiente=(k != "v2"))

    print("-> SKAB y Hydraulic Systems...")
    po.OUT = OUT / "skab_hidraulico"
    po.OUT.mkdir(parents=True, exist_ok=True)
    po.COLORES.update(COLORES)
    po.NOMBRES.update(NOMBRES)
    entrada = {k: (m["bundle"], m["ref"]) for k, m in modelos.items()}
    skab, _ = po.probar_skab(entrada)
    for escala in po.SKAB_ESCALAS:
        for k in modelos:
            r = skab[f"{k}__{escala}"]
            agregar(f"SKAB (real) · {escala}", k, r["auc_anomalia_vs_normal"], "—", r)
    hyd, _, _ = po.probar_hidraulico(entrada)
    for k in modelos:
        r = {a: b for a, b in hyd[k].items()}
        agregar("Hydraulic Systems (real)", k, r["auc_severidad_alta_vs_base"], "—",
                {"spearman_rul_vs_severidad": r["spearman_rul_vs_severidad"],
                 "rul_medio_por_nivel_enfriador": r["rul_medio_por_nivel_de_cada_componente"]["enfriador"],
                 "rul_medio_por_nivel_valvula": r["rul_medio_por_nivel_de_cada_componente"]["valvula"],
                 "rul_medio_por_nivel_fuga_bomba": r["rul_medio_por_nivel_de_cada_componente"]["fuga_bomba"],
                 "rul_medio_por_nivel_acumulador": r["rul_medio_por_nivel_de_cada_componente"]["acumulador"]})

    tabla = pd.DataFrame(filas)
    tabla.to_json(OUT / "resultados.json", orient="records", force_ascii=False, indent=2)
    resumen = tabla.pivot_table(index="prueba", columns="modelo", values="auc", aggfunc="first", sort=False).round(3)
    alertas = tabla.pivot_table(index="prueba", columns="modelo", values="alertas_criticas", aggfunc="first", sort=False)
    resumen.to_csv(OUT / "auc_por_prueba.csv", encoding="utf-8-sig")
    alertas.to_csv(OUT / "alertas_por_prueba.csv", encoding="utf-8-sig")
    importancias = {k: dict(pd.Series(m["bundle"]["modelo"].feature_importances_, index=m["bundle"]["feature_cols"])
                            .sort_values(ascending=False).head(6).round(4).astype(float)) for k, m in modelos.items()}
    (OUT / "importancias.json").write_text(json.dumps(importancias, ensure_ascii=False, indent=2), encoding="utf-8")

    pruebas = list(resumen.index)
    fig, ax = plt.subplots(figsize=(15, 6))
    x = np.arange(len(pruebas))
    w = 0.27
    for i, k in enumerate(modelos):
        v = resumen[k].to_numpy(dtype=float)
        ax.bar(x + (i - 1) * w, v, w, color=COLORES[k], label=NOMBRES[k])
        for xi, vi in zip(x, v):
            ax.text(xi + (i - 1) * w, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7.5)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.text(len(pruebas) - 0.5, 0.51, "azar", color="gray", ha="right", fontsize=9)
    ax.set_xticks(x, [p.replace(" · ", "\n") for p in pruebas], fontsize=8.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("AUC (0,5 = azar · 1 = distingue siempre)")
    ax.set_title("¿Distingue el modelo cuándo está cerca la falla? · comparación de los tres modelos")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "auc_por_prueba.png", dpi=150)
    plt.close(fig)

    with pd.option_context("display.width", 200):
        print(resumen)
        print(alertas)
    print(json.dumps(importancias, ensure_ascii=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
