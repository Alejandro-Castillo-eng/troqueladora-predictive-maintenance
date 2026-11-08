"""
analisis.py
===========
Analisis estadistico de los experimentos numericos (lee results/resultados_replicas.csv).

Produce en results/:
- resumen_escenarios.csv   : media ± desviacion e IC 95 % por escenario y metodo (n = 30 replicas).
- pruebas_estadisticas.csv : propuesto vs linea base por escenario (pareado por semilla, CRN):
                             Shapiro-Wilk de las diferencias -> t pareada (normal) o Wilcoxon
                             (no normal); p corregido por Holm; d de Cohen pareado; IC 95 % de la
                             diferencia.
- efecto_escenario.csv     : Kruskal-Wallis del modelo propuesto entre los 4 escenarios.
- barrido_magnitud.csv     : curva de rendimiento frente a la magnitud de la falla.
- boxplots_escenarios.png, curva_magnitud.png
Uso:  python analisis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

AQUI = Path(__file__).resolve().parent
R = AQUI / "results"
ESCENARIOS = ["tipico", "borde", "extremo", "patologico"]
NOMBRE_ESC = {"tipico": "Típico", "borde": "Borde", "extremo": "Extremo", "patologico": "Patológico"}
M_RUL, B_RUL, I_RUL = "RUL · XGBoost (propuesto)", "RUL · Ridge (línea base)", "RUL · promedio (ingenuo)"
M_TIPO, B_TIPO = "Tipo · XGBoost (propuesto)", "Tipo · reglas del tablero (línea base)"
METRICAS = {
    "rul": ["auc_1d", "auc_3d", "mae_h", "rmse_h", "r2", "precision_alerta", "recall_alerta", "f1_alerta",
            "acc_bal_alerta", "pct_normal_en_critico", "pct_fallas_con_alerta"],
    "tipo": ["accuracy", "acc_bal", "f1_macro", "f1_normal", "f1_presion", "f1_desalineacion", "f1_sobrecarga",
             "pct_fallas_causa_correcta"],
}
PRUEBAS = [("rul", M_RUL, B_RUL, "auc_1d", "mayor"), ("rul", M_RUL, B_RUL, "mae_h", "menor"),
           ("rul", M_RUL, B_RUL, "f1_alerta", "mayor"), ("tipo", M_TIPO, B_TIPO, "f1_macro", "mayor"),
           ("tipo", M_TIPO, B_TIPO, "acc_bal", "mayor"), ("tipo", M_TIPO, B_TIPO, "pct_fallas_causa_correcta", "mayor")]


def ic95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def holm(p):
    p = np.asarray(p, dtype=float)
    orden = np.argsort(p)
    m = len(p)
    ajust = np.empty(m)
    acum = 0.0
    for rank, i in enumerate(orden):
        acum = max(acum, min(1.0, (m - rank) * p[i]))
        ajust[i] = acum
    return ajust


def cargar():
    d = pd.read_csv(R / "resultados_replicas.csv")
    d["pct_fallas_con_alerta"] = d["fallas_con_alerta"] / d["fallas_test"] * 100
    d["pct_fallas_causa_correcta"] = d["fallas_causa_correcta"] / d["fallas_test"] * 100
    return d


def resumen(d):
    filas = []
    for esc in ESCENARIOS:
        for tarea, mets in METRICAS.items():
            for metodo in sorted(d.loc[(d.escenario == esc) & (d.tarea == tarea), "metodo"].unique()):
                x = d[(d.escenario == esc) & (d.metodo == metodo)]
                for m in mets:
                    lo, hi = ic95(x[m])
                    filas.append({"escenario": esc, "tarea": tarea, "metodo": metodo, "metrica": m, "n": int(x[m].notna().sum()),
                                  "media": x[m].mean(), "desv": x[m].std(ddof=1), "ic95_inf": lo, "ic95_sup": hi,
                                  "mediana": x[m].median(), "min": x[m].min(), "max": x[m].max()})
    return pd.DataFrame(filas)


def pruebas(d):
    filas = []
    for esc in ESCENARIOS:
        for tarea, prop, base, m, mejor in PRUEBAS:
            a = d[(d.escenario == esc) & (d.metodo == prop)].set_index("semilla")[m]
            b = d[(d.escenario == esc) & (d.metodo == base)].set_index("semilla")[m]
            ab = pd.concat([a, b], axis=1, keys=["p", "b"]).dropna()
            dif = ab["p"] - ab["b"]
            normal = stats.shapiro(dif).pvalue > 0.05 if dif.std() > 0 else False
            if dif.abs().sum() == 0:
                p = 1.0
                prueba = "sin diferencias"
            elif normal:
                p = stats.ttest_rel(ab["p"], ab["b"]).pvalue
                prueba = "t pareada"
            else:
                p = stats.wilcoxon(ab["p"], ab["b"]).pvalue
                prueba = "Wilcoxon"
            lo, hi = ic95(dif)
            filas.append({"escenario": esc, "tarea": tarea, "metrica": m, "propuesto": prop, "linea_base": base, "n": len(ab),
                          "media_propuesto": ab["p"].mean(), "media_linea_base": ab["b"].mean(), "diferencia": dif.mean(),
                          "dif_ic95_inf": lo, "dif_ic95_sup": hi,
                          "shapiro_p": stats.shapiro(dif).pvalue if dif.std() > 0 else np.nan, "prueba": prueba, "p": p,
                          "d_cohen": dif.mean() / dif.std(ddof=1) if dif.std() > 0 else np.nan,
                          "mejor_si": mejor,
                          "gana_propuesto": (dif.mean() > 0) if mejor == "mayor" else (dif.mean() < 0)})
    t = pd.DataFrame(filas)
    t["p_holm"] = holm(t["p"])
    t["significativo_0.05"] = t["p_holm"] < 0.05
    return t


def efecto_escenario(d):
    filas = []
    for metodo, m in [(M_RUL, "auc_1d"), (M_RUL, "mae_h"), (M_TIPO, "f1_macro")]:
        grupos = [d[(d.escenario == e) & (d.metodo == metodo)][m].dropna() for e in ESCENARIOS]
        h, p = stats.kruskal(*grupos)
        filas.append({"metodo": metodo, "metrica": m, "H": h, "p": p,
                      **{f"mediana_{e}": g.median() for e, g in zip(ESCENARIOS, grupos)}})
    return pd.DataFrame(filas)


def barrido(d):
    b = d[d.escenario.str.startswith("magnitud_")].copy()
    b["magnitud"] = b["escenario"].str.replace("magnitud_", "").astype(float)
    filas = []
    for (mag, metodo), g in b.groupby(["magnitud", "metodo"]):
        for m in ["auc_1d", "mae_h", "f1_macro", "pct_fallas_causa_correcta"]:
            if g[m].notna().any():
                lo, hi = ic95(g[m])
                filas.append({"magnitud": mag, "metodo": metodo, "metrica": m, "n": int(g[m].notna().sum()),
                              "media": g[m].mean(), "ic95_inf": lo, "ic95_sup": hi})
    return pd.DataFrame(filas)


def graficos(d, bar):
    colores = {M_RUL: "#ea580c", B_RUL: "#64748b", M_TIPO: "#7c3aed", B_TIPO: "#64748b"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (m, metodos, titulo) in zip(axes, [("auc_1d", [M_RUL, B_RUL], "Anticipación (RUL): AUC del último día"),
                                               ("f1_macro", [M_TIPO, B_TIPO], "Tipo de falla: F1 macro")]):
        pos = np.arange(len(ESCENARIOS))
        for i, met in enumerate(metodos):
            datos = [d[(d.escenario == e) & (d.metodo == met)][m].dropna() for e in ESCENARIOS]
            bp = ax.boxplot(datos, positions=pos + (i - 0.5) * 0.36, widths=0.32, patch_artist=True, showfliers=True)
            for caja in bp["boxes"]:
                caja.set_facecolor(colores[met])
                caja.set_alpha(0.75)
            ax.plot([], [], color=colores[met], lw=8, alpha=0.75, label=met.split(" · ")[1])
        if m == "auc_1d":
            ax.axhline(0.5, color="gray", ls="--", lw=1)
            ax.text(len(ESCENARIOS) - 0.5, 0.51, "azar", color="gray", ha="right", fontsize=9)
        ax.set_xticks(pos, [NOMBRE_ESC[e] for e in ESCENARIOS])
        ax.set_title(titulo + " · 30 réplicas por escenario")
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "boxplots_escenarios.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, (m, metodos, titulo) in zip(axes, [("auc_1d", [M_RUL, B_RUL], "Anticipación (RUL): AUC del último día"),
                                               ("f1_macro", [M_TIPO, B_TIPO], "Tipo de falla: F1 macro")]):
        for met in metodos:
            g = bar[(bar.metodo == met) & (bar.metrica == m)].sort_values("magnitud")
            ax.plot(g["magnitud"] * 100, g["media"], marker="o", color=colores[met], label=met.split(" · ")[1])
            ax.fill_between(g["magnitud"] * 100, g["ic95_inf"], g["ic95_sup"], color=colores[met], alpha=0.18)
        if m == "auc_1d":
            ax.axhline(0.5, color="gray", ls="--", lw=1)
        ax.set_xlabel("Magnitud de la firma de falla (% de la del dataset v3)")
        ax.set_title(titulo + " · media e IC 95 % (10 réplicas)")
        ax.invert_xaxis()
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.2)
        ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(R / "curva_magnitud.png", dpi=150)
    plt.close(fig)


def main():
    d = cargar()
    res, pr, ef, bar = resumen(d), pruebas(d), efecto_escenario(d), barrido(d)
    res.to_csv(R / "resumen_escenarios.csv", index=False, encoding="utf-8-sig")
    pr.to_csv(R / "pruebas_estadisticas.csv", index=False, encoding="utf-8-sig")
    ef.to_csv(R / "efecto_escenario.csv", index=False, encoding="utf-8-sig")
    bar.to_csv(R / "barrido_magnitud.csv", index=False, encoding="utf-8-sig")
    graficos(d, bar)
    pd.set_option("display.width", 250)
    clave = res[res.metrica.isin(["auc_1d", "mae_h", "rmse_h", "r2", "f1_alerta", "pct_normal_en_critico", "pct_fallas_con_alerta",
                                  "f1_macro", "acc_bal", "accuracy", "pct_fallas_causa_correcta"])]
    print(clave.assign(v=lambda x: x.media.round(3).astype(str) + " ± " + x.desv.round(3).astype(str))
          .pivot_table(index=["tarea", "metrica", "metodo"], columns="escenario", values="v", aggfunc="first")[ESCENARIOS].to_string())
    print(pr[["escenario", "metrica", "media_propuesto", "media_linea_base", "diferencia", "dif_ic95_inf", "dif_ic95_sup",
              "prueba", "p", "p_holm", "d_cohen", "gana_propuesto"]].round(4).to_string(index=False))
    print(ef.round(4).to_string(index=False))
    print(bar.pivot_table(index=["metodo", "metrica"], columns="magnitud", values="media").round(3).to_string())


if __name__ == "__main__":
    main()
