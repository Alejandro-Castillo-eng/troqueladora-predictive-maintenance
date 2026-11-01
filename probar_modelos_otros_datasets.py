"""
probar_modelos_otros_datasets.py
================================
Prueba el modelo original (modelo_rul.joblib) y el modelo v2 (modelo_rul_v2.joblib), SIN
reentrenarlos, con dos datasets REALES de laboratorio que tienen sensores equivalentes a los
cuatro del modelo. Ninguno trae RUL, asi que se mide si el RUL predicho BAJA cuando el equipo
esta peor (no se puede calcular el error en horas).

1. SKAB (Skoltech Anomaly Benchmark, GitHub waico/SKAB, ~4 MB): banco con bomba de agua;
   34 experimentos de ~20 min con una anomalia provocada cada uno (cierre de valvulas,
   desbalance del rotor, cavitacion, fugas...). Temperatura, presion, vibracion y corriente.
2. Condition Monitoring of Hydraulic Systems (UCI, ZeMA, CC BY 4.0, ~76 MB comprimido):
   banco hidraulico; 2205 ciclos de 60 s con cuatro componentes degradados por niveles.
   Presion, temperatura, vibracion (mm/s) y potencia del motor (en lugar de corriente).

Los datos se descargan en 'datos_externos/' si no existen (no se suben a git).
Resultados en 'output/prueba_otros_datasets/'.  Uso:  python probar_modelos_otros_datasets.py
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
from sklearn.metrics import roc_auc_score

import train_model as tm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "prueba_otros_datasets"
MODELOS = {"original": ("modelo_rul.joblib", "telemetria_troqueladora.csv"),
           "v2": ("modelo_rul_v2.joblib", "telemetria_troqueladora_v2.csv")}
NOMBRES = {"original": "Modelo original", "v2": "Modelo v2"}
COLORES = {"original": "#0d9488", "v2": "#7c3aed"}
UMBRAL_ADVERTENCIA, UMBRAL_CRITICO = 4.0, 1.0

SKAB_DIR = ROOT / "datos_externos" / "skab"
SKAB_URL = "https://raw.githubusercontent.com/waico/SKAB/master/data/"
SKAB_ARCHIVOS = (["anomaly-free/anomaly-free.csv"] + [f"valve1/{i}.csv" for i in range(16)] +
                 [f"valve2/{i}.csv" for i in range(4)] + [f"other/{i}.csv" for i in range(1, 15)])
SKAB_EQUIV = {"temp_motor_c": "Temperature", "presion_hidraulica_bar": "Pressure",
              "vibracion_rms_mms": "Accelerometer1RMS", "corriente_motor_a": "Current"}
SKAB_ESCALAS = {"minuto": "1 fila = 1 minuto (escala real)",
                "segundo": "1 fila = 1 segundo (escala comprimida)"}

HYD_DIR = ROOT / "datos_externos" / "hydraulic"
HYD_URL = "https://archive.ics.uci.edu/static/public/447/condition+monitoring+of+hydraulic+systems.zip"
HYD_EQUIV = {"temp_motor_c": "TS1", "presion_hidraulica_bar": "PS1",
             "vibracion_rms_mms": "VS1", "corriente_motor_a": "EPS1"}
# Severidad de cada componente: 0 = optimo, 1 = cerca de la falla total
SEVERIDAD = {
    "enfriador": ("cooler", {100: 0.0, 20: 0.5, 3: 1.0}),
    "valvula": ("valve", {100: 0.0, 90: 1 / 3, 80: 2 / 3, 73: 1.0}),
    "fuga_bomba": ("pump", {0: 0.0, 1: 0.5, 2: 1.0}),
    "acumulador": ("acc", {130: 0.0, 115: 1 / 3, 100: 2 / 3, 90: 1.0}),
}
SEVERIDAD_BASE = 0.5      # ciclos con severidad total <= 0.5 = referencia de operacion normal
SEVERIDAD_ALTA = 2.0      # ciclos con severidad total >= 2 = equipo muy degradado


# ---------------------------------------------------------------- utilidades comunes
def cargar_modelos():
    out = {}
    for k, (modelo, datos) in MODELOS.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bundle = joblib.load(ROOT / modelo)
        with contextlib.redirect_stdout(io.StringIO()):
            df, fallas = tm.cargar_datos(str(ROOT / datos), str(ROOT / tm.FILE_FALLAS))
            df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas)
        sano = df[df["rul_dias"] >= tm.MAX_RUL_DIAS]
        ref = {s: (float(sano[s].mean()), float(sano[s].std())) for s in tm.SENSORES}
        out[k] = (bundle, ref)
    return out


def trasladar(valores, media, desv, ref):
    """valores: DataFrame con columnas = sensores del modelo, ya en la unidad del dataset."""
    return pd.DataFrame({s: ref[s][0] + (valores[s] - media[s]) / desv[s] * ref[s][1] for s in tm.SENSORES})


def predecir(bundle, sensores):
    d = sensores.reset_index(drop=True).copy()
    with contextlib.redirect_stdout(io.StringIO()):
        d, features = tm.generar_variables_ventana_movil(d)
    return np.clip(bundle["modelo"].predict(d[features]), 0, bundle["max_rul_dias"])


def auc_menor_rul(pred, malo):
    malo = np.asarray(malo, dtype=bool)
    if malo.all() or not malo.any():
        return None
    return float(roc_auc_score(malo.astype(int), -np.asarray(pred)))


# ---------------------------------------------------------------- SKAB
def descargar_skab():
    for rel in SKAB_ARCHIVOS:
        destino = SKAB_DIR / rel
        if not destino.exists():
            destino.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(SKAB_URL + rel, destino)


def leer_skab(rel, escala):
    x = pd.read_csv(SKAB_DIR / rel, sep=";", parse_dates=["datetime"]).sort_values("datetime")
    if "anomaly" not in x:
        x["anomaly"] = 0.0
    x = x.set_index("datetime")[list(SKAB_EQUIV.values()) + ["anomaly"]]
    if escala == "minuto":
        x = x.resample("1min").mean().dropna()
    x["anomalia"] = x["anomaly"] >= 0.5
    return x.rename(columns={v: k for k, v in SKAB_EQUIV.items()}).reset_index()


def probar_skab(modelos):
    descargar_skab()
    filas, ejemplos = [], {}
    for escala in SKAB_ESCALAS:
        libre = leer_skab("anomaly-free/anomaly-free.csv", escala)
        desv = libre[tm.SENSORES].std()
        for rel in SKAB_ARCHIVOS[1:]:
            x = leer_skab(rel, escala)
            primera = x.index[x["anomalia"]].min() if x["anomalia"].any() else len(x)
            normal_previo = x.iloc[:primera]
            # Base: media de la parte normal previa del propio experimento (elimina diferencias
            # entre dias) y desviacion del experimento sin anomalias (mas estable).
            media = normal_previo[tm.SENSORES].mean()
            for k, (bundle, ref) in modelos.items():
                pred = predecir(bundle, trasladar(x[tm.SENSORES], media, desv, ref))
                x[f"pred_{k}"] = pred
            x["experimento"], x["grupo"], x["escala"] = rel, rel.split("/")[0], escala
            x["parte"] = np.where(x["anomalia"], "anomalia", np.where(x.index < primera, "normal_previa", "normal_posterior"))
            filas.append(x)
            if escala == "segundo" and rel in ["valve1/0.csv", "valve2/0.csv", "other/9.csv", "other/14.csv"]:
                ejemplos[rel] = x
    datos = pd.concat(filas, ignore_index=True)

    resultados = {}
    for escala in SKAB_ESCALAS:
        e = datos[datos["escala"] == escala]
        for k in modelos:
            p = e[f"pred_{k}"].to_numpy()
            an, nor = e["anomalia"].to_numpy(), (e["parte"] == "normal_previa").to_numpy()
            sel = an | nor
            por_exp = e.groupby("experimento").apply(
                lambda g: pd.Series({"baja": g.loc[g["anomalia"], f"pred_{k}"].mean() <
                                     g.loc[g["parte"] == "normal_previa", f"pred_{k}"].mean(),
                                     "auc": auc_menor_rul(g.loc[g["anomalia"] | (g["parte"] == "normal_previa"), f"pred_{k}"],
                                                          g.loc[g["anomalia"] | (g["parte"] == "normal_previa"), "anomalia"])}),
                include_groups=False)
            resultados[f"{k}__{escala}"] = {
                "filas_normales": int(nor.sum()), "filas_anomalia": int(an.sum()),
                "rul_medio_normal": float(p[nor].mean()), "rul_medio_anomalia": float(p[an].mean()),
                "auc_anomalia_vs_normal": auc_menor_rul(p[sel], an[sel]),
                "auc_mediana_por_experimento": float(por_exp["auc"].dropna().median()),
                "experimentos_donde_baja_el_rul": f"{int(por_exp['baja'].sum())} de {len(por_exp)}",
                "pct_normal_en_advertencia": float((p[nor] <= UMBRAL_ADVERTENCIA).mean() * 100),
                "pct_anomalia_en_advertencia": float((p[an] <= UMBRAL_ADVERTENCIA).mean() * 100),
                "pct_normal_en_critico": float((p[nor] <= UMBRAL_CRITICO).mean() * 100),
                "pct_anomalia_en_critico": float((p[an] <= UMBRAL_CRITICO).mean() * 100),
            }

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titulos = {"valve1/0.csv": "valve1/0 · cierre de válvula de entrada", "valve2/0.csv": "valve2/0 · cierre de válvula de salida",
               "other/9.csv": "other/9 · desbalance del rotor (exponencial)", "other/14.csv": "other/14 · agua a mayor temperatura"}
    for ax, (rel, x) in zip(axes.ravel(), ejemplos.items()):
        t = (x["datetime"] - x["datetime"].min()).dt.total_seconds() / 60
        for k in modelos:
            ax.plot(t, pd.Series(x[f"pred_{k}"]).rolling(15, min_periods=1).mean(), color=COLORES[k], lw=1.2, label=NOMBRES[k])
        ini = t[x["anomalia"]]
        if len(ini):
            ax.axvspan(ini.min(), ini.max(), color="red", alpha=0.15, label="anomalía")
        ax.axhline(UMBRAL_ADVERTENCIA, color="gray", ls=":", lw=0.8)
        ax.axhline(UMBRAL_CRITICO, color="gray", ls="--", lw=0.8)
        ax.set_ylim(-0.2, 10.3)
        ax.set_title(titulos[rel], fontsize=10)
        ax.set_xlabel("minutos desde el inicio del experimento")
        ax.grid(alpha=0.2)
    axes[0, 0].set_ylabel("RUL predicho")
    axes[1, 0].set_ylabel("RUL predicho")
    axes[0, 0].legend(fontsize=8, loc="lower left")
    fig.suptitle("SKAB (escala comprimida: 1 fila = 1 s) · ¿baja el RUL predicho durante la anomalía?")
    fig.tight_layout()
    fig.savefig(OUT / "skab_ejemplos.png", dpi=150)
    plt.close(fig)
    return resultados, datos


# ---------------------------------------------------------------- Hydraulic Systems
def descargar_hidraulico():
    necesarios = [f"{v}.txt" for v in HYD_EQUIV.values()] + ["profile.txt"]
    if all((HYD_DIR / f).exists() for f in necesarios):
        return
    HYD_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = HYD_DIR / "hydraulic.zip"
    print(f"-> Descargando Hydraulic Systems desde UCI (~76 MB): {HYD_URL}")
    urllib.request.urlretrieve(HYD_URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        for f in necesarios + ["description.txt"]:
            z.extract(f, HYD_DIR)
    zip_path.unlink()


def probar_hidraulico(modelos):
    descargar_hidraulico()
    valores = pd.DataFrame({s: np.loadtxt(HYD_DIR / f"{c}.txt").mean(axis=1) for s, c in HYD_EQUIV.items()})
    perfil = pd.DataFrame(np.loadtxt(HYD_DIR / "profile.txt"), columns=["cooler", "valve", "pump", "acc", "stable"])
    for nombre, (col, mapa) in SEVERIDAD.items():
        perfil[nombre] = perfil[col].map(mapa)
    perfil["severidad_total"] = perfil[list(SEVERIDAD)].sum(axis=1)
    estable = perfil["stable"] == 0
    base = estable & (perfil["severidad_total"] <= SEVERIDAD_BASE)
    media, desv = valores[base].mean(), valores[base].std()

    resultados, preds = {}, {}
    for k, (bundle, ref) in modelos.items():
        p = predecir(bundle, trasladar(valores, media, desv, ref))
        preds[k] = p
        sev = perfil["severidad_total"].to_numpy()
        alta, baja = estable & (perfil["severidad_total"] >= SEVERIDAD_ALTA), base
        sel = (alta | baja).to_numpy()
        por_comp = {}
        for nombre, (col, mapa) in SEVERIDAD.items():
            por_comp[nombre] = {str(nivel): float(p[(estable & (perfil[col] == nivel)).to_numpy()].mean())
                                for nivel in sorted(mapa, key=lambda n: mapa[n])}
        resultados[k] = {
            "ciclos_estables": int(estable.sum()), "ciclos_base": int(base.sum()), "ciclos_severidad_alta": int(alta.sum()),
            "spearman_rul_vs_severidad": float(pd.Series(p[estable]).corr(pd.Series(sev[estable]), method="spearman")),
            "auc_severidad_alta_vs_base": auc_menor_rul(p[sel], alta.to_numpy()[sel]),
            "rul_medio_base": float(p[base.to_numpy()].mean()), "rul_medio_severidad_alta": float(p[alta.to_numpy()].mean()),
            "pct_base_en_advertencia": float((p[base.to_numpy()] <= UMBRAL_ADVERTENCIA).mean() * 100),
            "pct_alta_en_advertencia": float((p[alta.to_numpy()] <= UMBRAL_ADVERTENCIA).mean() * 100),
            "rul_medio_por_nivel_de_cada_componente": por_comp,
        }

    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    etiquetas = {"enfriador": "Enfriador (% eficiencia)", "valvula": "Válvula (% conmutación)",
                 "fuga_bomba": "Fuga interna de la bomba", "acumulador": "Acumulador (bar)"}
    for ax, (nombre, (col, mapa)) in zip(axes, SEVERIDAD.items()):
        niveles = sorted(mapa, key=lambda n: mapa[n])
        x = np.arange(len(niveles))
        for i, k in enumerate(modelos):
            v = [resultados[k]["rul_medio_por_nivel_de_cada_componente"][nombre][str(n)] for n in niveles]
            ax.bar(x + (i - 0.5) * 0.38, v, 0.38, color=COLORES[k], label=NOMBRES[k])
        ax.set_xticks(x, [str(int(n)) for n in niveles])
        ax.set_xlabel(f"{etiquetas[nombre]}\n← óptimo · cerca de falla total →", fontsize=9)
        ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("RUL predicho medio")
    axes[0].legend(fontsize=8)
    fig.suptitle("Hydraulic Systems · RUL predicho según el estado de cada componente (debería bajar hacia la derecha)")
    fig.tight_layout()
    fig.savefig(OUT / "hidraulico_por_componente.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax2 = ax.twinx()
    ax2.fill_between(np.arange(len(perfil)), perfil["severidad_total"], color="red", alpha=0.12, step="mid")
    ax2.set_ylabel("Severidad total (0 = óptimo, 4 = todo cerca de fallar)", color="#b91c1c")
    for k in modelos:
        ax.plot(pd.Series(preds[k]).rolling(10, min_periods=1).mean(), color=COLORES[k], lw=1.1, label=NOMBRES[k])
    ax.set_xlabel("Ciclo de 60 s (orden en que se registraron)")
    ax.set_ylabel("RUL predicho (media de 10 ciclos)")
    ax.set_ylim(-0.2, 10.3)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("Hydraulic Systems · RUL predicho a lo largo de los 2205 ciclos")
    fig.tight_layout()
    fig.savefig(OUT / "hidraulico_ciclos.png", dpi=150)
    plt.close(fig)
    rangos = {s: {"min": float(valores[s].min()), "mediana": float(valores[s].median()), "max": float(valores[s].max())}
              for s in tm.SENSORES}
    return resultados, rangos, {s: {"media_base": float(media[s]), "desv_base": float(desv[s])} for s in tm.SENSORES}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    modelos = cargar_modelos()
    print("-> SKAB...")
    skab, _ = probar_skab(modelos)
    print("-> Hydraulic Systems...")
    hyd, hyd_rangos, hyd_base = probar_hidraulico(modelos)
    resumen = {
        "skab": {"equivalencias": SKAB_EQUIV, "escalas": SKAB_ESCALAS, "resultados": skab,
                 "metodo": ("Cada experimento por separado. Media base = parte normal previa a la anomalia del propio "
                            "experimento; desviacion base = experimento sin anomalias. Anomalia por minuto si >= 50 % "
                            "de sus segundos son anomalos. AUC: probabilidad de que una toma anomala tenga menos RUL "
                            "que una normal previa (0.5 = azar).")},
        "hidraulico": {"equivalencias": {**HYD_EQUIV, "nota": "EPS1 es potencia del motor (W), usada en lugar de corriente"},
                       "rangos_originales_por_ciclo": hyd_rangos, "base_normal": hyd_base, "resultados": hyd,
                       "metodo": ("Media por ciclo de 60 s; 1 ciclo = 1 fila. Base normal = ciclos estables con severidad "
                                  "total <= 0.5. Severidad de cada componente 0 (optimo) a 1 (cerca de la falla total); "
                                  "total 0-4. Solo se evaluan ciclos estables.")},
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    filas = [{"dataset": "SKAB", "escala": c.split("__")[1], "modelo": c.split("__")[0], **{k: v for k, v in r.items()}}
             for c, r in skab.items()]
    filas += [{"dataset": "Hydraulic", "modelo": k, **{a: b for a, b in r.items() if a != "rul_medio_por_nivel_de_cada_componente"}}
              for k, r in hyd.items()]
    pd.DataFrame(filas).to_csv(OUT / "tabla_resultados.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(skab, ensure_ascii=False, indent=1))
    print(json.dumps(hyd, ensure_ascii=False, indent=1))
    print(json.dumps(hyd_rangos, ensure_ascii=False))
    print(f"\n[OK] Resultados en {OUT}")


if __name__ == "__main__":
    main()
