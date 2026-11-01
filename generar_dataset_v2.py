"""
generar_dataset_v2.py
=====================
Construye 'telemetria_troqueladora_v2.csv': la telemetria de 2025 con las MISMAS columnas,
marcas de tiempo, rangos normales y fechas de falla (log_fallas_mantenimiento.csv), pero con
el COMPORTAMIENTO de los sensores tomado del dataset publico "Industrial Machine Predictive
Maintenance (syn)" (Kaggle, CC0): fluctuaciones normales y forma del desgaste antes de cada falla.

Procedimiento:
1. Industrial syn se interpola a 1 toma por minuto (igual que probar_modelo_industrial_syn.py).
2. Cada sensor se traslada a la escala de la troqueladora:
       x' = media_normal_troqueladora + (x - media_normal_tipo) / desv_normal_tipo * desv_normal_troqueladora
   (normal = lecturas con RUL > 48 h; troqueladora = filas de 2025 con RUL = 10).
3. Para cada falla de la bitacora se toma un segmento de desgaste de Industrial syn que termina en
   una falla de tipo equivalente, y se coloca de forma que termine exactamente en la fecha de la falla.
   El resto de filas se completa con fragmentos de operacion normal (RUL > 48 h) del mismo tipo de maquina.
4. Para que la validacion sea independiente, 4 maquinas fuente (una por tipo) se reservan para el
   ultimo 20 % de 2025 (el tramo de validacion de train_model.py) y no se usan en el resto.

No modifica los archivos originales. Uso:  python generar_dataset_v2.py
"""

from pathlib import Path
import contextlib
import io
import json

import numpy as np
import pandas as pd

import train_model as tm
import probar_modelo_industrial_syn as pi
from probar_modelo_metropt3 import referencia_troqueladora


ROOT = Path(__file__).resolve().parent
SALIDA = ROOT / "telemetria_troqueladora_v2.csv"
SALIDA_FUENTES = ROOT / "output" / "dataset_v2" / "fuentes_por_falla.csv"
SALIDA_RESUMEN = ROOT / "output" / "dataset_v2" / "resumen.json"
SEMILLA = 42
MIN_SEGMENTO_MIN = 24 * 60        # segmentos de desgaste de al menos 24 h
FRAGMENTO_NORMAL_MIN = (300, 1200)
TIPOS_EQUIVALENTES = {
    "Sobrecarga Electrica y Friccion Excesiva": ["electrical", "motor_overheat"],
    "Falla Critica por Desbalance y Vibracion en Rodamiento": ["bearing"],
    "Fatiga Mecanica y Vibracion en Bancada": ["bearing"],
}


def minutos_fuente():
    lecturas = pd.read_csv(pi.descargar_si_falta(), parse_dates=["timestamp"])
    df = pi.por_minuto(lecturas).sort_values(["machine_id", "dt"]).reset_index(drop=True)
    corte = (df["machine_id"] != df["machine_id"].shift()) | ~df["evaluable"] | \
            (df["dt"].diff() != pd.Timedelta("1min"))
    df["tramo"] = corte.cumsum()
    return lecturas, df


def trasladar_a_troqueladora(df, lecturas, ref):
    base = pi.base_por_tipo(lecturas)
    out = pd.DataFrame(index=df.index)
    for destino, origen in pi.EQUIVALENCIAS.items():
        mu = df["machine_type"].map(lambda t: base[t][origen][0])
        sd = df["machine_type"].map(lambda t: base[t][origen][1])
        out[destino] = ref[destino][0] + (df[origen] - mu) / sd * ref[destino][1]
    return out[tm.SENSORES].to_numpy()


def segmentos_de_falla(df, lecturas):
    """Tramos evaluables continuos que terminan en una falla, con el tipo de falla etiquetado."""
    ev = df[df["evaluable"]]
    tramos = ev.groupby("tramo").agg(maquina=("machine_id", "first"), tipo_maquina=("machine_type", "first"),
                                     inicio=("dt", "min"), fin=("dt", "max"), n=("dt", "size"),
                                     rul_fin=("rul_hours", "last"))
    tramos = tramos[tramos["rul_fin"] < 1.0].copy()
    tipos = []
    for _, r in tramos.iterrows():
        x = lecturas[(lecturas["machine_id"] == r["maquina"]) &
                     (lecturas["timestamp"] >= r["fin"] - pd.Timedelta("3h")) &
                     (lecturas["timestamp"] <= r["fin"] + pd.Timedelta("20min"))]
        etiquetas = x.loc[x["failure_type"] != "none", "failure_type"]
        tipos.append(etiquetas.iloc[-1] if len(etiquetas) else "sin_etiqueta")
    tramos["tipo_falla"] = tipos
    return tramos[tramos["n"] >= MIN_SEGMENTO_MIN].reset_index()


def elegir_maquinas_reservadas(segmentos, fallas_validacion, rng):
    """Una maquina por tipo cuyos segmentos cubran los tipos de falla del tramo de validacion."""
    necesidad = {}
    for tipo in fallas_validacion:
        clave = tuple(TIPOS_EQUIVALENTES[tipo])
        necesidad[clave] = necesidad.get(clave, 0) + 1
    por_tipo = {t: sorted(g["maquina"].unique()) for t, g in segmentos.groupby("tipo_maquina")}
    for _ in range(5000):
        elegidas = [int(rng.choice(m)) for m in por_tipo.values()]
        s = segmentos[segmentos["maquina"].isin(elegidas)]
        if all(s["tipo_falla"].isin(list(k)).sum() >= n for k, n in necesidad.items()):
            return sorted(elegidas)
    raise RuntimeError("No hay combinacion de maquinas reservadas que cubra la validacion.")


class Rellenador:
    """Concatena fragmentos aleatorios de operacion normal (RUL > 48 h) de un conjunto de maquinas."""

    def __init__(self, df, valores, maquinas, rng):
        normal = df["evaluable"] & (df["rul_hours"] > pi.HORAS_BASE_NORMAL) & df["machine_id"].isin(maquinas)
        g = df[normal].groupby("tramo")
        self.tramos = {}
        for _, idx in g.groups.items():
            idx = np.asarray(idx)
            if len(idx) >= FRAGMENTO_NORMAL_MIN[0]:
                tipo = df.at[idx[0], "machine_type"]
                self.tramos.setdefault(tipo, []).append(idx)
        self.valores = valores
        self.rng = rng

    def tomar(self, n, tipo):
        partes, faltan = [], n
        opciones = self.tramos[tipo]
        while faltan > 0:
            idx = opciones[self.rng.integers(len(opciones))]
            largo = min(faltan, int(self.rng.integers(*FRAGMENTO_NORMAL_MIN)), len(idx))
            ini = int(self.rng.integers(0, len(idx) - largo + 1))
            partes.append(self.valores[idx[ini:ini + largo]])
            faltan -= largo
        return np.vstack(partes)


def main():
    rng = np.random.default_rng(SEMILLA)
    with contextlib.redirect_stdout(io.StringIO()):
        tel, fallas = tm.cargar_datos(str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS))
    n = len(tel)
    split = int(n * 0.80)
    filas_falla = [int(tel.index[tel["dt"] == t][0]) for t in fallas["dt_falla"]]
    tipos_falla = fallas["tipo_falla"].tolist()

    print("-> Preparando Industrial syn...")
    lecturas, df = minutos_fuente()
    ref = referencia_troqueladora()
    valores = trasladar_a_troqueladora(df, lecturas, ref)
    segmentos = segmentos_de_falla(df, lecturas)
    segmentos = segmentos[segmentos["tipo_falla"].isin(sum(TIPOS_EQUIVALENTES.values(), []))]

    validacion = [t for f, t in zip(filas_falla, tipos_falla) if f >= split]
    reservadas = elegir_maquinas_reservadas(segmentos, validacion, rng)
    entrenamiento_maq = sorted(set(df["machine_id"]) - set(reservadas))
    relleno = {"entrenamiento": Rellenador(df, valores, entrenamiento_maq, rng),
               "validacion": Rellenador(df, valores, reservadas, rng)}
    disponibles = {"entrenamiento": segmentos[~segmentos["maquina"].isin(reservadas)],
                   "validacion": segmentos[segmentos["maquina"].isin(reservadas)]}
    usados = {"entrenamiento": set(), "validacion": set()}

    salida = np.full((n, len(tm.SENSORES)), np.nan)
    registro = []
    inicio_ciclo = 0
    for fila, tipo, (_, f) in zip(filas_falla, tipos_falla, fallas.iterrows()):
        conjunto = "validacion" if fila >= split else "entrenamiento"
        cand = disponibles[conjunto][disponibles[conjunto]["tipo_falla"].isin(TIPOS_EQUIVALENTES[tipo])]
        libres = cand[~cand.index.isin(usados[conjunto])]
        if len(libres) == 0:            # se agotaron: se reutilizan (solo dentro del mismo conjunto)
            usados[conjunto] -= set(cand.index)
            libres = cand
        seg = libres.loc[rng.choice(libres.index)]
        usados[conjunto].add(seg.name)
        idx_seg = df.index[(df["tramo"] == seg["tramo"]) & df["evaluable"]].to_numpy()

        largo_ciclo = fila - inicio_ciclo + 1
        tomar = min(len(idx_seg), largo_ciclo)
        salida[fila - tomar + 1:fila + 1] = valores[idx_seg[-tomar:]]
        hueco_ini, hueco_fin = inicio_ciclo, fila - tomar + 1   # filas [hueco_ini, hueco_fin) por rellenar
        for a, b, conj in [(hueco_ini, min(hueco_fin, split), "entrenamiento"),
                           (max(hueco_ini, split), hueco_fin, "validacion")]:
            if b > a:
                salida[a:b] = relleno[conj].tomar(b - a, seg["tipo_maquina"])
        registro.append({
            "id_parada": f["id_parada"], "timestamp_falla": f["timestamp_falla"], "tipo_falla": tipo,
            "conjunto": conjunto, "fuente_maquina": int(seg["maquina"]),
            "fuente_tipo_maquina": seg["tipo_maquina"], "fuente_tipo_falla": seg["tipo_falla"],
            "fuente_inicio": str(seg["fin"] - pd.Timedelta(minutes=tomar - 1)), "fuente_fin": str(seg["fin"]),
            "minutos_de_desgaste_tomados": int(tomar), "filas_de_operacion_normal_antes": int(hueco_fin - hueco_ini),
        })
        inicio_ciclo = fila + 1
    # Despues de la ultima falla: operacion normal hasta el final del dataset
    for a, b, conj in [(inicio_ciclo, min(n, split), "entrenamiento"), (max(inicio_ciclo, split), n, "validacion")]:
        if b > a:
            salida[a:b] = relleno[conj].tomar(b - a, rng.choice(list(relleno[conj].tramos)))
    if np.isnan(salida).any():
        raise RuntimeError("Quedaron filas sin completar.")

    nuevo = pd.DataFrame(np.round(salida, 2), columns=tm.SENSORES)
    nuevo.insert(0, "timestamp", tel["timestamp"].to_numpy())
    nuevo.to_csv(SALIDA, index=False)

    SALIDA_FUENTES.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(registro).to_csv(SALIDA_FUENTES, index=False, encoding="utf-8-sig")

    with contextlib.redirect_stdout(io.StringIO()):
        d2, f2 = tm.cargar_datos(str(SALIDA), str(ROOT / tm.FILE_FALLAS))
        d2, _ = tm.cruzar_marcas_y_calcular_rul(d2, f2)
        d1, f1 = tm.cargar_datos(str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS))
        d1, _ = tm.cruzar_marcas_y_calcular_rul(d1, f1)

    def perfil(d):
        sano, ultimo = d["rul_dias"] >= 10, d["rul_dias"] <= 1
        return {s: {"normal_media": float(d.loc[sano, s].mean()), "normal_desv": float(d.loc[sano, s].std()),
                    "ultimo_dia_media": float(d.loc[ultimo, s].mean()),
                    "ultimo_dia_en_desviaciones": float((d.loc[ultimo, s].mean() - ref[s][0]) / ref[s][1]),
                    "min": float(d[s].min()), "max": float(d[s].max())} for s in tm.SENSORES}

    resumen = {
        "archivo": SALIDA.name, "filas": n, "semilla": SEMILLA,
        "fuente": "Industrial Machine Predictive Maintenance (syn), Kaggle, CC0",
        "maquinas_fuente_reservadas_para_validacion": reservadas,
        "maquinas_fuente_entrenamiento": entrenamiento_maq,
        "segmentos_de_desgaste_disponibles": int(len(segmentos)),
        "segmentos_reutilizados_en_entrenamiento": int(pd.DataFrame(registro).query("conjunto=='entrenamiento'")
                                                       .duplicated(["fuente_maquina", "fuente_fin"]).sum()),
        "perfil_original": perfil(d1), "perfil_v2": perfil(d2),
    }
    SALIDA_RESUMEN.write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    print(pd.DataFrame(registro)[["id_parada", "conjunto", "fuente_maquina", "fuente_tipo_maquina",
                                  "fuente_tipo_falla", "minutos_de_desgaste_tomados"]].to_string(index=False))
    print(json.dumps({k: resumen[k] for k in list(resumen)[:7]}, ensure_ascii=False))
    for s in tm.SENSORES:
        o, v = resumen["perfil_original"][s], resumen["perfil_v2"][s]
        print(f"{s:24s} normal {o['normal_media']:.2f}±{o['normal_desv']:.2f} -> {v['normal_media']:.2f}±{v['normal_desv']:.2f} | "
              f"ultimo dia {o['ultimo_dia_en_desviaciones']:+.1f} desv -> {v['ultimo_dia_en_desviaciones']:+.1f} desv | "
              f"rango [{o['min']:.1f}, {o['max']:.1f}] -> [{v['min']:.1f}, {v['max']:.1f}]")
    graficar_comparacion(d1, d2, fallas)
    print(f"\n[OK] {SALIDA.name} generado ({n:,} filas).")


def graficar_comparacion(original, v2, fallas):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, b = pd.Timestamp("2025-04-01"), pd.Timestamp("2025-05-01")
    o = original[(original["dt"] >= a) & (original["dt"] < b)].reset_index(drop=True)
    n = v2[(v2["dt"] >= a) & (v2["dt"] < b)].reset_index(drop=True)
    x = np.arange(len(o)) / tm.MUESTRAS_POR_TURNO_DIA
    fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
    for ax, s in zip(axes, tm.SENSORES):
        ax.plot(x, o[s], lw=0.5, color="#0d9488", alpha=0.8, label="original")
        ax.plot(x, n[s], lw=0.6, color="#7c3aed", label="v2 (comportamiento de Industrial syn)")
        for f in fallas["dt_falla"]:
            if a <= f < b:
                ax.axvline((o["dt"] <= f).sum() / tm.MUESTRAS_POR_TURNO_DIA, color="red", ls=":", lw=1)
        ax.set_ylabel(s, fontsize=9)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper left", fontsize=9)
    axes[0].set_title("Abril de 2025, toma a toma · líneas rojas: fallas de la bitácora (las mismas en ambos)")
    axes[-1].set_xlabel("Días de turno desde el 1 de abril")
    fig.tight_layout()
    fig.savefig(SALIDA_RESUMEN.parent / "original_vs_v2_abril.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
