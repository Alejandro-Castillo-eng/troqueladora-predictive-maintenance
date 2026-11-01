"""
generar_dataset_v3.py
=====================
Genera 'telemetria_troqueladora_2024_v3.csv': telemetria SIMULADA de 2024 construida sobre las
27 FALLAS REALES de 'log_fallas_2024.csv' (fecha, causa, hora de inicio y fin de la parada).

Calendario: lunes a sabado, 08:00-17:59, 1 lectura por minuto. Entre la hora de inicio y la hora
de fin de cada falla la maquina esta parada y no hay lecturas.

Firma de cada causa (cuadro del usuario: causa -> variables relacionadas):
- Desalineacion del troquel        -> vibracion sube (hasta 2.8-4.5 mm/s, zonas C-D de ISO 10816-3).
- Desajuste de presion de corte    -> presion baja (8-18 bar por debajo de la consigna).
- Sobrecarga del motor principal   -> corriente sube (110-125 % de la nominal) y temperatura sube
                                      con retraso termico (+5 a +15 degC).
Las demas variables no cambian por la falla.

Variabilidad: duracion del desgaste, forma (lineal, exponencial o escalonada) e intensidad
aleatorias por falla. Todas las fallas tienen desgaste previo (sin fallas repentinas: segun el usuario,
estas causas se desarrollan con tendencia durante dias). Operacion normal con cambios de carga por
trabajo/troquel, calentamiento diario del motor y picos transitorios aislados.

Los parametros se basan en referencias tecnicas generales (ISO 10816-3 para vibracion, factor de
servicio tipico de motores para corriente), NO en los datasets externos usados para las pruebas.

Uso:  python generar_dataset_v3.py
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
BITACORA = ROOT / "log_fallas_2024.csv"
SALIDA = ROOT / "telemetria_troqueladora_2024_v3.csv"
OUT = ROOT / "output" / "dataset_v3"
SEMILLA = 2024
MIN_POR_DIA = 600

# Operacion normal
NOMINAL = {"temp_motor_c": 52.5, "presion_hidraulica_bar": 214.0, "vibracion_rms_mms": 1.25, "corriente_motor_a": 21.0}
CARGAS = [0.92, 0.96, 1.0, 1.0, 1.04, 1.08]     # factor de carga por trabajo/troquel
TRABAJO_MIN = (60, 240)                          # duracion de cada trabajo (min)
PROB_PICO = 0.0015                               # picos transitorios aislados por lectura

# Firmas por causa: (rango de duracion del desgaste en dias de turno, rango de magnitud)
CAUSAS = {
    "Desalineación del troquel": {"dias": (1.0, 6.0), "vib_final": (2.8, 4.5)},
    "Desajuste de presión de corte": {"dias": (0.5, 4.0), "caida_bar": (8.0, 18.0)},
    "Sobrecarga del motor principal": {"dias": (1.0, 5.0), "factor_corriente": (1.10, 1.25), "sube_temp": (5.0, 15.0)},
}
# Fallas casi sin aviso (30-180 min de desgaste y mitad de magnitud). 0: segun el usuario, en esta prensa
# las tres causas se desarrollan con tendencia durante dias.
PROB_REPENTINA = 0.0
SENSORES = ["temp_motor_c", "presion_hidraulica_bar", "vibracion_rms_mms", "corriente_motor_a"]


def cargar_bitacora():
    b = pd.read_csv(BITACORA)
    b["inicio"] = pd.to_datetime(b["fecha"] + " " + b["hora_inicio"])
    b["fin"] = pd.to_datetime(b["fecha"] + " " + b["hora_fin"])
    return b.sort_values("inicio").reset_index(drop=True)


def calendario(bitacora):
    dias = pd.date_range("2024-01-01", "2024-12-31", freq="D")
    dias = dias[dias.dayofweek < 6]                                  # lunes a sabado
    t = (dias.values[:, None] + (np.arange(MIN_POR_DIA) * 60 + 8 * 3600).astype("timedelta64[s]")).ravel()
    t = pd.DatetimeIndex(t)
    parada = np.zeros(len(t), dtype=bool)
    for _, f in bitacora.iterrows():
        parada |= (t >= f["inicio"]) & (t < f["fin"])
    return t[~parada]


def forma(p, tipo, rng):
    if tipo == "lineal":
        return p
    if tipo == "exponencial":
        k = 3.0
        return (np.exp(k * p) - 1) / (np.exp(k) - 1)
    escalones = np.sort(rng.uniform(0, 1, rng.integers(2, 5)))
    return np.searchsorted(escalones, p, side="right") / len(escalones)


def operacion_normal(t, rng):
    n = len(t)
    carga = np.empty(n)
    offset_presion = np.empty(n)
    i = 0
    while i < n:
        largo = int(rng.integers(*TRABAJO_MIN))
        carga[i:i + largo] = rng.choice(CARGAS)
        offset_presion[i:i + largo] = rng.normal(0, 1.5)
        i += largo

    def ar1(sigma, phi=0.9):
        e = rng.normal(0, sigma * np.sqrt(1 - phi ** 2), n)
        x = np.empty(n)
        x[0] = e[0]
        for k in range(1, n):
            x[k] = phi * x[k - 1] + e[k]
        return x

    minuto_del_dia = (t.hour - 8) * 60 + t.minute
    fecha = t.normalize()
    offset_dia = pd.Series(rng.normal(0, 0.7, len(fecha.unique())), index=fecha.unique()).reindex(fecha).to_numpy()
    calentamiento = -5.5 * np.exp(-minuto_del_dia / 60.0)            # el motor arranca ~5 degC mas frio
    temp = NOMINAL["temp_motor_c"] + calentamiento + 8.0 * (carga - 1) + offset_dia + ar1(0.6)
    corriente = NOMINAL["corriente_motor_a"] * carga + ar1(0.45, 0.7)
    presion = NOMINAL["presion_hidraulica_bar"] + offset_presion + 10.0 * (carga - 1) + ar1(1.2, 0.6)
    vibracion = NOMINAL["vibracion_rms_mms"] + 0.6 * (carga - 1) + ar1(0.10, 0.6)
    picos = rng.random(n) < PROB_PICO
    vibracion[picos] += rng.uniform(1.0, 2.5, picos.sum())
    picos_p = rng.random(n) < PROB_PICO / 2
    presion[picos_p] -= rng.uniform(8, 20, picos_p.sum())
    return {"temp_motor_c": temp, "presion_hidraulica_bar": presion,
            "vibracion_rms_mms": vibracion, "corriente_motor_a": corriente}, carga


def aplicar_fallas(t, valores, bitacora, rng):
    n = len(t)
    desv = {s: np.zeros(n) for s in SENSORES}
    ruido_extra = {s: np.zeros(n) for s in SENSORES}
    registro = []
    ultima_reparacion = {}      # por causa: primera fila despues de la ultima reparacion de esa causa
    for _, f in bitacora.iterrows():
        fin = int(np.searchsorted(t.values, f["inicio"].to_datetime64(), side="left"))  # primera fila de la parada o despues
        p_causa = CAUSAS[f["causa"]]
        repentina = rng.random() < PROB_REPENTINA
        largo = int(rng.uniform(30, 180)) if repentina else int(rng.uniform(*p_causa["dias"]) * MIN_POR_DIA)
        escala = 0.5 if repentina else 1.0
        tipo = "lineal" if repentina else rng.choice(["lineal", "exponencial", "escalonada"])
        # La reparacion anterior de la misma causa reinicia el desgaste.
        ini = max(0, fin - largo, min(ultima_reparacion.get(f["causa"], 0), fin - 1))
        ultima_reparacion[f["causa"]] = int(np.searchsorted(t.values, f["fin"].to_datetime64(), side="left"))
        p = np.linspace(0, 1, fin - ini, endpoint=False) + 1.0 / max(fin - ini, 1)
        g = forma(p, tipo, rng)
        detalle = {}
        if f["causa"] == "Desalineación del troquel":
            final = rng.uniform(*p_causa["vib_final"])
            delta = (final - NOMINAL["vibracion_rms_mms"]) * escala
            desv["vibracion_rms_mms"][ini:fin] = np.maximum(desv["vibracion_rms_mms"][ini:fin], delta * g)
            ruido_extra["vibracion_rms_mms"][ini:fin] = np.maximum(ruido_extra["vibracion_rms_mms"][ini:fin], 0.25 * g * escala)
            detalle = {"vibracion_final_mm_s": round(NOMINAL["vibracion_rms_mms"] + delta, 2)}
        elif f["causa"] == "Desajuste de presión de corte":
            caida = rng.uniform(*p_causa["caida_bar"]) * escala
            desv["presion_hidraulica_bar"][ini:fin] = np.minimum(desv["presion_hidraulica_bar"][ini:fin], -caida * g)
            ruido_extra["presion_hidraulica_bar"][ini:fin] = np.maximum(ruido_extra["presion_hidraulica_bar"][ini:fin], 1.5 * g * escala)
            detalle = {"caida_presion_bar": round(caida, 1)}
        else:
            factor = 1 + (rng.uniform(*p_causa["factor_corriente"]) - 1) * escala
            sube_t = rng.uniform(*p_causa["sube_temp"]) * escala
            desv["corriente_motor_a"][ini:fin] = np.maximum(desv["corriente_motor_a"][ini:fin],
                                                            NOMINAL["corriente_motor_a"] * (factor - 1) * g)
            desv["temp_motor_c"][ini:fin] = np.maximum(desv["temp_motor_c"][ini:fin], sube_t * g ** 1.5)  # retraso termico
            ruido_extra["corriente_motor_a"][ini:fin] = np.maximum(ruido_extra["corriente_motor_a"][ini:fin], 0.4 * g * escala)
            detalle = {"corriente_final_pct": round(factor * 100, 1), "subida_temperatura_c": round(sube_t, 1)}
        registro.append({"id_evento": f["id_evento"], "causa": f["causa"], "inicio_falla": str(f["inicio"]),
                         "repentina": bool(repentina), "forma": tipo, "minutos_de_desgaste": int(fin - ini),
                         "dias_de_turno_de_desgaste": round((fin - ini) / MIN_POR_DIA, 2), **detalle})
    for s in SENSORES:
        valores[s] = valores[s] + desv[s] + ruido_extra[s] * rng.normal(0, 1, n)
    return valores, pd.DataFrame(registro)


def graficar(df, registro, bitacora):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ejemplos = []
    for causa in CAUSAS:
        r = registro[(registro["causa"] == causa) & ~registro["repentina"]].sort_values("minutos_de_desgaste")
        ejemplos.append(r.iloc[len(r) // 2])
    fig, axes = plt.subplots(4, 3, figsize=(17, 11), sharex="col")
    nombres = {"temp_motor_c": "Temperatura (°C)", "presion_hidraulica_bar": "Presión (bar)",
               "vibracion_rms_mms": "Vibración (mm/s)", "corriente_motor_a": "Corriente (A)"}
    for j, ev in enumerate(ejemplos):
        t_f = pd.Timestamp(ev["inicio_falla"])
        fin = int(np.searchsorted(df["timestamp"].values, t_f.to_datetime64()))
        a = max(0, fin - int(ev["minutos_de_desgaste"]) - 3 * MIN_POR_DIA)
        z = df.iloc[a:min(len(df), fin + MIN_POR_DIA)]
        x = (np.arange(len(z)) - (fin - a)) / MIN_POR_DIA
        for i, s in enumerate(SENSORES):
            ax = axes[i, j]
            ax.plot(x, z[s], lw=0.4, color="#334155")
            ax.axvline(0, color="red", ls=":", lw=1)
            ax.axvspan(-ev["minutos_de_desgaste"] / MIN_POR_DIA, 0, color="orange", alpha=0.12)
            ax.grid(alpha=0.2)
            if j == 0:
                ax.set_ylabel(nombres[s])
        axes[0, j].set_title(f"{ev['causa']}\n{ev['id_evento']} · {t_f:%d/%m/%Y %H:%M}", fontsize=10)
        axes[-1, j].set_xlabel("Días de turno respecto a la falla (0 = falla)")
    fig.suptitle("Dataset v3 · un ejemplo de cada causa · franja naranja: desgaste previo · línea roja: falla", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "ejemplos_por_causa.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(16, 11), sharex=True)
    diario = df.set_index("timestamp")[SENSORES].resample("D").mean().dropna()
    colores = {"Desalineación del troquel": "#2563eb", "Desajuste de presión de corte": "#0d9488",
               "Sobrecarga del motor principal": "#dc2626"}
    for ax, s in zip(axes, SENSORES):
        ax.plot(diario.index, diario[s], lw=1)
        for _, f in bitacora.iterrows():
            ax.axvline(f["inicio"], color=colores[f["causa"]], alpha=0.5, lw=1)
        ax.set_ylabel(nombres[s])
        ax.grid(alpha=0.2)
    for c, col in colores.items():
        axes[0].plot([], [], color=col, label=c)
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_title("Dataset v3 · promedio diario de 2024 · líneas verticales: fallas reales por causa")
    fig.tight_layout()
    fig.savefig(OUT / "anio_2024.png", dpi=140)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEMILLA)
    bitacora = cargar_bitacora()
    t = calendario(bitacora)
    valores, carga = operacion_normal(t, rng)
    valores, registro = aplicar_fallas(t, valores, bitacora, rng)
    df = pd.DataFrame({"timestamp": t, **{s: np.round(valores[s], 2) for s in SENSORES}})
    df.to_csv(SALIDA, index=False, date_format="%Y-%m-%d %H:%M:%S")
    registro.to_csv(OUT / "parametros_por_falla.csv", index=False, encoding="utf-8-sig")
    graficar(df, registro, bitacora)

    resumen = {
        "archivo": SALIDA.name, "filas": len(df), "semilla": SEMILLA,
        "periodo": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "dias_de_turno": int(df["timestamp"].dt.normalize().nunique()),
        "fallas": int(len(bitacora)), "fallas_por_causa": bitacora["causa"].value_counts().to_dict(),
        "fallas_repentinas": int(registro["repentina"].sum()),
        "estadisticas": df[SENSORES].describe().round(2).to_dict(),
        "parametros": {"nominal": NOMINAL, "cargas": CARGAS, "causas": CAUSAS, "prob_repentina": PROB_REPENTINA},
    }
    (OUT / "resumen.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: resumen[k] for k in ["filas", "periodo", "dias_de_turno", "fallas", "fallas_por_causa",
                                              "fallas_repentinas"]}, ensure_ascii=False, indent=1))
    print(df[SENSORES].describe().round(2).to_string())
    print(registro.to_string(index=False))


if __name__ == "__main__":
    main()
