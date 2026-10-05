"""Evalua el modelo guardado por tiempo restante; no entrena ni modifica datos."""

from pathlib import Path
import contextlib
import hashlib
import io
import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import train_model as tm


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "evaluacion_intervalos_rul"


def metricas(frame):
    error = frame["error_dias"]
    return {
        "filas": int(len(frame)),
        "fallas_referencia": int(frame["siguiente_falla"].nunique()),
        "mae_dias": float(error.abs().mean()),
        "mae_horas_operativas": float(error.abs().mean() * 10),
        "mae_minutos_operativos": float(error.abs().mean() * 600),
        "rmse_horas_operativas": float(np.sqrt((error ** 2).mean()) * 10),
        "sesgo_horas_operativas": float(error.mean() * 10),
        "p90_error_horas_operativas": float(error.abs().quantile(.90) * 10),
        "dentro_1_hora_pct": float((error.abs() <= .1).mean() * 100),
        "dentro_1_dia_pct": float((error.abs() <= 1).mean() * 100),
    }


def main():
    model_path = ROOT / "modelo_rul.joblib"
    bundle = joblib.load(model_path)
    with contextlib.redirect_stdout(io.StringIO()):
        df, failures = tm.cargar_datos(
            str(ROOT / tm.FILE_TELEMETRIA), str(ROOT / tm.FILE_FALLAS)
        )
        df, failure_indices = tm.cruzar_marcas_y_calcular_rul(df, failures)
        df, features = tm.generar_variables_ventana_movil(df)

    split = int(len(df) * .8)
    holdout = df.iloc[split:].copy()
    if features != bundle["feature_cols"]:
        raise ValueError("Las caracteristicas actuales difieren del modelo guardado.")
    raw_prediction = bundle["modelo"].predict(holdout[features])
    prediction = np.clip(raw_prediction, 0, bundle["max_rul_dias"])
    positions = np.searchsorted(failure_indices, holdout.index.to_numpy(), side="left")
    known = positions < len(failure_indices)
    actual_days = np.full(len(holdout), np.nan)
    next_failure = np.full(len(holdout), np.datetime64("NaT"), dtype="datetime64[ns]")
    actual_days[known] = (
        failure_indices[positions[known]] - holdout.index.to_numpy()[known]
    ) / bundle["muestras_por_turno"]
    next_failure[known] = df.iloc[failure_indices[positions[known]]]["dt"].to_numpy()

    rows = pd.DataFrame({
        "timestamp": holdout["dt"].to_numpy(),
        "siguiente_falla": next_failure,
        "rul_sin_tope_dias": actual_days,
        "rul_objetivo_codigo_dias": holdout["rul_dias"].to_numpy(),
        "rul_predicho_sin_recorte_dias": raw_prediction,
        "rul_predicho_evaluacion_dias": prediction,
    })
    rows["error_dias"] = prediction - rows["rul_objetivo_codigo_dias"]
    rows["grupo"] = "sin_siguiente_falla_registrada"
    rows.loc[known & (actual_days >= 10), "grupo"] = "objetivo_limitado_a_10_dias"
    exact = known & (actual_days < 10)
    rows.loc[exact, "grupo"] = "rul_conocido_menor_a_10_dias"
    rows["intervalo"] = ""

    summaries = []
    per_event = []
    for day in range(10):
        mask = exact & (actual_days <= day + 1)
        mask &= (actual_days >= 0) if day == 0 else (actual_days > day)
        label = "0 a 1" if day == 0 else f">{day} a {day + 1}"
        if day == 9:
            label = ">9 a <10"
        rows.loc[mask, "intervalo"] = label
        group = rows.loc[mask]
        summaries.append({"intervalo_dias_operativos": label, **metricas(group)})
        for event, event_rows in group.groupby("siguiente_falla"):
            per_event.append({
                "intervalo_dias_operativos": label,
                "siguiente_falla": str(event),
                **metricas(event_rows),
            })

    interval_table = pd.DataFrame(summaries)
    event_table = pd.DataFrame(per_event)
    if int(interval_table["filas"].sum()) != int(exact.sum()):
        raise ValueError("Los intervalos no cubren las filas con RUL conocido.")
    global_mae = float(rows["error_dias"].abs().mean())
    if not np.isclose(global_mae, bundle["metricas"]["mae_global"], atol=1e-6):
        raise ValueError("No se reproduce el MAE de la validacion del modelo guardado.")
    near = rows.loc[known & (actual_days <= 3)]
    far = rows.loc[exact & (actual_days > 3)]
    sequence = interval_table.loc[interval_table["filas"] > 0, "mae_horas_operativas"].to_numpy()
    reversal_indices = np.flatnonzero(np.diff(sequence) < 0)
    metadata = {
        "modelo": bundle["nombre_modelo"],
        "sha256_modelo": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "entrenamiento_guardado": bundle.get("fecha_entrenamiento"),
        "validacion_inicio": str(holdout["dt"].min()),
        "validacion_fin": str(holdout["dt"].max()),
        "filas_validacion": len(rows),
        "filas_rul_conocido_menor_10": int(exact.sum()),
        "filas_objetivo_limitado_10": int((known & (actual_days >= 10)).sum()),
        "filas_sin_siguiente_falla": int((~known).sum()),
        "fallas_con_referencia_exacta": int(rows.loc[exact, "siguiente_falla"].nunique()),
        "predicciones_recortadas": int(((raw_prediction < 0) | (raw_prediction > 10)).sum()),
        "mae_global_reproducido_dias": global_mae,
        "cercanas_hasta_3_dias": metricas(near),
        "alejadas_mas_3_menos_10_dias": metricas(far),
        "mae_monotono_creciente_al_alejarse": bool(len(reversal_indices) == 0),
        "descensos_al_alejarse": [
            {"desde": summaries[int(i)]["intervalo_dias_operativos"],
             "hasta": summaries[int(i + 1)]["intervalo_dias_operativos"]}
            for i in reversal_indices
        ],
        "metodo": (
            "Mismo modelo guardado, primer 80% cronologico para entrenamiento y ultimo "
            "20% de 2025 para validacion. Predicciones recortadas 0-10 como train_model.py. "
            "Intervalos basados en RUL de referencia, no predicho. Solo RUL conocido <10 "
            "en la comparacion principal. RUL=distancia en filas/600; dia operativo=10h. "
            "Las filas en el instante de falla se incluyen con RUL=0. No se reentrena."
        ),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT / "predicciones_validacion_2025.csv", index=False, encoding="utf-8-sig")
    interval_table.to_csv(OUT / "errores_por_intervalo.csv", index=False, encoding="utf-8-sig")
    event_table.to_csv(OUT / "errores_por_falla_e_intervalo.csv", index=False, encoding="utf-8-sig")
    (OUT / "resumen.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(13, 7))
    values = interval_table["mae_horas_operativas"].to_numpy()
    colors = ["#0d9488"] * 3 + ["#2563eb"] * 7
    ax.bar(range(10), values, color=colors, width=.72)
    for i, value in enumerate(values):
        if np.isfinite(value):
            ax.text(i, value + .18, f"{value:.2f} h", ha="center", fontsize=12, fontweight="bold")
    ax.set_xticks(range(10), interval_table["intervalo_dias_operativos"], fontsize=11)
    ax.set_xlabel("Tiempo restante de referencia: días operativos (1 día = 10 horas)", fontsize=13, labelpad=12)
    ax.set_ylabel("Error promedio absoluto: horas de operación", fontsize=13)
    ax.set_title("¿El modelo estima mejor cuando la falla está cerca?", fontsize=19, pad=18)
    ax.text(.5, 1.01, "Validación reservada de 2025 · Menor altura = menor error", transform=ax.transAxes,
            ha="center", fontsize=12)
    ax.set_ylim(0, np.nanmax(values) * 1.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=.18)
    ax.set_axisbelow(True)
    fig.text(.5, .025, "Solo tomas con siguiente falla registrada y RUL menor a 10 días. "
             "4 eventos de falla; las filas consecutivas no son casos independientes.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(OUT / "error_por_intervalo.png", dpi=180)
    plt.close(fig)
    print(interval_table.to_string(index=False))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
