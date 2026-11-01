"""
entrenar_modelo_v3.py
=====================
Entrena 'modelo_rul_v3.joblib' con el MISMO metodo de train_model.py (RUL hacia la siguiente
falla con tope de 10 dias de turno, 29 variables, particion cronologica 80/20, XGBoost con los
mismos hiperparametros) usando:

- telemetria_troqueladora_2024_v3.csv  (generada con generar_dataset_v3.py)
- log_fallas_2024.csv                   (27 fallas reales; momento de falla = hora de inicio)

No modifica modelo_rul.joblib, modelo_rul_v2.joblib ni train_model.py.
Uso:  python generar_dataset_v3.py && python entrenar_modelo_v3.py
"""

from pathlib import Path

import pandas as pd

import train_model as tm


ROOT = Path(__file__).resolve().parent
TELEMETRIA_V3 = ROOT / "telemetria_troqueladora_2024_v3.csv"
BITACORA_2024 = ROOT / "log_fallas_2024.csv"
MODELO_V3 = ROOT / "modelo_rul_v3.joblib"


def cargar_v3():
    df = pd.read_csv(TELEMETRIA_V3)
    df["dt"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("dt").reset_index(drop=True)
    fallas = pd.read_csv(BITACORA_2024)
    fallas["dt_falla"] = pd.to_datetime(fallas["fecha"] + " " + fallas["hora_inicio"])
    return df, fallas.sort_values("dt_falla").reset_index(drop=True)


def main():
    df, fallas = cargar_v3()
    print(f"Telemetria v3: {len(df):,} filas · Fallas reales 2024: {len(fallas)}")
    df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas, max_rul=tm.MAX_RUL_DIAS)
    df, features = tm.generar_variables_ventana_movil(df)
    modelo, nombre, metricas = tm.entrenar_modelo(df, features)
    tm.guardar_paquete_modelo(modelo, nombre, features, metricas, str(MODELO_V3))


if __name__ == "__main__":
    main()
