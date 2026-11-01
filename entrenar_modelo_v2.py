"""
entrenar_modelo_v2.py
=====================
Entrena un modelo NUEVO con el mismo metodo de train_model.py (mismo calculo de RUL, mismas
29 variables, misma particion 80/20 y mismos hiperparametros de XGBoost), pero con
'telemetria_troqueladora_v2.csv' (comportamiento calibrado con Industrial syn).

Guarda 'modelo_rul_v2.joblib'. NO modifica modelo_rul.joblib ni train_model.py.

Uso:  python generar_dataset_v2.py   (si aun no existe el CSV)
      python entrenar_modelo_v2.py
"""

from pathlib import Path

import train_model as tm


ROOT = Path(__file__).resolve().parent
TELEMETRIA_V2 = ROOT / "telemetria_troqueladora_v2.csv"
MODELO_V2 = ROOT / "modelo_rul_v2.joblib"


def main():
    df, fallas = tm.cargar_datos(str(TELEMETRIA_V2), str(ROOT / tm.FILE_FALLAS))
    df, _ = tm.cruzar_marcas_y_calcular_rul(df, fallas, max_rul=tm.MAX_RUL_DIAS)
    df, features = tm.generar_variables_ventana_movil(df)
    modelo, nombre, metricas = tm.entrenar_modelo(df, features)
    tm.guardar_paquete_modelo(modelo, nombre, features, metricas, str(MODELO_V2))


if __name__ == "__main__":
    main()
