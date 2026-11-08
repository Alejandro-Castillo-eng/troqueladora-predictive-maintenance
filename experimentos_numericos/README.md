# Experimentos numéricos (validación, etapa 1)

Validación del sistema de mantenimiento predictivo con **datos sintéticos controlados**. Se varían de forma sistemática las propiedades de los datos para ver cuándo los modelos funcionan y cuándo dejan de hacerlo, comparándolos con métodos simples (líneas base) sobre los mismos datos.

## Archivos

| Archivo | Qué hace |
|---|---|
| `synthetic_generator.py` | Generador configurable: calendario real de 2024, **27 fallas reales** de `log_fallas_2024.csv` y la firma de cada causa. Varía magnitud de la falla, ruido, duración del desgaste y datos faltantes |
| `run_experiments.py` | Ejecuta todos los escenarios y réplicas; guarda `results/resultados_replicas.csv` (una fila por escenario, réplica y método) |
| `analisis.py` | Estadística y gráficos: tablas con media ± desviación e IC 95 %, pruebas pareadas, Holm, d de Cohen, Kruskal-Wallis, boxplots y curva |
| `analisis.ipynb` | Reproduce las tablas y figuras |
| `results/` | Resultados brutos, tablas, figuras y `entorno.json` (versiones y semillas) |

## Diseño

**Factores del generador.**

| Factor | Qué controla | Niveles |
|---|---|---|
| Magnitud de la falla | Cuánto se aleja la variable de su causa (presión −8 a −18 bar, vibración hasta 2,8–4,5 mm/s, corriente +10 a +25 %, temperatura +5 a +15 °C) | 100 % · 50 % · 25 % (barrido: 100, 75, 50, 35, 25 y 15 %) |
| Ruido | Desviación del ruido de los sensores y frecuencia de picos | ×1 · ×2 · ×3 |
| Duración del desgaste | Días de turno con señal antes de la falla | v3 por causa (0,5–6) · 0,5–2 · 0,2–0,6 (2–6 h) |
| Datos faltantes | Lecturas perdidas en rachas de 1–30 min, imputadas con el último valor | 0 % · 30 % |

**Escenarios.**

| Escenario | Magnitud | Ruido | Desgaste | Faltantes | Réplicas |
|---|---|---|---|---|---|
| Típico | 100 % | ×1 | v3 | 0 % | 30 |
| Borde | 50 % | ×2 | 0,5–2 días | 0 % | 30 |
| Extremo | 25 % | ×3 | 0,5–2 días | 0 % | 30 |
| Patológico | 25 % | ×3 | 2–6 h | 30 % | 30 |
| Barrido de magnitud | 100 → 15 % | ×1 | v3 | 0 % | 10 por nivel |

**Métodos comparados (mismos datos por réplica, CRN).**

| Tarea | Propuesto | Línea base | Referencia |
|---|---|---|---|
| ¿Cuándo? (RUL) | XGBoost (`train_model.py`) | Regresión lineal Ridge, mismas 29 variables | Predecir siempre el RUL medio |
| ¿Qué tipo? | XGBoost multiclase (`clasificador_tipo_falla_v3.py`) | Reglas de umbral del tablero (`app.py`) | — |

**Protocolo por réplica.** Semilla `s` (1–30) → dataset → partición cronológica 80/20 → entrenamiento de los métodos con el 80 % → métricas en el 20 % (6 fallas reales: EV-22 a EV-27).

**Métricas.** RUL: AUC (último día y 3 días frente a operación normal), MAE, RMSE, R², precision/recall/F1 de la alerta crítica (RUL ≤ 1), fallas con alerta previa y falsas alarmas. Tipo: accuracy, accuracy balanceada, F1 macro y por clase, y fallas con la causa correcta.

**Estadística.** Diferencias pareadas por semilla. Shapiro-Wilk → t pareada (normal) o Wilcoxon (no normal). Corrección de Holm sobre todas las comparaciones. d de Cohen pareado. IC 95 % (t de Student). Kruskal-Wallis para el efecto del escenario.

## Reproducción

```bash
python run_experiments.py
```

```bash
python analisis.py
```

Semillas 1–30 (barrido: 1–10). `run_experiments.py --rapido` ejecuta 2 réplicas por escenario para probar.
