# Prueba del modelo RUL con un dataset con las mismas cuatro variables: Industrial Machine PdM (syn)

**Conclusión:** este dataset tiene las cuatro variables del modelo, y la temperatura y la vibración ya están en rangos parecidos a los de la troqueladora. Aun así, el modelo guardado **no estima bien el RUL**:
- Su error medio (22 a 25 h) es **mayor que el de predecir siempre el promedio** (19,8 h).
- El RUL predicho apenas cambia al acercarse la falla (correlación de rangos de 0,17).
- Al distinguir las 10 horas previas a una falla de la operación lejana, acierta poco más que el azar (AUC de 0,63 a 0,64).

Hay un matiz. Si la temperatura y la vibración se usan con sus valores originales, el modelo emite alertas críticas **64 veces más a menudo cerca de la falla que lejos de ella**, y avisa antes de 20 de las 63 fallas. Hay algo de señal, pero no la suficiente para ser útil.

## Qué se probó

| | Troqueladora (entrenamiento) | Industrial Machine PdM (prueba) |
|---|---|---|
| Máquinas | 1 prensa troqueladora | 20: 5 CNC, 5 compresores, 5 bombas, 5 brazos robóticos |
| Origen | Simulado | Simulado (lo declara el autor; licencia CC0) |
| Periodo | 2025, de 08:00 a 17:59 | 1 al 14 de enero de 2024, las 24 h |
| Lecturas | 1 por minuto | Irregulares, cada ~16 min (24 042 en total) |
| RUL | Calculado desde la bitácora (en días de 10 h) | Columna `rul_hours`. Baja 1 h por hora y llega a 0,5 h en cada falla |
| Fallas | 37 | 63 |

Se usó `modelo_rul.joblib` **sin reentrenar**.

### Preparación de los datos

- Cada máquina se interpoló a **una toma por minuto**, sin cruzar huecos de más de 60 min. Resultaron 381 670 minutos.
- RUL real = `rul_hours` ÷ 10, porque en el modelo 1 unidad = 600 tomas de 1 min = 10 h.
- **Se excluyeron de la evaluación** dos tipos de minutos, y quedaron 275 945 evaluables:
  - Los que tienen el RUL fijo en 0,5 h. Es un valor de relleno tras la falla: el 28 % de las lecturas, con periodos de hasta 180 h.
  - Los tramos donde el RUL salta hacia arriba sin mantenimiento (19 casos).

### Equivalencia de sensores y rangos originales

| Variable del modelo | Columna del dataset | Rango del dataset (mín · mediana · máx) | Mediana de la troqueladora |
|---|---|---|---|
| `temp_motor_c` | `temperature_motor` | 28,0 · **50,1** · 95,0 °C | 52,9 °C |
| `vibracion_rms_mms` | `vibration_rms` | 0,35 · **1,27** · 10,0 mm/s | 1,39 mm/s |
| `corriente_motor_a` | `current_phase_avg` | 2,2 · **6,4** · 35,0 A | 21,4 A |
| `presion_hidraulica_bar` | `pressure_level` | 10,1 · **46,3** · 206,5 | 213,8 bar |

### Dos variantes

- **Directa:** temperatura y vibración con sus **valores originales**. Corriente y presión trasladadas a la escala de la troqueladora.
- **Trasladada:** los cuatro sensores trasladados.

El traslado usa la fórmula: valor trasladado = media normal de la troqueladora + (valor − media normal del tipo de máquina) ÷ su desviación × desviación normal de la troqueladora. Se considera "normal" cada lectura con RUL > 48 h.

## Resultados

Para comparar en igualdad de condiciones, la columna de 2025 usa solo las tomas con RUL conocido menor de 10 días.

| Métrica | Troqueladora · validación 2025 | Directa | Trasladada |
|---|---:|---:|---:|
| Error medio absoluto (MAE) | 8,2 h | **22,2 h** | **24,7 h** |
| MAE de predecir siempre el promedio | — | 19,8 h | 19,8 h |
| R² | 0,73 | −0,38 | −0,64 |
| Correlación de rangos (Spearman) | 0,93 | 0,17 | 0,17 |
| AUC: últimas 10 h frente a ≥ 48 h de la falla | **1,00** | 0,64 | 0,63 |
| AUC: últimas 24 h frente a ≥ 48 h | **1,00** | 0,62 | 0,61 |
| Tiempo lejos de la falla (≥ 48 h) en advertencia (RUL ≤ 4) | 0 % | 55,2 % | 29,6 % |
| Tiempo lejos de la falla en crítico (RUL ≤ 1) | 0 % | 0,32 % | 0,00 % |
| Fallas con alerta crítica sostenida (≥ 30 min) en las 100 h previas | — | **20 de 63** | 0 de 63 |

El AUC es la probabilidad de que una toma cercana a la falla reciba menos RUL que una toma lejana: 0,5 equivale al azar y 1,0 a distinguirlas siempre.

### Alertas críticas en la variante directa

| Zona según el RUL real | Alertas críticas sostenidas | Horas evaluadas | Alertas por cada 100 h |
|---|---:|---:|---:|
| Últimas 10 h antes de la falla | 25 | 585 | **4,27** |
| Entre 10 y 48 h antes | 43 | 2 530 | 1,70 |
| A 48 h o más | 1 | 1 484 | **0,07** |

Las alertas críticas se concentran cerca de la falla, así que hay una señal real. Pero son pocas: 43 de las 63 fallas no tuvieron ninguna. Además, las que hubo llegaron con antelaciones muy dispersas, de 3 a 100 h (mediana 38 h), lo que no coincide con lo que significa "RUL ≤ 1" (10 h). Por tipo de máquina: CNC 10, compresor 7, bomba 3 y brazo robótico 0.

Gráficos:
- `rul_predicho_vs_real.png`: RUL predicho frente a real, incluida la troqueladora 2025.
- `ejemplos_por_maquina.png`: una máquina de cada tipo.
- `error_por_intervalo_comparado.png`: error por distancia a la falla frente a 2025.

## Por qué no funciona

1. **Las fallas cambian poco los sensores en comparación con la troqueladora.** En las últimas 10 h antes de la falla, los sensores se alejan de lo normal:

   | Sensor | Este dataset (últimas 10 h) | Troqueladora 2025 (último día) |
   |---|---:|---:|
   | Temperatura | **+1,4** desviaciones | **+18,6** |
   | Vibración | +0,8 | — |
   | Corriente | +0,5 | **+17,1** |
   | Presión | +0,1 | **−14,9** |

   Es el mismo problema que con MetroPT-3. El modelo aprendió fallas que se alejan 15–19 desviaciones de lo normal y no reconoce cambios de 1–2.

2. **El modo de operación pesa más que el desgaste.** En modo "pico", una bomba sana llega a 80 °C y una CNC a 4,1 mm/s, más que antes de una falla. El RUL predicho oscila con cada cambio de modo. Lejos de la falla, en la variante directa, el RUL medio fue 4,45 en reposo, 4,00 en normal y 3,41 en pico. La troqueladora no tiene modos.

3. **El modelo solo reconoce la fase final de desgaste de sus propios datos.** En la curva de 2025, el modelo predice ~8,7 siempre que faltan más de ~4,5 días y sigue la diagonal casi perfectamente por debajo. Es decir, no estima el RUL a partir de un desgaste gradual: detecta el patrón concreto con que los datos simulados empiezan a degradarse unos 4,5 días antes de cada falla.

4. **Las lecturas originales cada ~16 min** se interpolaron a 1 min. Las ventanas de 15 y 60 minutos contienen solo de 1 a 4 lecturas reales.

## Qué se puede afirmar

- Que el dataset tenga las **mismas cuatro variables y rangos parecidos no basta** para que el modelo generalice. Importa cómo se manifiesta la falla en los sensores.
- Con la temperatura y la vibración en su escala original hay una señal débil pero real: alertas críticas mucho más frecuentes cerca de la falla. No basta para estimar el RUL: el error supera al de predecir siempre el promedio.
- Junto con la prueba de MetroPT-3, el resultado indica que las métricas de 2025 dependen de lo marcadas que son las fallas en los datos simulados de entrenamiento.

## Límites de esta prueba

- El dataset es **simulado** y su generación no está documentada en detalle. El RUL de relleno (0,5 h) y los saltos sin mantenimiento muestran inconsistencias.
- Solo hay 14 días por máquina. Los minutos consecutivos y las máquinas del mismo tipo no son casos independientes.
- Se interpolaron lecturas irregulares a 1 min.
- La base "normal" (RUL > 48 h) y el horizonte "lejos" (≥ 48 h) son criterios de esta prueba.

## Reproducción

```bash
python probar_modelo_industrial_syn.py
```

El script descarga el dataset desde Kaggle en `datos_externos/` si no existe. Esa carpeta no se sube a git. El script no modifica el modelo ni los datos del proyecto.

Dataset: Tatheer Abbas, *Industrial Machine Predictive Maintenance (syn)*, Kaggle, licencia CC0: <https://www.kaggle.com/datasets/tatheerabbas/industrial-machine-predictive-maintenance>.
