# Dataset v2 calibrado con Industrial syn: ¿mejora la generalización?

**Conclusión:** se construyó una versión de la telemetría de 2025 con las mismas columnas, fechas, rangos normales y fallas, pero con el comportamiento de los sensores tomado de Industrial Machine PdM (syn). Con esos datos, el modelo v2 aprende a detectar fallas **sutiles**: en su propia validación distingue el último día antes de la falla con un AUC de 1,00, cuando las fallas solo cambian los sensores entre 0 y 2,4 desviaciones. Pero **no mejora con datos reales**: en MetroPT-3 obtiene el mismo AUC que el modelo original (0,61–0,65, cerca del azar) y más error. Calibrar con otro dataset **simulado** hizo el modelo más robusto entre simulaciones, pero no lo acercó a una máquina real.

## 1. Cómo se construyó el dataset v2

Script: `generar_dataset_v2.py`. Salida: `telemetria_troqueladora_v2.csv`.

| Se conserva del original | Se toma de Industrial syn |
|---|---|
| Columnas, 216 000 marcas de tiempo (08:00–17:59) | Fluctuaciones de operación normal (incluidos los modos reposo, normal y pico) |
| Las 37 fallas de `log_fallas_mantenimiento.csv`, sin cambios | Forma y magnitud del desgaste antes de cada falla |
| Media y desviación normales de cada sensor | Relación entre los cuatro sensores (salen de la misma máquina y el mismo minuto) |

Pasos:
1. Industrial syn se interpola a una toma por minuto. Cada sensor se traslada a la escala normal de la troqueladora según el estado normal (RUL > 48 h) de cada tipo de máquina.
2. A cada falla de la bitácora se le asigna un **segmento de desgaste real del dataset fuente**, de al menos 24 h, que termina en una falla de tipo equivalente:
   - Sobrecarga eléctrica y fricción → `electrical` o `motor_overheat`.
   - Desbalance en rodamiento y fatiga en bancada → `bearing`.
3. El segmento se coloca de modo que **termine exactamente en la fecha de la falla**. El resto se completa con fragmentos de operación normal del mismo tipo de máquina.
4. **Validación independiente:** 4 máquinas fuente (2, 8, 11 y 20, una por tipo) se reservaron solo para el último 20 % de 2025. Ningún dato de validación procede de máquinas usadas en el entrenamiento.

El detalle de qué fuente se usó en cada falla está en `output/dataset_v2/fuentes_por_falla.csv`.

### Resultado

| Sensor | Normal: original → v2 | Último día antes de la falla: original → v2 | Rango completo: original → v2 |
|---|---|---|---|
| Temperatura | 52,0 ± 1,2 → 52,0 ± 1,0 °C | **+18,6 → +2,4** desviaciones | 46,8–98,5 → 50,2–61,2 °C |
| Presión | 215,0 ± 2,0 → 215,1 ± 1,7 bar | **−14,9 → 0,0** | 154,6–240,0 → 212,6–219,7 bar |
| Vibración | 1,31 ± 0,16 → 1,30 ± 0,14 mm/s | **+20,3 → +1,1** | 0,7–8,1 → 1,1–2,8 mm/s |
| Corriente | 21,0 ± 0,6 → 21,0 ± 0,5 A | **+17,1 → +0,8** | 18,2–42,1 → 20,3–24,1 A |

Los valores normales se mantienen. Las fallas pasan de ser cambios enormes a cambios de 0 a 2,4 desviaciones, una magnitud parecida a la observada en el dataset real MetroPT-3 (0,2–1,8). Gráfico: `output/dataset_v2/original_vs_v2_abril.png`.

## 2. Modelo v2

Script: `entrenar_modelo_v2.py`. Salida: `modelo_rul_v2.joblib`. Usa el **mismo método** de `train_model.py`: mismo RUL, 29 variables, partición 80/20 e hiperparámetros. El modelo original no se modificó.

| Variables más importantes | Modelo original | Modelo v2 |
|---|---|---|
| 1 | `temp_motor_c_mean_15`: **73,7 %** | `temp_motor_c_zscore_nom`: 28,8 % |
| 2 | `vibracion_rms_mms`: 11,4 % | `temp_motor_c_mean_60`: 13,5 % |
| 3 | `temp_motor_c_mean_60`: 1,9 % | `indice_estres_termomecanico`: 11,6 % |

El modelo v2 reparte la importancia entre más variables y usa más las medias de 60 minutos y los z-scores. Es lo esperable cuando la señal es sutil y hay que filtrar las fluctuaciones.

## 3. Comparación

Script: `comparar_modelos_v1_v2.py`. Todas las métricas se calculan igual para ambos modelos:
- Se excluyen las filas con RUL desconocido.
- "Normal" significa RUL real ≥ 10.
- AUC: 0,5 = azar, 1,0 = distingue siempre.
- Alerta crítica: RUL ≤ 1 durante 30 tomas seguidas en los 10 días previos.

| Evaluación | Modelo | MAE | MAE con RUL ≤ 3 días | AUC último día | Normal en advertencia | Alertas críticas antes de la falla |
|---|---|---:|---:|---:|---:|---:|
| Validación, datos **originales** | original | 10,9 h | 0,5 h | 1,00 | 0,0 % | 4 de 4 |
| | v2 | 15,5 h | 11,0 h | 1,00 | 0,1 % | 0 de 4 |
| Validación, datos **v2** | original | 35,5 h | 27,6 h | 0,84 | 28,0 % | 0 de 4 |
| | v2 | 15,5 h | 5,4 h | **1,00** | 0,4 % | **4 de 4** |
| **MetroPT-3 (real)**, base fija | original | 49,0 h | 25,7 h | 0,66 | 48,4 % | 0 de 4 |
| | v2 | 60,8 h | 18,2 h | 0,65 | 60,4 % | 2 de 4 (con 6,8 % del tiempo normal en crítico) |
| **MetroPT-3 (real)**, base móvil | original | 29,2 h | 54,8 h | 0,62 | 6,0 % | 0 de 4 |
| | v2 | 38,5 h | 46,7 h | 0,61 | 11,7 % | 0 de 4 |

Gráficos: `comparacion_metricas.png` y `metropt3_zoom_fallas.png`.

### Qué muestran los resultados

1. **El modelo v2 aprendió a detectar fallas sutiles.** En los datos v2, con máquinas fuente que no vio al entrenar, distingue el último día con AUC 1,00 y avisa antes de las 4 fallas. El modelo original, con los mismos datos, baja a AUC 0,84 y no da ninguna alerta crítica.
2. **El modelo v2 también reconoce las fallas exageradas de los datos originales** (AUC 1,00), aunque con valores de RUL menos precisos y sin llegar al umbral crítico. Es más robusto entre simulaciones.
3. **En datos reales no hay mejora.** En MetroPT-3, ambos modelos tienen un AUC de 0,61–0,66, cerca del azar. El v2 incluso tiene más error y más falsas alarmas. Sus 2 alertas críticas con base fija no tienen valor, porque también pasa el 6,8 % del tiempo normal en crítico.

### Por qué no mejora con datos reales

- **Industrial syn también es simulado.** Sus fallas son sutiles, pero su forma (una subida gradual de temperatura durante 1–4 días) la decidió su autor. Las fallas reales de MetroPT-3, fugas de aire, se manifiestan de otra manera: casi no hay cambios antes y, durante la falla, el compresor trabaja sin parar.
- **Calibrar la magnitud no basta: importa la forma del desgaste.** El v2 aprendió la forma de la simulación de Industrial syn, no la de una máquina real.
- MetroPT-3 sigue sin vibración y es otro tipo de máquina con otro tipo de falla.

## Qué se puede afirmar

- Con un dataset de fallas sutiles, el **método** de este proyecto (ventanas móviles + XGBoost) es capaz de detectarlas cuando siguen un patrón consistente. Eso sugiere que el método no es el problema.
- Ningún dato simulado, ni el original ni el calibrado, permitió generalizar a una máquina real. Para eso hace falta **entrenar con datos reales** del tipo de máquina y de falla que se quiere vigilar.

## Límites

- Industrial syn tiene lecturas cada ~16 min interpoladas a 1 min. Por eso el dataset v2 es más suave toma a toma que el original.
- Solo hay 30 segmentos de desgaste de al menos 24 h con tipo compatible. Algunos se reutilizaron dentro del entrenamiento, nunca entre entrenamiento y validación.
- La equivalencia entre tipos de falla de la bitácora y del dataset fuente es aproximada.
- La validación de cada modelo contiene solo 4 fallas, y MetroPT-3 otras 4.

## Reproducción

```bash
python generar_dataset_v2.py
```

```bash
python entrenar_modelo_v2.py
```

```bash
python comparar_modelos_v1_v2.py
```

Los datasets externos se descargan en `datos_externos/` (no se suben a git). Ningún script modifica `telemetria_troqueladora.csv`, `log_fallas_mantenimiento.csv` ni `modelo_rul.joblib`.
