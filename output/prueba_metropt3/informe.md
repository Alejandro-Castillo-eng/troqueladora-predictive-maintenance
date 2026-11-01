# Prueba del modelo RUL con un dataset público: MetroPT-3

**Conclusión:** el modelo entrenado con la troqueladora **no funciona** con MetroPT-3. Su error es entre 2,6 y 4,3 veces mayor que en la validación de 2025. Además, apenas distingue los momentos previos a una falla de la operación normal (AUC de 0,51 a 0,66, donde 0,5 es azar) y **no emitió ninguna alerta crítica** antes de las 4 fallas. La causa principal es que las fallas de la troqueladora son desviaciones enormes y muy claras, mientras que las de una máquina real son sutiles.

## Qué se probó

| | Troqueladora (entrenamiento) | MetroPT-3 (prueba) |
|---|---|---|
| Máquina | Prensa troqueladora (datos simulados) | Compresor de aire de un tren del metro de Oporto (**datos reales**) |
| Periodo | 2025, de 08:00 a 17:59 | 1 de febrero a 1 de septiembre de 2020, las 24 h |
| Lecturas | 1 por minuto | 1 cada 10 s, promediadas por minuto (252 720 minutos con datos) |
| Fallas | 37 | 4 fugas de aire, según el documento oficial del dataset |

Se usó `modelo_rul.joblib` **sin reentrenar**.

### Equivalencia de sensores

| Variable del modelo | Sensor de MetroPT-3 |
|---|---|
| `temp_motor_c` | `Oil_temperature`: temperatura del aceite del compresor (°C) |
| `presion_hidraulica_bar` | `TP3`: presión neumática del panel (bar) |
| `corriente_motor_a` | `Motor_current`: corriente del motor (A) |
| `vibracion_rms_mms` | **No existe**: se fijó en su valor normal (1,31 mm/s) |

### Traslado a la escala de la troqueladora

Cada sensor se convirtió con la fórmula:

> valor trasladado = media normal de la troqueladora + (valor − media base de MetroPT-3) ÷ desviación base × desviación normal de la troqueladora

Se probaron dos bases:
- **Base fija:** febrero de 2020, el primer mes, sin fallas.
- **Base móvil:** los 7 días anteriores a cada toma, sin contar las ventanas de falla. Elimina el efecto del verano.

### RUL de referencia

Se aplicó la misma regla que `train_model.py`: filas hasta el inicio de la siguiente falla ÷ 600, con tope de 10. Una unidad equivale a 600 minutos de datos, es decir, 10 horas.

Se evaluaron 241 760 minutos. Se excluyeron:
- 4 960 minutos **dentro** de las ventanas de falla.
- 6 000 minutos finales cuyo RUL no se puede asegurar.

Después de la última falla, si quedaban al menos 6 000 minutos sin falla reportada, el RUL de referencia es 10.

## Resultados

| Métrica | Troqueladora · validación 2025 | MetroPT-3 · base fija | MetroPT-3 · base móvil |
|---|---:|---:|---:|
| MAE global | 11,3 h (1,13 días) | **49,0 h** (4,90 días) | **29,2 h** (2,92 días) |
| R² | 0,82 | −8,06 | −2,75 |
| MAE con RUL ≤ 3 días | 0,48 h (29 min) | 25,7 h | 54,8 h |
| Sesgo global | — | −47,0 h (predice menos vida) | −24,1 h |
| **AUC: distingue el último día antes de la falla** | **1,00** | 0,66 | 0,62 |
| **AUC: distingue los 3 últimos días** | **1,00** | 0,63 | 0,51 |
| RUL predicho medio en operación normal | — | 4,85 | 7,10 |
| RUL predicho medio en febrero (periodo base) | — | 6,92 | 6,83 |
| RUL predicho medio el último día antes de la falla | — | 3,79 | 6,20 |
| Tiempo normal en advertencia (RUL ≤ 4) | — | **48,4 %** | 6,0 % |
| Alertas críticas (RUL ≤ 1) antes de una falla | — | **0 de 4** | **0 de 4** |

El **AUC** es la probabilidad de que el modelo asigne menos RUL a una toma cercana a la falla que a una toma normal: 0,5 equivale al azar y 1,0 a distinguirlas siempre.

**Una referencia ingenua** que predice siempre 10 obtiene un MAE de 5,0 horas, mejor que el modelo. Esto ocurre porque el 90 % de los minutos evaluados son normales. Esa referencia nunca daría una alerta, así que el AUC es la métrica más útil aquí.

### Por falla (base fija)

| Falla | Tiempo en advertencia en las 100 h previas | Tiempo en crítico | RUL predicho medio en las 10 h previas |
|---|---:|---:|---:|
| 18 abr 2020 | 10,2 % | 0 % | 4,69 |
| 29 may 2020 | 54,4 % | 0 % | 3,63 |
| 5 jun 2020 | 67,6 % | 0 % | 3,84 |
| 15 jul 2020 | 93,4 % | 0 % | 3,00 |

Con la base fija, el tiempo en advertencia antes de las fallas parece alto. Pero el modelo también pasa el 48 % del tiempo **normal** en advertencia, así que esas alertas no indican la falla. Con la base móvil las falsas alarmas bajan al 6 %, pero las alertas previas a las fallas 2 y 3 casi desaparecen (0 % y 1,8 %).

Gráficos:
- `rul_real_vs_predicho.png`: periodo completo.
- `zoom_fallas.png`: los 6 días previos a cada falla.
- `error_por_intervalo_comparado.png`: error por distancia a la falla frente a 2025.
- `sensores_trasladados.png`: sensores en la escala de la troqueladora.

## Por qué no funciona

1. **Las fallas de la troqueladora son exageradas en comparación con una máquina real.** El último día antes de una falla, los sensores de la troqueladora se alejan **15 a 19 desviaciones** de lo normal: temperatura +18,6, presión −14,9, corriente +17,1. En MetroPT-3, el último día se alejan solo **0,2 a 1,8 desviaciones** (temperatura +1,8, presión −0,2, corriente +0,5), y durante la propia falla de 1,2 a 5,1. El modelo aprendió a reconocer cambios muy grandes y no ve los pequeños. Por eso, en 2025 el AUC es perfecto (1,00): los datos simulados hacen que distinguir sea demasiado fácil.

2. **El patrón de las lecturas es diferente.** El compresor se enciende y se apaga continuamente, así que la corriente y la presión oscilan aunque todo esté bien. Incluso en febrero, el mes usado como base, el modelo predice 6,9 en lugar de 10.

3. **Efecto de la estación.** La temperatura del aceite sube con el verano. Con la base fija, el modelo lo interpreta como desgaste y queda en advertencia casi todo el tiempo desde junio.

4. **Falta la vibración.** En los propios datos de 2025, poner la vibración constante hace que el RUL predicho el último día antes de la falla suba de 0,54 a 3,14. Sin vibración, el modelo no puede llegar al nivel crítico ni siquiera con la troqueladora. Esto explica en parte que no hubiera alertas críticas en MetroPT-3, pero no explica la poca capacidad de distinguir.

## Qué se puede afirmar

- Este modelo **no se puede usar en otra máquina** solo con trasladar la escala de los sensores.
- Las métricas de 2025 (MAE de 29 minutos cerca de la falla, AUC de 1,00) describen el comportamiento con **datos simulados de fallas muy marcadas**. No garantizan un desempeño parecido con datos reales.
- La prueba no demuestra que el **método** (ventanas móviles + XGBoost) sea malo. Para evaluarlo habría que reentrenarlo con datos reales que tengan fallas registradas.

## Límites de esta prueba

- Solo 4 fallas, todas del mismo tipo (fuga de aire). Los minutos consecutivos no son casos independientes.
- El momento de falla es el inicio de cada ventana del documento oficial. El documento numera dos veces la falla «#1» y menciona un mantenimiento el «30Apr» que parece corresponder a mayo. Se usaron las fechas de inicio y fin tal como aparecen.
- Una unidad de RUL equivale a 10 horas de datos del compresor, que trabaja casi las 24 h. En la troqueladora corresponde a un turno.
- La equivalencia de sensores es aproximada: aceite del compresor frente a motor, y presión neumática frente a hidráulica.

## Reproducción

```bash
python probar_modelo_metropt3.py
```

El script descarga MetroPT-3 desde UCI en `datos_externos/` si no existe. Esa carpeta no se sube a git. El script no modifica el modelo ni los datos del proyecto.

Dataset: Davari, N., Veloso, B., Ribeiro, R.P., Gama, J. *MetroPT-3 Dataset*. UCI Machine Learning Repository (2023), DOI [10.24432/C5VW3R](https://doi.org/10.24432/C5VW3R), licencia CC BY 4.0.
