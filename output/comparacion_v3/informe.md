# Dataset v3 (fallas reales de 2024) y comparación de los tres modelos

**Conclusión:** se construyó un dataset de 2024 sobre las **27 fallas reales** de la prensa (`log_fallas_2024.csv`). Cada causa tiene su propia firma en los sensores, según el cuadro causa → variables, y **todas las fallas se desarrollan con tendencia durante días**, como indicó el usuario para esta prensa.

Con esos datos, el modelo v3:
- Distingue el último día antes de la falla con un **AUC de 0,985** en su propia validación.
- Avisa antes de **5 de 6 fallas**, sin falsas alarmas críticas.
- Es el primero que **usa las cuatro variables**.

En las pruebas externas **mejora en dos** (SKAB, AUC 0,67; MetroPT-3 con base móvil, 0,66) y empeora en las demás. Ninguno de los tres modelos alcanza una detección útil y fiable en datos reales de otras máquinas.

## 1. Cómo se construyó el dataset v3

Script: `generar_dataset_v3.py`. Salida: `telemetria_troqueladora_2024_v3.csv` (184 896 filas).

| Elemento | Origen |
|---|---|
| **Fallas: fecha, causa, hora de inicio y de fin** | **Reales**: `log_fallas_2024.csv`, 27 eventos de 2024 |
| Calendario | Real: lunes a sábado, 08:00–17:59. Sin lecturas durante cada parada (inicio → fin) |
| Comportamiento de los sensores | **Simulado**, con parámetros de referencias técnicas generales |

### Firma de cada causa (cuadro del usuario)

| Causa | Fallas | Qué cambia antes de la falla | Magnitud | Duración del desgaste |
|---|---:|---|---|---|
| Desajuste de presión de corte | 13 | **Presión ↓** y fluctúa más | −8 a −18 bar | 0,5–4 días de turno |
| Desalineación del troquel | 8 | **Vibración ↑** y fluctúa más | hasta 2,8–4,5 mm/s (zonas C–D de la ISO 10816-3) | 1–6 días |
| Sobrecarga del motor principal | 6 | **Corriente ↑** y **temperatura ↑** con retraso térmico | 110–125 % de la corriente nominal; +5 a +15 °C | 1–5 días |

### Otros elementos

- **Variabilidad entre fallas.** La forma del desgaste puede ser lineal, exponencial o escalonada, y la intensidad varía.
- **Sin fallas repentinas.** Según el usuario, en esta prensa las tres causas se desarrollan con tendencia durante días. Una versión anterior incluía un 20 % de fallas casi repentinas. Era un supuesto del generador, no un dato, y se eliminó. Su resultado se resume al final.
- **La reparación reinicia el desgaste de esa causa.** Por eso dos fallas tienen muy poco desgaste propio, según las fechas reales:
  - EV-14 ocurrió durante la reparación de EV-13.
  - EV-27 ocurrió una hora después de la reparación de EV-26.
- **Operación normal:**
  - Cambios de carga por trabajo o troquel, cada 1–4 h, del 92 % al 108 %.
  - Calentamiento del motor al inicio de cada turno.
  - Variación diaria.
  - Picos transitorios aislados de vibración y presión.
- **Valores normales** iguales a los del proyecto: 52 °C, 214 bar, 1,3 mm/s y 21 A.
- **No se usó ningún dataset externo para calibrar.** Así siguen siendo pruebas independientes.

Parámetros de cada falla en `output/dataset_v3/parametros_por_falla.csv`. Gráficos en `output/dataset_v3/`:
- `ejemplos_por_causa.png`
- `anio_2024.png`

## 2. Modelo v3

Script: `entrenar_modelo_v3.py`. Salida: `modelo_rul_v3.joblib`. Mismo método de `train_model.py`. La falla se sitúa en la hora de inicio. Los dos pares de fallas solapadas se convierten en un solo punto cada uno, así que quedan 25 puntos.

- Entrenamiento: 1 de enero a 19 de octubre de 2024, con 19 fallas.
- Validación: 19 de octubre a 31 de diciembre, con 6 fallas.

Variables más importantes del v3:
1. Índice de estrés: 34,8 %.
2. Presión media de 60 min: 5,6 %.
3. Z-score de presión: 5,4 %.
4. Z-score de temperatura: 4,2 %.
5. Z-score de corriente: 3,5 %.

El modelo original dependía en un 74 % de la temperatura; el v3 reparte el peso entre las cuatro variables.

## 3. Comparación (AUC: 0,5 = azar, 1 = distingue siempre)

| Prueba | Original | v2 | v3 |
|---|---:|---:|---:|
| Validación propia (último 20 %) | 1,00 | 1,00 | **0,985** |
| MetroPT-3 (real), base fija | **0,66** | 0,65 | 0,58 |
| MetroPT-3 (real), base móvil | 0,62 | 0,61 | **0,66** |
| Industrial syn, 20 máquinas | 0,64 | 0,69* | 0,60 |
| Industrial syn, 5 CNC | 0,69 | 0,71* | 0,62 |
| SKAB (real), minuto | 0,45 | 0,37 | **0,67** |
| SKAB (real), segundo | 0,53 | 0,53 | **0,59** |
| Hydraulic Systems (real) | 0,78 | **0,91** | 0,79 |

\* No es independiente: el dataset v2 se construyó con Industrial syn.

### Validación propia del v3

| Métrica | Valor |
|---|---:|
| AUC del último día frente a operación normal | **0,985** |
| AUC de los últimos 3 días frente a operación normal | 0,76 |
| Fallas con alerta crítica previa | **5 de 6** |
| Tiempo normal en crítico | 0,03 % |
| Error medio absoluto del RUL | 24,7 h (R² 0,27) |

Detecta muy bien el último día. Cuanto más lejos está la falla, menos distingue, porque el desgaste dura solo de 0,5 a 6 días y antes de eso los sensores se ven normales.

### Alertas críticas y falsas alarmas en pruebas externas

| Prueba | Modelo | Fallas con alerta previa | Tiempo normal o lejano en crítico |
|---|---|---:|---:|
| MetroPT-3, base fija | original / v2 / v3 | 0 / 2 / 2 de 4 | 0 % / 6,8 % / 2,0 % |
| MetroPT-3, base móvil | original / v2 / v3 | 0 / 0 / 0 de 4 | 0 % / 0,7 % / 0,4 % |
| Industrial syn, 20 máquinas | original / v2 / v3 | 16 / 38 / 57 de 63 | 0,3 % / 2,3 % / **23,7 %** |
| Industrial syn, CNC | original / v2 / v3 | 9 / 13 / 15 de 15 | 0,8 % / 3,5 % / **29,3 %** |

En Industrial syn, el v3 pasa una cuarta parte del tiempo lejano en crítico. Sus alertas antes de las fallas no tienen valor ahí. Sus valores normales (presión 10–200, corriente 2–35 A con cambios de modo constantes), trasladados a la escala de la prensa, parecen desajustes o sobrecargas.

### Hydraulic Systems por componente (RUL predicho medio del v3, de óptimo a casi falla total)

| Componente | v3 |
|---|---|
| Enfriador (sube la temperatura) | **4,7 → 2,8 → 0,8** |
| Válvula | sin tendencia (2,7–2,9) |
| Fuga de la bomba | sin tendencia (2,6–2,9) |
| Acumulador | sin tendencia |

## Por qué el v3 no mejora con claridad en datos externos

1. **Las firmas que aprendió son específicas de esta prensa:** presión que baja, vibración que sube y corriente con temperatura. Una fuga de aire en un compresor (MetroPT-3) o un cambio de modo en una CNC (Industrial syn) no se parecen a esas firmas.
2. **Hay pocos ejemplos por causa:** 5 a 11 por causa en el entrenamiento.
3. **El comportamiento de los sensores sigue siendo simulado.** Solo las fallas son reales.

## Qué se puede afirmar

- El **dataset v3 es el más defendible**: fallas reales (fechas, horas, causas y paradas), una firma por causa y magnitudes justificadas por referencias técnicas.
- **Si las fallas de la prensa se desarrollan con tendencia durante días**, el método detecta el último día antes de la falla casi siempre (AUC 0,985) y avisa en 5 de 6 fallas, sin falsas alarmas.
- Ninguno de los tres modelos generaliza de forma fiable a **máquinas reales distintas**. La validación definitiva requiere **telemetría real de la propia prensa** junto con esta bitácora real.

## Versión anterior con fallas repentinas (supuesto descartado)

Con un 20 % de fallas casi repentinas (30–180 min de aviso), la validación propia del v3 daba un AUC de 0,67 y alerta en 2 de 6 fallas: detectaba al 100 % las fallas con desgaste de varios días y no las repentinas. Este resultado muestra que la capacidad de anticipar depende de que la falla tenga señales previas.

## Límites

- El comportamiento de los sensores es **simulado**. Las magnitudes, duraciones y formas son supuestos razonables, no mediciones de la prensa.
- La validación propia tiene solo 6 fallas.
- Que no haya fallas repentinas es un supuesto basado en la experiencia del usuario con esta prensa.

## Reproducción

```bash
python generar_dataset_v3.py
```

```bash
python entrenar_modelo_v3.py
```

```bash
python comparar_modelos_v3.py
```

Ningún script modifica los datos ni los modelos anteriores.
