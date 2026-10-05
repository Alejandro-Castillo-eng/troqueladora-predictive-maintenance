# Error del RUL por tiempo restante: validación de 2025

El modelo estima con menos error, en conjunto, en las tomas cercanas a una falla. Sin embargo, el error no disminuye de forma continua al acercarse la falla: esta evaluación no respalda la afirmación «cuanto más cerca, siempre más preciso».

## Método

Se utilizó `modelo_rul.joblib` sin volver a entrenar. Se reprodujo la separación cronológica del proyecto: primer 80 % de 2025 para entrenamiento y último 20 % para validación, del 16 de octubre al 26 de diciembre de 2025. No se utilizaron datos de 2026. Las predicciones se limitaron a 0–10 días para reproducir la evaluación de `train_model.py`; 13 valores se recortaron. Se reprodujo el MAE global guardado: 1.1300395018142522 días operativos.

Cada intervalo se define mediante el RUL de referencia calculado desde la siguiente falla registrada, nunca mediante el RUL predicho. Un día operativo equivale a 600 muestras, es decir, 10 horas de operación bajo el supuesto de una muestra por minuto. Las muestras del instante de falla se incluyen con RUL cero. El error es la diferencia absoluta entre predicción y referencia; se presenta su promedio (MAE).

De las 43 200 filas de validación:

- 17 400 tienen siguiente falla registrada y RUL menor de 10 días: constituyen la comparación principal por intervalos.
- 10 800 tienen siguiente falla registrada a 10 días o más: su objetivo está limitado a 10 y no permite medir el error del tiempo restante completo. Se conservan en el CSV de predicciones, fuera de la comparación principal.
- 15 000 están después de la última falla registrada: no se conoce una siguiente falla, aunque el código les asigna 10 días. También se conservan separadamente y se excluyen de los intervalos.

## Resultados

| Tiempo restante de referencia (días operativos) | Filas | Fallas representadas | Error promedio |
|---|---:|---:|---:|
| 0 a 1, ambos incluidos | 2404 | 4 | 41.12 minutos |
| Más de 1 hasta 2 | 2400 | 4 | 21.29 minutos |
| Más de 2 hasta 3 | 2399 | 4 | 24.44 minutos |
| Más de 3 hasta 4 | 1800 | 3 | 29.01 minutos |
| Más de 4 hasta 5 | 1800 | 3 | 13.89 horas |
| Más de 5 hasta 6 | 1799 | 3 | 31.43 horas |
| Más de 6 hasta 7 | 1200 | 2 | 21.91 horas |
| Más de 7 hasta 8 | 1200 | 2 | 13.17 horas |
| Más de 8 hasta 9 | 1200 | 2 | 3.69 horas |
| Más de 9 y menos de 10 | 1198 | 2 | 9.13 horas |

Todos los tiempos son operativos. La tabla muestra promedios, no límites máximos de error ni intervalos de confianza. El porcentaje dentro de una tolerancia, RMSE, sesgo y percentil 90 se conservan en `errores_por_intervalo.csv`.

En conjunto, las 7203 filas hasta 3 días tienen un MAE de 28.96 minutos. Las 10197 filas con RUL mayor de 3 y menor de 10 días tienen un MAE de 13.72 horas. Este último grupo es distinto del usado al calcular el MAE global, porque excluye objetivos limitados a 10 y filas sin siguiente falla.

## Qué podemos afirmar

En esta validación, el grupo cercano (hasta tres días operativos) tiene menor error medio que el grupo con RUL mayor de tres y menor de diez días. También se observa bajo error entre tres y cuatro días: el corte de tres días es un criterio del código, no un límite descubierto por el modelo.

No hay mejora estrictamente progresiva al acercarse la falla. El último día tiene más error que el intervalo de uno a dos días. Los intervalos alejados tampoco siguen un orden creciente: de ocho a nueve días hay menos error que de cinco a seis días.

La revisión por evento también muestra excepciones: para la falla del 22 de noviembre, el error del último día es 90.95 minutos y el de uno a dos días es 23.68 minutos. No se interpreta una caída o aumento puntual como efecto causal de la cercanía.

La evaluación principal representa solo cuatro fallas: 28 de octubre, 22 y 25 de noviembre, y 1 de diciembre de 2025. Las filas consecutivas están correlacionadas y no son miles de fallas independientes. Los intervalos más alejados incluyen menos eventos (dos o tres); sus promedios no comparan siempre las mismas fallas. Las etiquetas cuentan muestras, no tiempo calendario; no confirman cuánto se acertaría en otra máquina, otro año o datos reales. No se han evaluado intervalos estadísticos de confianza ni un estudio independiente adicional.

Esta medición describe el error de estimación del RUL. No demuestra por sí sola que las alertas anticipen correctamente todas las fallas: faltaría evaluar avisos por evento, antelación, omisiones y falsas alarmas con criterios definidos antes de analizar los resultados.

## Archivos y reproducción

- `error_por_intervalo.png`: gráfico de MAE por intervalo.
- `errores_por_intervalo.csv`: métricas agregadas por intervalo.
- `errores_por_falla_e_intervalo.csv`: métricas por evento e intervalo.
- `predicciones_validacion_2025.csv`: todas las predicciones, referencias y grupos de exclusión.
- `resumen.json`: método, identificación del modelo y resumen numérico.

Para reproducir: ejecutar `python evaluar_intervalos_rul.py` en la carpeta del proyecto. El script guarda resultados de evaluación y no modifica los datasets ni el modelo.

Definición del MAE: [documentación oficial de scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html#mean-absolute-error).
