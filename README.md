Este proyecto tiene como objetivo predecir el precio de viviendas en Madrid utilizando técnicas de aprendizaje supervisado de regresión. 
A partir de un conjunto de datos con información detallada de los inmuebles (ubicación, tipología, medida, número de habitaciones, baños, etc.), se desarrollan y comparan varios modelos de machine learning para encontrar el que mejor prediga el valor de mercado.
Este proyecto busca estimar el precio de una vivienda en Madrid utilizando modelos de regresión supervisada. 
El objetivo es encontrar el modelo que mejor prediga el precio basándose en características estructurales y de localización del inmueble.

Tecnologías / librerías utilizadas:
- Python 3.x
- pandas, numpy
- matplotlib, seaborn
- scikit-learn
- joblib

Modelos implementados:
- Regresión Lineal
- Árbol de Decisión
- Random Forest
- Gradient Boosting

Todos los modelos fueron entrenados usando `pipelines` de scikit-learn y evaluados con métricas como MAE, RMSE y R².
El modelo con mejor rendimiento fue RandomForest, obteniendo un R² de 0.692172 y un MAE de 351131.797517, demostrando un buen ajuste para predecir precios de viviendas en Madrid con alta precisión.
