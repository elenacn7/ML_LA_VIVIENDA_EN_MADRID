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

----------------------------------------------------


This project aims to predict housing prices in Madrid using supervised regression learning techniques. Based on a dataset with detailed information about the properties (location, type, size, number of bedrooms, bathrooms, etc.), several machine learning models are developed and compared to find the one that best predicts the market value. The goal is to estimate the price of a home in Madrid using supervised regression models. The objective is to find the model that best predicts the price based on the structural and location features of the property.

Technologies / libraries used:

Python 3.x

pandas, numpy

matplotlib, seaborn

scikit-learn

joblib

Implemented models:

Linear Regression

Decision Tree

Random Forest

Gradient Boosting

All models were trained using scikit-learn pipelines and evaluated with metrics such as MAE, RMSE, and R². The best performing model was Random Forest, achieving an R² of 0.692172 and an MAE of 351131.797517, demonstrating a good fit for accurately predicting housing prices in Madrid.
