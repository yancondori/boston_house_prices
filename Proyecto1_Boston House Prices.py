
# Proyecto 1.b :  "Boston Housing Prices"
# Boston House Prices-Advanced Regression Techniques

# Tarea 1: Comprensión del Conjunto de Datos
# Descargar el conjunto de datos "Boston Housing Prices" desde Kaggle.
# RPTA: La datos se descargaron de  https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data/

# Explorar la documentación y la descripción del conjunto de datos para comprender sus características y variables.
# Cargar el conjunto de datos en un entorno de análisis (Colab) y examinar las primeras filas para una comprensión inicial.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

boston_hp = pd.read_csv('boston.csv')
boston_hp.info()
boston_hp.head()

#  A continuacion la descripcion de las variables del dataframe 
# 1) CRIM: per capita crime rate by town
# 2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
# 3) INDUS: proportion of non-retail business acres per town
# 4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# 5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
# 6) RM: average number of rooms per dwelling
# 7) AGE: proportion of owner-occupied units built prior to 1940
# 8) DIS: weighted distances to five Boston employment centres
# 9) RAD: index of accessibility to radial highways
# 10) TAX: full-value property-tax rate per $10,000 [$/10k]
# 11) PTRATIO: pupil-teacher ratio by town
# 12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# 13) LSTAT: % lower status of the population

# USE OR DONT'T A PREFIX
# If you're calling a method on an object (an instance of a class), you won't use the library prefix. The method is part of the object's interface:
# 'df' is a DataFrame object, and 'head' is a method of DataFrame.
# df.head()

# Tarea 2: Análisis Exploratorio de Datos (EDA)

# A. Realizar un análisis descriptivo de las variables, incluyendo estadísticas resumidas y visualización de distribuciones individuales.
# B. RPTA: Para poder visualizar las distribuciones utilizamos histogramas que cuentan los los valores de que toman cada una de las variables 
#opcion1
boston_hp.describe()

import seaborn as sns

for col in boston_hp.columns:
    sns.histplot(boston_hp[col], bins=30, kde=False)
    plt.title(col)
    plt.xlabel(col)
    plt.tight_layout() #ajustando tamano de plot
    plt.show()

#opcion2
# for col in boston_hp.columns:
#     plt.hist(boston_hp[col], bins=30)
#     plt.title(col)
#     plt.xlabel(col)
#     plt.show()


# Identificar valores atípicos (outliers) y decidir cómo manejarlos (eliminarlos, imputarlos por la media, etc).
# Set the size of the overall figure
plt.figure(figsize=(10, 20))

# Hacemos un Loop a traves de las columnas del DataFrame para crear boxplot de cada una de las variables
for i, col in enumerate(boston_hp.columns):
    plt.subplot(len(boston_hp.columns), 1, i+1)
    sns.boxplot(x=boston_hp[col], orient='h')
    plt.title(f'Boxplot de {col}')

plt.tight_layout()
plt.show() 

# Graficamente observamos que las variables con mayor outliers son B (Proportion of blacks by town), 
# ZN (Proportion of residential land zoned for lots over 25,000 sq.ft.), CRIM(Per capita crime rate by town), 
# MEDV(Median value of owner-occupied homes), CHAS(Charles River dummy variable) and RM (Average number of rooms per dwelling)

# opcion 2
# boston_hp.boxplot(figsize=(10, 20), vert=False)
# plt.tight_layout()
# plt.show()

# Opcion 3

# Create boxplots for each variable to visualize the outliers
fig, axes = plt.subplots(nrows=len(boston_df.columns), ncols=1, figsize=(10, 4*len(boston_df.columns)))

# Plotting each column
for i, col in enumerate(boston_df.columns):
    axes[i].boxplot(boston_df[col], vert=False)
    axes[i].set_title(f'Boxplot of {col}')

plt.tight_layout()
plt.show()


#Vamos a utilizar la metodologia Capping (Winsorizing), el cual consiste en reemplazar o imputar los valores outliers
#con el valor mas cercano que nos es outlier

from scipy.stats.mstats import winsorize

def cap_outliers(series, lower_quantile=0.01, upper_quantile=0.99):
    lower_bound = series.quantile(lower_quantile)
    upper_bound = series.quantile(upper_quantile)
    return series.apply(lambda x: max(min(x, upper_bound), lower_bound))

capped_df = boston_hp.copy()
for col in capped_df.columns:
    capped_df[col] = cap_outliers(capped_df[col])


# Visualizar relaciones entre variables mediante gráficos de dispersión y matriz de correlación.


# Tarea 3: Preprocesamiento de Datos

# Manejar valores faltantes si es necesario (imputación o eliminación).
# Escalar las variables numéricas si es necesario.
# Dividir el conjunto de datos en un conjunto de entrenamiento y un conjunto de prueba.
 

# Tarea 4 Comunicación de Resultados

# Crear un informe técnico que incluya una descripción detallada de los gráficos y visualizaciones.
# Preparar una presentación para comunicar los resultados a un público no técnico.
 

# Tarea 5: Documentación y Código

# Documentar el código de manera clara y detallada.
# Utilizar control de versiones para rastrear cambios en el código. (Opcional)
 

# Tarea 6: Entrega Final

# Entregar el informe técnico, la presentación y el código del proyecto.
