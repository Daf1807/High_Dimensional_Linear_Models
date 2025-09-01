#!/usr/bin/env python
# coding: utf-8

# # Instructions

# In this assignment, you will estimate a hedonic pricing model using data on apartment prices in Poland. A hedonic pricing model estimates the value of a good based on its features. For apartments, the price depends on attributes such as area, number of rooms, distance to points of interests, etc.
# 
# Data is available at CausalAI-Course/Data/apartments.csv. Below, you will find a detailed description of each variable in the dataset. Make sure to carefully review these variable definitions

# # Dataset Description

# - price: Apartment price in PLN (Polish złoty).
# - month: Month of year
# - id: Unique identifier for each listing.
# - type: Type of apartment (e.g., flat, studio, etc.).
# - area: Total usable area of the apartment (in m²).
# - rooms: Number of rooms.
# - schoolDistance: Distance to the nearest school (in km).
# - clinicDistance: Distance to the nearest clinic or hospital (in km).
# - postOfficeDistance: Distance to the nearest post office (in km).
# - kindergartenDistance: Distance to the nearest kindergarten (in km).
# - restaurantDistance: Distance to the nearest restaurant (in km).
# - collegeDistance: Distance to the nearest college/university (in km).
# - pharmacyDistance: Distance to the nearest pharmacy (in km).
# - ownership: Type of ownership (e.g., freehold, cooperative).
# - buildingMaterial: Main material used for construction (e.g., brick, concrete).
# - hasParkingSpace: Boolean (1/0) indicating if a parking space is available.
# - hasBalcony: Boolean (1/0) indicating if the apartment has a balcony.
# - hasElevator: Boolean (1/0) indicating if the building has an elevator.
# - hasSecurity: Boolean (1/0) indicating if the building has security features.
# - hasStorageRoom: Boolean (1/0) indicating if the apartment has a storage room.

# # 3a Cleaning (2 points)

# In this section you'll need to do the following:
# 
# - Create a variable area2 that's the square of area (0.25 points)
# - Convert 'hasparkingspace', 'hasbalcony', 'haselevator', 'hassecurity', 'hasstorageroom' to dummy variables (where 'yes' 'no' get mapped to 1, 0) (0.75 points)
# - For each last digit of area (i.e. 0,1,...,9), create a dummy variable if the last digit of area happens to be that number. Name your variables accordingly (e.g. end_0, end_1, ...end_9). (1 point)

# ##### Cargar base de datos

# In[2]:


import pandas as pd

# If you uploaded apartments.csv to the same working directory:
df = pd.read_csv("/Users/rominarattoyanez/Downloads/apartments.csv")

# Show first rows
print(df.head())


# ##### Crear area2

# In[3]:


df["area2"] = df["area"] ** 2


# ##### Convertir columnas yes/no a dummies

# In[5]:


print(df.columns.tolist())


# In[6]:


cols = ["hasparkingspace", "hasbalcony", "haselevator", "hassecurity", "hasstorageroom"]

for c in cols:
    df[c] = df[c].str.lower().map({"yes": 1, "no": 0}) # (.str.lower() lo hace robusto a "Yes"/"YES"/"yes")
                                                       # .map() reemplaza yes por 1, no por 0, y el resto por NaN


# ##### Create dummy variables for last digit of area

# In[7]:


# Get last digit
df["area_last_digit"] = df["area"] % 10 # % es “el residuo de una división entera”.

# Create dummies
for d in range(10):
    df[f"end_{d}"] = (df["area_last_digit"] == d).astype(int)

# Drop helper column if you don’t want it
df = df.drop(columns=["area_last_digit"])


# # 3b Linear model estimation (4 points)

# 1 Regress 'price' against the following covariates:
# 
# - Area's last digit dummies (ommit 9 to have a base category)
# - Area, area squared
# - Distance from apartment to point of interest (such as school, clinic, postoffice, etc.)
# 'hasparkingspace', 'hasbalcony', 'haselevator', 'hassecurity', 'hasstorageroom'
# - Month, type, rooms, ownership, buildingmaterial (treat these as categorical variables)
# 
# Print a summary table and comment your results on the area's last digit dummy when the area's last digit is 0 (end_0).
# 
# (2 points)

# ##### Variables de interés

# In[9]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

last_digit_dummies = [f"end_{i}" for i in range(9)]  # Quitamos end_9 para que sea la categoría base

# Variables continuas
continuous_vars = ["area", "area2",
                   "schooldistance", "clinicdistance", "postofficedistance",
                   "kindergartendistance", "restaurantdistance",
                   "collegedistance", "pharmacydistance"]

# Variables binarias ya convertidas
binary_vars = ["hasparkingspace", "hasbalcony", "haselevator", "hassecurity", "hasstorageroom"]

# Variables categóricas → usamos C() en fórmula
categorical_vars = ["month", "type", "rooms", "ownership", "buildingmaterial"]


# In[11]:


print("Last digit dummies:")
print(last_digit_dummies)

print("\nContinuous variables:")
print(continuous_vars)

print("\nBinary variables:")
print(binary_vars)

print("\nCategorical variables:")
print(categorical_vars)


# ##### Regresión

# In[13]:


import statsmodels.formula.api as smf

# Fórmula simple: statsmodels crea dummies automáticamente para las categóricas con C()
model = smf.ols(
    "price ~ area + area2 \
             + schooldistance + clinicdistance + postofficedistance + kindergartendistance + restaurantdistance + collegedistance + pharmacydistance \
             + hasparkingspace + hasbalcony + haselevator + hassecurity + hasstorageroom \
             + C(month) + C(type) + C(rooms) + C(ownership) + C(buildingmaterial) \
             + end_0 + end_1 + end_2 + end_3 + end_4 + end_5 + end_6 + end_7 + end_8",
    data=df
).fit()

print(model.summary())


# In[18]:


with open("regression_results.txt", "w") as f:
    f.write(model.summary().as_text())


# In[16]:


"""
# Forma Corta

formula = (
    "price ~ " 
    + " + ".join(last_digit_dummies + continuous_vars + binary_vars) 
    + " + " 
    + " + ".join([f"C({c})" for c in categorical_vars])
)

model = smf.ols(formula=formula, data=df).fit()

print(model.summary())
"""


# ##### Comentario

# - El coeficiente de end_0 se interpreta en relación con la categoría base, que es end_9 (departamentos cuyo área termina en 9).
# - Signo positivo: los departamentos cuyo área termina en 0 tienen un precio mayor que los que terminan en 9, manteniendo constantes las demás variables del modelo (área, área², distancias, características, tipo de construcción, etc.).
# - Magnitud: el precio promedio es ≈ 27,600 (soles) más alto.
# - Significancia estadística: el p-valor = 0.000 indica que este efecto es robusto y difícilmente atribuible al azar.

# 2 Perform the same regression but this time by partialling-out. Your target parameter will be the one associated with end_0. Print a summary table and verify the coefficients are the same with both methods.
# 
# (2 points)

# In[37]:


from patsy import dmatrices
import statsmodels.api as sm
import numpy as np
import pandas as pd

# MISMA fórmula que tu OLS completo
formula = (
    "price ~ area + area2 "
    "+ schooldistance + clinicdistance + postofficedistance "
    "+ kindergartendistance + restaurantdistance "
    "+ collegedistance + pharmacydistance "
    "+ hasparkingspace + hasbalcony + haselevator + hassecurity + hasstorageroom "
    "+ C(month) + C(type) + C(rooms) + C(ownership) + C(buildingmaterial) "
    "+ end_0 + end_1 + end_2 + end_3 + end_4 + end_5 + end_6 + end_7 + end_8"
)

# 1) Construye Y y X EXACTAMENTE como los usa la fórmula (mismas dummies, misma base, mismas filas)
y_mat, X_mat = dmatrices(formula, data=df, return_type="dataframe")

# 2) OLS completo (referencia)
full = sm.OLS(y_mat, X_mat).fit()
beta_full = full.params["end_0"]
print("Coef (full OLS) end_0:", beta_full)

# 3) Partialling-out usando la MISMA X:
#    W = controles (incluye Intercept), x = columna end_0
W = X_mat.drop(columns=["end_0"])   # ya VIENE con 'Intercept', no agregues otra constante
x = X_mat["end_0"]
y = y_mat.iloc[:, 0]                # columna 'price' como Serie

# Residualiza con los mismos controles W
res_y = sm.OLS(y, W).fit().resid
res_x = sm.OLS(x, W).fit().resid

# Re-regresión de residuos (sin constante)
partial = sm.OLS(res_y, res_x).fit()
beta_partial = partial.params.iloc[0]
print("Coef (partialling-out) end_0:", beta_partial)

# 4) Verificación
print("¿Iguales?:", np.isclose(beta_full, beta_partial, rtol=1e-10, atol=1e-8))


# In[39]:


with open("regression_results_partialling_out.txt", "w") as f:
    f.write(partial.summary().as_text())


# # 3c Price premium for area that ends in 0-digit (3 points)

# In this section we'll attempt to see if apartments whose area ends at 0 are valued higher than what their features would suggest. Perform the following tasks.
# 
# 

# 1 Train the model
# Estimate the same linear regression model, but only using apartments whose area does not end in 0.
# (1.25 points)
# 
# 

# In[38]:


import statsmodels.formula.api as smf

# Filtrar: solo observaciones cuyo área NO termina en 0
df_no_end0 = df[df["end_0"] == 0].copy()

# Misma fórmula de antes
formula = (
    "price ~ area + area2 "
    "+ schooldistance + clinicdistance + postofficedistance "
    "+ kindergartendistance + restaurantdistance "
    "+ collegedistance + pharmacydistance "
    "+ hasparkingspace + hasbalcony + haselevator + hassecurity + hasstorageroom "
    "+ C(month) + C(type) + C(rooms) + C(ownership) + C(buildingmaterial) "
    "+ end_1 + end_2 + end_3 + end_4 + end_5 + end_6 + end_7 + end_8"
)

# Ajustar el modelo solo con df_no_end0
model_no_end0 = smf.ols(formula, data=df_no_end0).fit()

print(model_no_end0.summary())


# 2 Predict prices
# Using the estimated coefficients from step 1, predict apartment prices for the entire sample, including those apartments whose area ends in 0.
# (1.25 points)

# In[40]:


# Predecir precios para toda la muestra con el modelo entrenado en df_no_end0
df["predicted_price_no_end0"] = model_no_end0.predict(df)

# Revisar algunas predicciones
print(df[["price", "predicted_price_no_end0"]].head(10))


# 3 Compare averages
# For apartments whose area ends in 0, compute both the average actual price and the average predicted price.
# Based on this comparison, try to determine whether apartments with areas ending in 0 are sold at a higher price than what the model predicts. (You don't need to make a statistical tests, just say a guess based on your results)
# (0.5 points)

# In[41]:


# Filtrar solo los apartamentos cuyo área termina en 0
df_end0 = df[df["end_0"] == 1]

# Calcular promedios
avg_actual = df_end0["price"].mean()
avg_predicted = df_end0["predicted_price_no_end0"].mean()

print("Precio promedio real (end_0 = 1):", avg_actual)
print("Precio promedio predicho (end_0 = 1):", avg_predicted)

# Diferencia
print("Diferencia (real - predicho):", avg_actual - avg_predicted)


# ##### Comentario

# Como el promedio real < promedio predicho → en realidad se venden más baratos de lo esperado.
