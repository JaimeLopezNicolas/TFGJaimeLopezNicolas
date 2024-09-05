from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from scipy import stats
import numpy as np
import pandas as pd

# regla para imputar valores faltantes usando la media
def regla_imputacion_avanzada(data):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# regla para normalizar datos usando StandardScaler
def regla_normalizacion_datos(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# regla para aplicar escalado Min-Max
def regla_escalado_minmax(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# regla para eliminar outliers usando z-score
def regla_remocion_outliers(X, y):
    z_scores = np.abs(stats.zscore(X))
    mask = (z_scores < 3).all(axis=1)
    return X[mask], y[mask]

# regla para crear variables de interaccion (combinacion de caracteristicas)
def regla_inclusion_variables_interaccion(X):
    poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
    return pd.DataFrame(poly.fit_transform(X), columns=poly.get_feature_names_out())

# regla para balancear clases usando SMOTE
def regla_balanceo_clases(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

# regla para aplicar penalizacion de clases desbalanceadas
def regla_penalizacion_clase_desbalanceada(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(class_weight='balanced')
    model.fit(X_train, y_train)
    return X_train, y_train, X_test, y_test

# funcion para evaluar el modelo con validacion cruzada
def evaluar_modelo_cv(X, y, model):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Cross-Validation Accuracy: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})")
    return scores.mean()
