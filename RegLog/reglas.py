from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
from scipy import stats

# eliminar columnas con alto porcentaje de valores faltantes
def eliminar_columnas_con_muchos_na(data, threshold=0.5):
    cols_to_drop = [col for col in data.columns if data[col].isnull().mean() > threshold]
    return data.drop(columns=cols_to_drop)

# selecciona las caracteristicas mas importantes con un modelo de arboles
def seleccionar_caracteristicas_importantes(X, y, n_features=10):
    model = ExtraTreesClassifier()
    model.fit(X, y)
    importances = model.feature_importances_
    indices = np.argsort(importances)[-n_features:]
    return X.iloc[:, indices].columns

# transforma las caracteristicas con alto sesgo usando logaritmo
def transformar_caracteristicas_sesgadas(X):
    from scipy.stats import skew
    skewed_features = X.apply(lambda x: skew(x.dropna()))  # calcula el sesgo
    skewness = skewed_features[skewed_features > 0.5]
    skewness = skewness.index
    # aplica logaritmo natural a las caracteristicas sesgadas
    for feature in skewness:
        if X[feature].isnull().any():
            print(f"Advertencia: La caracteristica '{feature}' contiene valores NaN")
        X[feature] = X[feature].apply(lambda x: np.log1p(x) if pd.notnull(x) and x >= 0 else np.nan)
    return X

# normaliza las caracteristicas numericas (media 0, desviacion estandar 1)
def normalizar_datos(X):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled)

# regla para esclado entre 0 y 1
def regla_escalado_minmax(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    return pd.DataFrame(X_scaled)

# regla para estandarizar los datos
def regla_estandarizacion(X):
    scaler = StandardScaler()
    X_standardized = scaler.fit_transform(X)
    return pd.DataFrame(X_standardized)

# elimina caracteristicas con baja varianza
def regla_eliminacion_baja_varianza(X, threshold=0.01):
    from sklearn.feature_selection import VarianceThreshold
    selector = VarianceThreshold(threshold)
    X_high_variance = selector.fit_transform(X)
    return pd.DataFrame(X_high_variance)

# elimina caracteristicas con alta correlacion
def regla_eliminacion_correlacion_alta(X, threshold=0.9):
    corr_matrix = X.corr().abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
    return X.drop(columns=to_drop)

# selecciona el mejor tipo de penalizacion (L1 o L2) para el modelo
def seleccionar_mejor_penalty(X, y):
    param_grid = {'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
    grid_search = GridSearchCV(LogisticRegression(C=1, max_iter=100), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_['penalty']

# ajusta el parametro de regularizacion C mediante GridSearchCV
def ajustar_C(X, y, penalty):
    param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': [penalty], 'solver': ['liblinear']}
    grid_search = GridSearchCV(LogisticRegression(max_iter=100), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, y)
    return grid_search.best_params_['C']


# Regla eliminada ya que se sabe que causa problemas
# def remover_outliers(X, y):
#     X, y = X.reset_index(drop=True), y.reset_index(drop=True)
#     z_scores = stats.zscore(X)
#     abs_z_scores = np.abs(z_scores)
#     mask = (abs_z_scores < 3).all(axis=1)
#     X_filtered = X[mask].copy()
#     y_filtered = y[mask].copy()
#     return X_filtered, y_filtered

# def crear_variables_interaccion(X):
#     poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
#     X_interaction = poly.fit_transform(X)
#     return pd.DataFrame(X_interaction)

# validacion cruzada estratificada para mantener distribucion de clases
def validacion_cruzada_estratificada(X, y):
    skf = StratifiedKFold(n_splits=5)
    return skf.split(X, y)

# balancea clases usando SMOTE
def balancear_clases(X, y):
    imputer = SimpleImputer(strategy='mean')
    X_imputed = pd.DataFrame(imputer.fit_transform(X))
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_imputed, y)
    return X_res, y_res

# aplicar penalizacion para clases desbalanceadas en regresion logistica
def aplicar_penalizacion_clase_desbalanceada(C, penalty, solver, max_iter):
    return LogisticRegression(C=C, penalty=penalty, solver=solver, max_iter=max_iter, class_weight='balanced')


# evaluar el modelo usando validacion cruzada
def evaluar_modelo_cv(X, y, model):
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print("Cross-Validation Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    return scores.mean()

# imputacion por mean
def regla_imputacion_avanzada(data, threshold=0.5):
    imputer = SimpleImputer(strategy='mean')
    return pd.DataFrame(imputer.fit_transform(data))

def regla_transformacion_caracteristicas(X):
    return transformar_caracteristicas_sesgadas(X)

def regla_normalizacion_datos(X):
    return normalizar_datos(X)

def regla_mejor_penalty(X, y):
    return seleccionar_mejor_penalty(X, y)

def regla_ajuste_C(X, y, penalty):
    return ajustar_C(X, y, penalty)

def regla_validacion_cruzada_estratificada(X, y):
    return validacion_cruzada_estratificada(X, y)

def regla_balanceo_clases(X, y):
    return balancear_clases(X, y)

def regla_penalizacion_clase_desbalanceada(C, penalty, solver, max_iter):
    return aplicar_penalizacion_clase_desbalanceada(C, penalty, solver, max_iter)

def regla_evaluacion_cv(X, y, model):
    return evaluar_modelo_cv(X, y, model)
