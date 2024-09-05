import time
from itertools import combinations
import csv
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from RegLog.modelo import modelo_reglog
from RegLog.reglas import (
    regla_transformacion_caracteristicas,
    regla_normalizacion_datos,
    regla_balanceo_clases,
    regla_imputacion_avanzada,
    regla_escalado_minmax,
    regla_estandarizacion,
    regla_eliminacion_baja_varianza
)
from procesamiento_datos import procesamiento_integral
from sklearn.impute import SimpleImputer
import numpy as np
import os

# Configuración inicial
data_path = "dataset/train/diabetes_prediction_dataset.csv"
num_atributos = 8
nombre_variable_objetivo = 'diabetes'
columnas_a_ignorar = []
ruta_salida = "dataset/processed/data"
ruta_resultados = "resultados/DiabetesRegLogGraficos/"

if not os.path.exists(ruta_resultados):
    os.makedirs(ruta_resultados)

# Procesamiento de datos
X_train, X_test, y_train, y_test, preprocessor, features = procesamiento_integral(
    data_path, num_atributos, nombre_variable_objetivo, columnas_a_ignorar, ruta_salida
)

# Lista de reglas a aplicar
reglas = [
    regla_transformacion_caracteristicas,
    regla_normalizacion_datos,
    regla_imputacion_avanzada,
    regla_escalado_minmax,
    regla_estandarizacion,
    regla_eliminacion_baja_varianza,
    regla_balanceo_clases,  # Esta regla requiere tanto X como y
]

# Reglas que necesitan tanto X como y
reglas_con_y = [
    regla_balanceo_clases
]

# Crear archivo CSV para guardar resultados detallados
csv_file = os.path.join(ruta_resultados, "resultados_detallados.csv")
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([
        "Numero de reglas", "C", "Penalty", "Solver", "Iteraciones",
        "Accuracy", "Precision", "Recall", "F1 Score", "Media", "Tiempo"
    ])

def guardar_resultados(nombre_archivo, valor):
    ruta_completa = os.path.join(ruta_resultados, nombre_archivo)
    with open(ruta_completa, "a") as archivo:
        archivo.write(f"{valor}\n")

def guardar_resultados_detallados(num_reglas, c, pen, solv, iteraciones, accuracy, precision, recall, f1, media, tiempo):
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            num_reglas, c, pen, solv, iteraciones,
            accuracy, precision, recall, f1, media, tiempo
        ])

def imputar_valores_faltantes(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

def evaluar_modelo(X_train, X_test, y_train, y_test, c, pen, solv, iteraciones, reglas_aplicadas):
    inicio_modelo = time.time()
    model, _, _, _, _ = modelo_reglog(X_train, X_test, y_train, y_test,
                                      c=c, pen=pen, solv=solv, iteraciones=iteraciones)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    media = np.mean([accuracy, precision, recall, f1])
    tiempo_ejecucion = time.time() - inicio_modelo

    # Imprimir los resultados
    print(f"Reglas aplicadas: {[regla.__name__ for regla in reglas_aplicadas]}")
    print(f"Hiperparámetros: C={c}, Penalty={pen}, Solver={solv}, Iteraciones={iteraciones}")
    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
    print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")
    print("-" * 50)

    # Guardar los resultados en archivos CSV separados
    guardar_resultados("accuracy.csv", accuracy)
    guardar_resultados("precision.csv", precision)
    guardar_resultados("recall.csv", recall)
    guardar_resultados("f1.csv", f1)
    guardar_resultados("media.csv", media)

    # Guardar los resultados detallados en un solo archivo CSV
    guardar_resultados_detallados(len(reglas_aplicadas), c, pen, solv, iteraciones, accuracy, precision, recall, f1, media, tiempo_ejecucion)

    return media

def aplicar_combinaciones(X_train, X_test, y_train, y_test, hyperparameters):
    mejor_media_global = 0
    mejor_configuracion = None
    mejor_reglas_aplicadas = []

    # Probar todas las combinaciones posibles de reglas
    for r in range(1, len(reglas) + 1):
        for combinacion in combinations(reglas, r):
            X_train_modificado, y_train_modificado = X_train.copy(), y_train.copy()
            X_test_modificado = X_test.copy()

            for regla in combinacion:
                if regla in reglas_con_y:
                    X_train_modificado, y_train_modificado = regla(X_train_modificado, y_train_modificado)
                else:
                    X_train_modificado = regla(X_train_modificado)
                    X_test_modificado = regla(X_test_modificado)

            # Imputar valores faltantes después de aplicar las reglas
            X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train_modificado, X_test_modificado)

            # Evaluar todas las combinaciones de hiperparámetros
            for params in hyperparameters:
                media = evaluar_modelo(X_train_imputed, X_test_imputed, y_train_modificado, y_test,
                                       c=params['c'], pen=params['pen'], solv=params['solv'],
                                       iteraciones=params['iteraciones'], reglas_aplicadas=combinacion)
                if media > mejor_media_global:
                    mejor_media_global = media
                    mejor_configuracion = params
                    mejor_reglas_aplicadas = combinacion

    print("\n--- Final del proceso ---")
    print(f"Mejor conjunto de reglas: {[regla.__name__ for regla in mejor_reglas_aplicadas]}")
    print(f"Mejor configuración de hiperparámetros: {mejor_configuracion}")
    print(f"Mejor media de métricas alcanzada: {mejor_media_global:.4f}")

# Definir hiperparámetros iniciales
hyperparameters = [
    {'c': 0.01, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.1, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.5, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 1.0, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
]

# Ejecutar el proceso completo
aplicar_combinaciones(X_train, X_test, y_train, y_test, hyperparameters)
