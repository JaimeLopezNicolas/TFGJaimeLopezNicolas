import pandas as pd
import time
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from DecisionTree.modelo import modelo_arbol_decision
from DecisionTree.reglas_arboles import (
    regla_imputacion_avanzada,
    regla_normalizacion_datos,
    regla_balanceo_clases,
    regla_remocion_outliers,
    regla_inclusion_variables_interaccion,
    regla_penalizacion_clase_desbalanceada,
    regla_escalado_minmax
)
from procesamiento_datos import procesamiento_integral
import numpy as np
import os
from itertools import product

# Configuración inicial
data_path = "dataset/train/diabetes_prediction_dataset.csv"
num_atributos = 8
nombre_variable_objetivo = 'diabetes'
columnas_a_ignorar = []
ruta_salida = "dataset/processed/data"
ruta_resultados = "resultados/DiabetesArbolesGraficos/"

if not os.path.exists(ruta_resultados):
    os.makedirs(ruta_resultados)

# Procesamiento de datos
X_train, X_test, y_train, y_test, preprocessor, features = procesamiento_integral(
    data_path, num_atributos, nombre_variable_objetivo, columnas_a_ignorar, ruta_salida
)

# Lista de reglas a aplicar
reglas = [
    regla_imputacion_avanzada,
    regla_normalizacion_datos,
    regla_balanceo_clases,
    regla_penalizacion_clase_desbalanceada,
    regla_escalado_minmax
]

# Reglas que necesitan tanto X como y
reglas_con_y = [
    regla_balanceo_clases,
    regla_remocion_outliers,
    regla_penalizacion_clase_desbalanceada
]

# Hiperparámetros a probar
hiperparametros = {
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

# CSV de resultados
csv_resultados = os.path.join(ruta_resultados, "resultados_arboles.csv")

def guardar_resultados_csv(resultados, nombre_archivo):
    if not os.path.exists(nombre_archivo):
        df = pd.DataFrame(resultados)
        df.to_csv(nombre_archivo, index=False)
    else:
        df = pd.DataFrame(resultados)
        df.to_csv(nombre_archivo, mode='a', header=False, index=False)

def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test):
    resultados = []

    # Modelo base sin reglas aplicadas
    print("Entrenando modelo base sin reglas...")

    for combinacion in product(hiperparametros['max_depth'], hiperparametros['min_samples_split']):
        max_depth, min_samples_split = combinacion
        inicio_modelo = time.time()
        model_result = modelo_arbol_decision(X_train, X_test, y_train, y_test, max_depth=max_depth, min_samples_split=min_samples_split)
        tiempo_ejecucion = time.time() - inicio_modelo

        accuracy, precision, recall, f1 = model_result[1], model_result[2], model_result[3], model_result[4]
        media = np.mean([accuracy, precision, recall, f1])

        print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}")
        print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
        print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

        # Guardar los resultados
        resultados.append({
            'Numero de reglas': 0,
            'Reglas aplicadas': 'Ninguna',
            'Max Depth': max_depth,
            'Min Samples Split': min_samples_split,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'Media': media,
            'Tiempo de ejecucion': tiempo_ejecucion
        })

        # Guardar en CSV inmediatamente
        guardar_resultados_csv([resultados[-1]], csv_resultados)

    # Aplicar reglas de una en una y probar hiperparámetros
    for num_reglas in range(1, len(reglas) + 1):
        reglas_aplicadas = []

        for combinacion_reglas in product(reglas, repeat=num_reglas):
            # Aplicar las reglas a los datos
            X_train_modificado, X_test_modificado, y_train_modificado, y_test_modificado = X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()

            for regla in combinacion_reglas:
                if regla in reglas_con_y:
                    if regla == regla_penalizacion_clase_desbalanceada:
                        X_train_modificado, y_train_modificado, X_test_modificado, y_test_modificado = regla(X_train_modificado, y_train_modificado, X_test_modificado, y_test_modificado)
                    else:
                        X_train_modificado, y_train_modificado = regla(X_train_modificado, y_train_modificado)
                else:
                    X_train_modificado = regla(X_train_modificado)
                    X_test_modificado = regla(X_test_modificado)

            # Probar todas las combinaciones de hiperparámetros
            for combinacion in product(hiperparametros['max_depth'], hiperparametros['min_samples_split']):
                max_depth, min_samples_split = combinacion
                inicio_modelo = time.time()
                model_result = modelo_arbol_decision(X_train_modificado, X_test_modificado, y_train_modificado, y_test_modificado, max_depth=max_depth, min_samples_split=min_samples_split)
                tiempo_ejecucion = time.time() - inicio_modelo

                accuracy, precision, recall, f1 = model_result[1], model_result[2], model_result[3], model_result[4]
                media = np.mean([accuracy, precision, recall, f1])

                print(f"Reglas aplicadas: {[r.__name__ for r in combinacion_reglas]}")
                print(f"Max Depth: {max_depth}, Min Samples Split: {min_samples_split}")
                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

                # Guardar los resultados
                resultados.append({
                    'Numero de reglas': len(combinacion_reglas),
                    'Reglas aplicadas': ', '.join([r.__name__ for r in combinacion_reglas]),
                    'Max Depth': max_depth,
                    'Min Samples Split': min_samples_split,
                    'Accuracy': accuracy,
                    'Precision': precision,
                    'Recall': recall,
                    'F1 Score': f1,
                    'Media': media,
                    'Tiempo de ejecucion': tiempo_ejecucion
                })

                # Guardar en CSV inmediatamente
                guardar_resultados_csv([resultados[-1]], csv_resultados)

    print(f"\nResultados guardados en: {csv_resultados}")


# Ejecutar el proceso completo
aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test)
