import time
from DecisionTree.modelo import modelo_arbol_decision
from DecisionTree.reglas_arboles import (
    regla_imputacion_avanzada,
    regla_normalizacion_datos,
    regla_balanceo_clases,
    regla_remocion_outliers,
    regla_penalizacion_clase_desbalanceada,
    regla_escalado_minmax
)
from procesamiento_datos import procesamiento_integral
import numpy as np
import os

# configuracion inicial para las rutas y dataset
data_path = "dataset/train/nba_logreg.csv"
num_atributos = 20
nombre_variable_objetivo = 'TARGET_5Yrs'
columnas_a_ignorar = []
ruta_salida = "dataset/processed/data"
ruta_resultados = "resultados/pruebaNBAArboles/"

# crea el directorio de resultados si no existe
if not os.path.exists(ruta_resultados):
    os.makedirs(ruta_resultados)

# procesamiento de los datos: separa en conjunto de entrenamiento y test
X_train, X_test, y_train, y_test, preprocessor, features = procesamiento_integral(
    data_path, num_atributos, nombre_variable_objetivo, columnas_a_ignorar, ruta_salida
)

# lista de reglas de preprocesamiento a aplicar
reglas = [
    regla_imputacion_avanzada,
    regla_normalizacion_datos,
    regla_balanceo_clases,
    regla_penalizacion_clase_desbalanceada,
    regla_escalado_minmax
]

# reglas que requieren tanto X como y para aplicarse
reglas_con_y = [
    regla_balanceo_clases,
    regla_remocion_outliers,
    regla_penalizacion_clase_desbalanceada
]

# funcion para guardar los resultados de las métricas en archivos CSV
def guardar_resultados(nombre_archivo, valor):
    ruta_completa = os.path.join(ruta_resultados, nombre_archivo)
    with open(ruta_completa, "a") as archivo:
        archivo.write(f"{valor}\n")

# modifica la funcion para entrenar el modelo aplicando reglas de preprocesamiento
def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test):
    mejor_media = 0
    mejor_configuracion = None
    reglas_aplicadas = []

    # entrena el modelo base sin reglas aplicadas
    print("Entrenando modelo base sin reglas...")
    inicio_modelo = time.time()
    model_result = modelo_arbol_decision(X_train, X_test, y_train, y_test, max_depth=10, min_samples_split=5)
    tiempo_ejecucion = time.time() - inicio_modelo

    # calcula las métricas del modelo
    accuracy, precision, recall, f1 = model_result[1], model_result[2], model_result[3], model_result[4]
    media = np.mean([accuracy, precision, recall, f1])

    print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
    print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

    # guarda los resultados de las métricas en los CSV
    guardar_resultados("accuracy.csv", accuracy)
    guardar_resultados("precision.csv", precision)
    guardar_resultados("recall.csv", recall)
    guardar_resultados("f1.csv", f1)
    guardar_resultados("media.csv", media)

    # si las métricas mejoran, se actualiza la mejor configuración
    if media > mejor_media:
        mejor_media = media
        mejor_configuracion = ("Sin reglas", None)
        print(f"Nueva mejor media de métricas: {mejor_media:.4f}")

    # aplica reglas de una en una y evalua el modelo
    for i in range(len(reglas)):
        mejor_regla = None
        for regla in reglas:
            if regla not in reglas_aplicadas:
                print(f"\nAplicando regla: {regla.__name__}")

                # algunas reglas requieren X e y (como penalizacion de clases desbalanceadas)
                if regla in reglas_con_y:
                    if regla == regla_penalizacion_clase_desbalanceada:
                        X_train_modificado, y_train_modificado, X_test_modificado, y_test_modificado = regla(X_train.copy(), y_train.copy(), X_test.copy(), y_test.copy())
                    else:
                        X_train_modificado, y_train_modificado = regla(X_train.copy(), y_train.copy())
                        X_test_modificado = X_test.copy()
                        y_test_modificado = y_test
                else:
                    # otras reglas solo modifican X
                    X_train_modificado = regla(X_train.copy())
                    X_test_modificado = regla(X_test.copy())
                    y_train_modificado = y_train
                    y_test_modificado = y_test

                # reentrena y evalua el modelo tras aplicar la regla
                inicio_modelo = time.time()
                model_result = modelo_arbol_decision(X_train_modificado, X_test_modificado, y_train_modificado, y_test_modificado,
                                                     max_depth=10, min_samples_split=5)
                tiempo_ejecucion = time.time() - inicio_modelo

                # calcula las nuevas métricas
                accuracy, precision, recall, f1 = model_result[1], model_result[2], model_result[3], model_result[4]
                media = np.mean([accuracy, precision, recall, f1])

                print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

                # guarda los resultados en los CSV
                guardar_resultados("accuracy.csv", accuracy)
                guardar_resultados("precision.csv", precision)
                guardar_resultados("recall.csv", recall)
                guardar_resultados("f1.csv", f1)
                guardar_resultados("media.csv", media)

                # actualiza la mejor regla si las métricas mejoran
                if media > mejor_media:
                    mejor_media = media
                    mejor_regla = regla
                    print(f"Nueva mejor media con regla {regla.__name__}: {mejor_media:.4f}")

        if mejor_regla:
            reglas_aplicadas.append(mejor_regla)
            print(f"Regla {mejor_regla.__name__} añadida al conjunto.")
        else:
            print(f"No se logró una mejora en la iteración {i + 1}. Deteniendo iteraciones.")
            break

    print("\n--- Final del proceso ---")
    print(f"Mejor conjunto de reglas: {[regla.__name__ for regla in reglas_aplicadas]}")
    print(f"Mejor media de métricas alcanzada: {mejor_media:.4f}")

# ejecuta el proceso de entrenamiento y evaluación con las reglas
aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test)
