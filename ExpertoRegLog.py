import time
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
# Define la ruta del dataset, el número de atributos, la variable objetivo y las rutas para salida de datos
data_path = "dataset/train/diabetes_prediction_dataset.csv"
num_atributos = 8  # numero de columnas menos una
nombre_variable_objetivo = 'diabetes'  # variable objetivo
columnas_a_ignorar = []  # añadir nombres de las columnas tipo string --> ["columna_1","columna_2"]
ruta_salida = "dataset/processed/data"  # añadir la ruta de salida
ruta_resultados = "resultados/pruebaDiabetesRegLog/"  # añadir la ruta para los CSVs

if not os.path.exists(ruta_resultados):
    os.makedirs(ruta_resultados)

# Procesamiento de datos (llamada al programa)
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
    regla_balanceo_clases,  # esta regla requiere tanto X como y
]

# reglas que necesitan X e y
reglas_con_y = [
    regla_balanceo_clases
]

# funcion que guarda los resultados de las metricas en archivos CSV para analisis posterior
def guardar_resultados(nombre_archivo, valor):
    ruta_completa = os.path.join(ruta_resultados, nombre_archivo)
    with open(ruta_completa, "a") as archivo:
        archivo.write(f"{valor}\n")

# funcion que rellena los valores faltantes en el conjunto de datos usando la media de cada columna
def imputar_valores_faltantes(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed

# funcion que ajusta los hiperparámetros buscando la mejor combinación que optimice el modelo
def refinar_hiperparametros(X_train, X_test, y_train, y_test, mejor_C, mejor_penalty, mejor_solver):
    steps = [0.005, 0.001]
    mejor_media = 0
    mejor_configuracion = (mejor_C, mejor_penalty, mejor_solver)

    # aqui itera sobre diferentes configuraciones de C y num de iteraciones
    for step in steps:
        for c in [mejor_C - step, mejor_C, mejor_C + step]:
            for iteraciones in [5000 - 1000, 5000, 5000 + 1000]:
                print(f"\nRefinando: C={c}, Iteraciones={iteraciones}")

                # se entrena el modelo con la configuracion actual
                inicio_modelo = time.time()
                model, _, _, _, _ = modelo_reglog(X_train, X_test, y_train, y_test,
                                                  c=c, pen=mejor_penalty, solv=mejor_solver,
                                                  iteraciones=iteraciones)
                y_pred = model.predict(X_test)

                # se calculan las metricas de rendimiento
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                media = np.mean([accuracy, precision, recall, f1])
                tiempo_ejecucion = time.time() - inicio_modelo

                print(
                    f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

                # guardamos metricas en los CSVs
                guardar_resultados("accuracy.csv", accuracy)
                guardar_resultados("precision.csv", precision)
                guardar_resultados("recall.csv", recall)
                guardar_resultados("f1.csv", f1)
                guardar_resultados("media.csv", media)

                # si las metricas mejoran, se actualiza el mejr modelo
                if media > mejor_media:
                    mejor_media = media
                    mejor_configuracion = (c, mejor_penalty, mejor_solver)
                    print(f"Nueva mejor media refinada: {mejor_media:.4f}")

    return mejor_configuracion

# funcion que aplica las reglas de preprocesamiento, entrena los modelos y optimiza los hiperparametros
def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters):
    mejor_media = 0
    mejor_C, mejor_penalty, mejor_solver = None, None, None
    mejor_conjunto_reglas = []
    reglas_aplicadas = []

    # evaluar hiperparámetros iniciales
    for params in hyperparameters:
        print(
            f"Evaluando hiperparámetros: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}, Iteraciones={params['iteraciones']}")
        inicio_modelo = time.time()

        # asegurarnos de que no hay NaNs
        X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train, X_test)

        # entrenamiento del modelo con la configuracion actual
        model, _, _, _, _ = modelo_reglog(X_train_imputed, X_test_imputed, y_train, y_test,
                                          c=params['c'], pen=params['pen'],
                                          solv=params['solv'], iteraciones=params['iteraciones'])
        y_pred = model.predict(X_test_imputed)
        # calculo de las metricas del modelo
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        media = np.mean([accuracy, precision, recall, f1])
        tiempo_ejecucion = time.time() - inicio_modelo

        print(
            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
        print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

        # guardar resultados
        guardar_resultados("accuracy.csv", accuracy)
        guardar_resultados("precision.csv", precision)
        guardar_resultados("recall.csv", recall)
        guardar_resultados("f1.csv", f1)
        guardar_resultados("media.csv", media)

        # actualizacion del mejor modelo
        if media > mejor_media:
            mejor_media = media
            mejor_C, mejor_penalty, mejor_solver = params['c'], params['pen'], params['solv']
            print(f"Nueva mejor media de métricas: {mejor_media:.4f}")

    # aplicar reglas una a una
    for i in range(len(reglas)):
        mejor_regla = None
        for regla in reglas:
            if regla not in reglas_aplicadas:
                print(f"\nAplicando regla: {regla.__name__}")

                # reglas que requieren tanto X como y
                if regla in reglas_con_y:
                    X_train_modificado, y_train_modificado = regla(X_train.copy(), y_train.copy())
                    X_test_modificado = X_test.copy()
                else:
                    #si solo afecta a x
                    X_train_modificado = regla(X_train.copy())
                    X_test_modificado = regla(X_test.copy())
                    y_train_modificado = y_train

                # imputar valores faltantes despues de aplicar la regla
                X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train_modificado, X_test_modificado)

                # re-evaluar con el mejor conjunto de hiperparametros encontrado
                inicio_modelo = time.time()
                model, _, _, _, _ = modelo_reglog(X_train_imputed, X_test_imputed, y_train_modificado, y_test,
                                                  c=mejor_C, pen=mejor_penalty, solv=mejor_solver,
                                                  iteraciones=5000)
                y_pred = model.predict(X_test_imputed)
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                media = np.mean([accuracy, precision, recall, f1])
                tiempo_ejecucion = time.time() - inicio_modelo

                print(
                    f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                print(f"Tiempo de ejecución del modelo: {tiempo_ejecucion:.2f} segundos")

                guardar_resultados("accuracy.csv", accuracy)
                guardar_resultados("precision.csv", precision)
                guardar_resultados("recall.csv", recall)
                guardar_resultados("f1.csv", f1)
                guardar_resultados("media.csv", media)

                if media > mejor_media:
                    mejor_media = media
                    mejor_regla = regla
                    print(f"Nueva mejor media con regla {regla.__name__}: {mejor_media:.4f}")

        if mejor_regla:
            reglas_aplicadas.append(mejor_regla)
            print(f"Regla {mejor_regla.__name__} añadida al conjunto.")

            # Re-evaluar hiperparámetros después de aplicar la mejor regla
            for params in hyperparameters:
                print(
                    f"\nReevaluando hiperparámetros con regla {mejor_regla.__name__}: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}")
                inicio_modelo = time.time()

                # Imputar valores faltantes nuevamente
                X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train_modificado, X_test_modificado)

                model, _, _, _, _ = modelo_reglog(X_train_imputed, X_test_imputed, y_train_modificado, y_test,
                                                  c=params['c'], pen=params['pen'], solv=params['solv'],
                                                  iteraciones=5000)
                # hace predicciones con el modelo usando el conjunto de test preprocesado
                y_pred = model.predict(X_test_imputed)

                # calcula la exactitud del modelo
                accuracy = accuracy_score(y_test, y_pred)

                # calcula la precision del modelo
                precision = precision_score(y_test, y_pred)

                # calcula el recall del modelo
                recall = recall_score(y_test, y_pred)

                # calcula el F1 score del modelo
                f1 = f1_score(y_test, y_pred)

                # media de las metricas principales (accuracy, precision, recall, F1)
                media = np.mean([accuracy, precision, recall, f1])

                # mide el tiempo de ejecucion del modelo
                tiempo_ejecucion = time.time() - inicio_modelo

                # imprime las metricas del modelo
                print(
                    f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                print(f"Tiempo de ejecucion del modelo: {tiempo_ejecucion:.2f} segundos")

                # guarda las metricas en archivos csv
                guardar_resultados("accuracy.csv", accuracy)
                guardar_resultados("precision.csv", precision)
                guardar_resultados("recall.csv", recall)
                guardar_resultados("f1.csv", f1)
                guardar_resultados("media.csv", media)

                # si la media de las metricas mejora, actualiza el mejor modelo encontrado
                if media > mejor_media:
                    mejor_media = media
                    mejor_regla = regla
                    print(f"Nueva mejor media con regla {regla.__name__}: {mejor_media:.4f}")

                # si se ha encontrado una mejor regla, se añade al conjunto de reglas aplicadas
                if mejor_regla:
                    reglas_aplicadas.append(mejor_regla)
                    print(f"Regla {mejor_regla.__name__} añadida al conjunto.")

                    # reevalua los hiperparametros despues de aplicar la mejor regla
                    for params in hyperparameters:
                        print(
                            f"\nReevaluando hiperparametros con regla {mejor_regla.__name__}: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}")
                        inicio_modelo = time.time()

                        # se imputan los valores faltantes nuevamente
                        X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train_modificado,
                                                                                    X_test_modificado)

                        # se entrena el modelo con los hiperparametros ajustados
                        model, _, _, _, _ = modelo_reglog(X_train_imputed, X_test_imputed, y_train_modificado, y_test,
                                                          c=params['c'], pen=params['pen'], solv=params['solv'],
                                                          iteraciones=5000)

                        # se vuelven a calcular las metricas del modelo
                        y_pred = model.predict(X_test_imputed)
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred)
                        recall = recall_score(y_test, y_pred)
                        f1 = f1_score(y_test, y_pred)
                        media = np.mean([accuracy, precision, recall, f1])
                        tiempo_ejecucion = time.time() - inicio_modelo

                        print(
                            f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}, Media: {media:.4f}")
                        print(f"Tiempo de ejecucion del modelo: {tiempo_ejecucion:.2f} segundos")

                        guardar_resultados("accuracy.csv", accuracy)
                        guardar_resultados("precision.csv", precision)
                        guardar_resultados("recall.csv", recall)
                        guardar_resultados("f1.csv", f1)
                        guardar_resultados("media.csv", media)

                        # si mejora la media de las metricas, se actualizan los mejores hiperparametros
                        if media > mejor_media:
                            mejor_media = media
                            mejor_C, mejor_penalty, mejor_solver = params['c'], params['pen'], params['solv']
                            print(f"Nueva mejor media despues de reevaluar hiperparametros: {mejor_media:.4f}")

                    # refina los hiperparametros numericos para obtener el mejor modelo posible
                    mejor_C, mejor_penalty, mejor_solver = refinar_hiperparametros(X_train_imputed, X_test_imputed,
                                                                                   y_train_modificado, y_test, mejor_C,
                                                                                   mejor_penalty, mejor_solver)

                # si no hay mejora en la iteracion, termina el ciclo
                else:
                    print(f"No se logro una mejora en la iteracion {i + 1}. Deteniendo iteraciones.")
                    break

                # imprime los resultados finales
                print("\n--- Final del proceso ---")
                print(f"Mejor conjunto de reglas: {[regla.__name__ for regla in reglas_aplicadas]}")
                print(f"Mejor conjunto de hiperparametros: C={mejor_C}, Penalty={mejor_penalty}, Solver={mejor_solver}")
                print(f"Mejor media de metricas alcanzada: {mejor_media:.4f}")


# Definir hiperparámetros iniciales
hyperparameters = [
    {'c': 0.01, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.1, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.5, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 1.0, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
]

# Ejecutar el proceso completo
aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters)
