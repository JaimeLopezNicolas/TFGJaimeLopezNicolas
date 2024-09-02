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
data_path = "dataset/train/diabetes_prediction_dataset.csv"
num_atributos = 8
nombre_variable_objetivo = 'diabetes'
columnas_a_ignorar = []
ruta_salida = "dataset/processed/data"
ruta_resultados = "resultados/pruebaDiabetesRegLog/"

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
    #regla_eliminacion_correlacion_alta,
    regla_balanceo_clases,  # Esta regla requiere tanto X como y
]

# Reglas que necesitan tanto X como y
reglas_con_y = [
    regla_balanceo_clases
]


def guardar_resultados(nombre_archivo, valor):
    ruta_completa = os.path.join(ruta_resultados, nombre_archivo)
    with open(ruta_completa, "a") as archivo:
        archivo.write(f"{valor}\n")


def imputar_valores_faltantes(X_train, X_test):
    imputer = SimpleImputer(strategy='mean')
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed


def refinar_hiperparametros(X_train, X_test, y_train, y_test, mejor_C, mejor_penalty, mejor_solver):
    steps = [0.005, 0.001]
    mejor_media = 0
    mejor_configuracion = (mejor_C, mejor_penalty, mejor_solver)

    for step in steps:
        for c in [mejor_C - step, mejor_C, mejor_C + step]:
            for iteraciones in [5000 - 1000, 5000, 5000 + 1000]:
                print(f"\nRefinando: C={c}, Iteraciones={iteraciones}")

                inicio_modelo = time.time()
                model, _, _, _, _ = modelo_reglog(X_train, X_test, y_train, y_test,
                                                  c=c, pen=mejor_penalty, solv=mejor_solver,
                                                  iteraciones=iteraciones)
                y_pred = model.predict(X_test)
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
                    mejor_configuracion = (c, mejor_penalty, mejor_solver)
                    print(f"Nueva mejor media refinada: {mejor_media:.4f}")

    return mejor_configuracion


def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters):
    mejor_media = 0
    mejor_C, mejor_penalty, mejor_solver = None, None, None
    mejor_conjunto_reglas = []
    reglas_aplicadas = []

    # Evaluar hiperparámetros iniciales
    for params in hyperparameters:
        print(
            f"Evaluando hiperparámetros: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}, Iteraciones={params['iteraciones']}")
        inicio_modelo = time.time()

        # Asegurarnos de que no hay NaNs
        X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train, X_test)

        model, _, _, _, _ = modelo_reglog(X_train_imputed, X_test_imputed, y_train, y_test,
                                          c=params['c'], pen=params['pen'],
                                          solv=params['solv'], iteraciones=params['iteraciones'])
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
            mejor_C, mejor_penalty, mejor_solver = params['c'], params['pen'], params['solv']
            print(f"Nueva mejor media de métricas: {mejor_media:.4f}")

    # Aplicar reglas de una en una
    for i in range(len(reglas)):
        mejor_regla = None
        for regla in reglas:
            if regla not in reglas_aplicadas:
                print(f"\nAplicando regla: {regla.__name__}")

                if regla in reglas_con_y:
                    X_train_modificado, y_train_modificado = regla(X_train.copy(), y_train.copy())
                    X_test_modificado = X_test.copy()
                else:
                    X_train_modificado = regla(X_train.copy())
                    X_test_modificado = regla(X_test.copy())
                    y_train_modificado = y_train

                # Imputar valores faltantes después de aplicar la regla
                X_train_imputed, X_test_imputed = imputar_valores_faltantes(X_train_modificado, X_test_modificado)

                # Re-evaluar con el mejor conjunto de hiperparámetros encontrado
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
                    mejor_C, mejor_penalty, mejor_solver = params['c'], params['pen'], params['solv']
                    print(f"Nueva mejor media de métricas después de reevaluar hiperparámetros: {mejor_media:.4f}")

            # Refinar hiperparámetros numéricos para obtener el mejor modelo posible
            mejor_C, mejor_penalty, mejor_solver = refinar_hiperparametros(
                X_train_imputed, X_test_imputed, y_train_modificado, y_test, mejor_C, mejor_penalty, mejor_solver)

        else:
            print(f"No se logró una mejora en la iteración {i + 1}. Deteniendo iteraciones.")
            break

    print("\n--- Final del proceso ---")
    print(f"Mejor conjunto de reglas: {[regla.__name__ for regla in reglas_aplicadas]}")
    print(f"Mejor conjunto de hiperparámetros: C={mejor_C}, Penalty={mejor_penalty}, Solver={mejor_solver}")
    print(f"Mejor media de métricas alcanzada: {mejor_media:.4f}")


# Definir hiperparámetros iniciales
hyperparameters = [
    {'c': 0.01, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.1, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 0.5, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
    {'c': 1.0, 'pen': 'l2', 'solv': 'saga', 'iteraciones': 5000},
]

# Ejecutar el proceso completo
aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters)
