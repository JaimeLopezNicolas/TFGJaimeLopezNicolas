import time
import json
from datetime import datetime
import os
import pandas as pd
from .modelo import modelo_is_downloaded  # Ajuste aquí

def guardar_resultados(resultados, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'a') as json_file:
        json.dump(resultados, json_file, indent=4)
        json_file.write('\n')  # Asegurar que cada resultado esté en una nueva línea

def guardar_resultados_csv(resultados, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df = pd.DataFrame([resultados])
    if not os.path.isfile(filepath):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, mode='a', header=False, index=False)

def RegLogExecutioner(c, pen, sol, iter, datos, printer, mod, variables_corr=None, variables_var=None, reglas_aplicadas=None):
    # Parámetros
    c = c
    pen = pen
    solv = sol
    max_iter = iter
    datos = datos
    mod = mod
    if datos == 1:
        data_path = '../dataset/train/trainCompleto.csv'
        print("Datos completos\n")
    elif datos == 0:
        data_path = '../dataset/train/trainMini.csv'
        print("Prueba pequeña de datos\n")
    else:
        print("Error en la elección de datos\n")
        exit(-1)

    params = {
        'data_path': data_path,
        'c': c,
        'pen': pen,
        'solv': solv,
        'iteraciones': max_iter,
        'variables_corr': variables_corr,  # Pasa las listas acumuladas
        'variables_var': variables_var
    }

    start = time.time()
    if mod == 2:
        (model,
         precision_train, recall_train, f1_train, accuracy_train,
         precision_test, recall_test, f1_test, accuracy_test, variables_corr, variables_var) = modelo_is_downloaded(**params)

    # Captura de la hora y fecha actual
    fecha_hora = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    end = time.time()
    total = end - start
    print("Tiempo total: ", total, " segundos.")

    # Datos a guardar
    data_to_save = {
        "modelo": "Regresion_Logistica_is_clicked" if mod == 1 else "Regresion_Logistica_is_downloaded",
        "data_path": data_path,
        "c": c,
        "penalty": pen,
        "solver": solv,
        "iteraciones": max_iter,
        "tiempo_de_ejecucion": total,
        "fecha_hora": fecha_hora,
        "score": accuracy_test,
        "reglas_aplicadas": reglas_aplicadas
    }

    # Escritura de los datos en un archivo JSON
    if printer == "Y" or printer == "y":
        guardar_resultados(data_to_save, '../resultados/resultado_modelo.json')

    # Escritura de los datos en un archivo CSV en la carpeta Registros
    fecha_hora_formateada = datetime.now().strftime("%Y%m%d_%H%M%S")
    registro_filepath = f'../RegresionLogistica/Registros/resultado_{fecha_hora_formateada}.csv'
    guardar_resultados_csv(data_to_save, registro_filepath)

    return (model,
            precision_train, recall_train, f1_train, accuracy_train,
            precision_test, recall_test, f1_test, accuracy_test,
            variables_corr, variables_var, total)
