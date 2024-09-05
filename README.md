# TFG_Entrega_Codigo

# Configuración y Ejecución
1. Preparacion de los datos:
    1. Colocar el archivo CSV dentro de la carpeta dataset.
    2. Observar el número de columnas, columnas no interesantes y nombre de variable objetivo.
    ADVERTENCIA: LA VARIABLE OBJETIVO DEBE SER BINARIA DE VALORES 0 O 1.

2. Entrenamiento y evaluación:
    1. Abrir el archivo de Experto a ejecutar y configurar:
           - *data_path*: ruta al archivo CSV con los datos.
           - *num_atributos*: número de atributos (columnas) en los datos que se van a usar como predictores menos uno (si son 9 poner 8, debido a la objetivo).
           - *nombre_variable_objetivo*: nombre de la columna que contiene la variable objetivo.
           - *ruta_resultados*: ruta donde se guardarán los resultados del entrenamiento y evaluación.
    2. El sistema genera de forma automática la combinaciones de reglas y configuraciones de hiperparámetros. Estas combinaciones se evaluan de una en una, midiendo las siguientes métricas para cada combinación:
           -Accuracy
           -Precision
           -Recall
           -F1 Score
           -Tiempo de ejecución por modelo
    3. Los resultados se guardan en la carpeta especificada bajo *ruta_resultados*.

3. Los CSVs de resultados serán los siguientes:
    1. accuracy.csv: contendrá los valores de la Accuracy en cada iteración
    2. f1.csv: contendrá los valores del F1 Score en cada iteración
    3. media.csv: contendrá los valores de la Media en cada iteración
    4. precision.csv: contendrá los valores de la Precisión en cada iteración
    5. recall.csv: contendrá los valores del Recall en cada iteración


# Contribuciones
Si deseas contribuir a este proyecto, por favor realiza un fork del repositorio y envía un pull request con tus cambios.

# Autor
Este proyecto ha sido desarrollado por Jaime López-Nicolás Algueró.
