import os
import pandas as pd
import matplotlib.pyplot as plt

# Configuración
carpetas_resultados = ["pruebaNBA_Arboles"]  # Añade aquí las carpetas que contienen los CSVs
nombre_salida = "NBAArboles"  # El nombre que quieres para los gráficos
colores = {
    "media": "darkblue",
    "accuracy": "darkred",
    "f1": "darkorange",
    "precision": "darkgreen",
    "recall": "darkviolet"
}
csv_files = ["accuracy.csv", "f1.csv", "media.csv", "precision.csv", "recall.csv"]

# Crear la carpeta de salida si no existe
if not os.path.exists(nombre_salida):
    os.makedirs(nombre_salida)


# Función para crear gráficos
def crear_grafico(datos, metrica, nombre_archivo):
    plt.figure(figsize=(10, 6))
    for carpeta in datos:
        plt.plot(datos[carpeta], label=carpeta)

    plt.title(f"Progress of {metrica.capitalize()}")
    plt.xlabel("Iterations")
    plt.ylabel(metrica.capitalize())
    plt.ylim(0, 1)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.gca().set_facecolor('lightgrey')
    plt.legend()
    plt.savefig(f"{nombre_salida}/{nombre_archivo}.png")
    plt.close()


# Leer y graficar datos
for file in csv_files:
    metrica = file.split('.')[0]
    datos_metricas = {}

    for carpeta in carpetas_resultados:
        ruta_csv = os.path.join(carpeta, file)
        if os.path.exists(ruta_csv):
            datos = pd.read_csv(ruta_csv, header=None).squeeze()
            datos_metricas[carpeta] = datos

    # Crear gráfico para la métrica
    crear_grafico(datos_metricas, metrica, f"{nombre_salida}_{metrica}")

print("Gráficos creados y guardados correctamente.")
