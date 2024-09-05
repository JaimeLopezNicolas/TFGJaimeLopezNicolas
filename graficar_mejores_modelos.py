import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Función para mostrar el podio
def mostrar_podio(csv_path, metrica, num_modelos=5):
    # Leer el archivo CSV
    data = pd.read_csv(csv_path)

    # Verificar si existen suficientes modelos en los datos
    if len(data) < num_modelos:
        print(f"Advertencia: Solo se encontraron {len(data)} modelos en el CSV.")
        num_modelos = len(data)

    # Ordenar los modelos por la métrica seleccionada
    data_ordenada = data.sort_values(by=metrica, ascending=False).head(num_modelos)

    # Crear etiquetas con hiperparámetros y reglas específicas para el modelo de regresión logística
    data_ordenada['Hiperparámetros'] = data_ordenada.apply(
        lambda row: f"Reglas: {row['Numero de reglas']}\nC: {row['C']}\nSolver: {row['Solver']}", axis=1)

    # Configurar el tamaño del gráfico
    plt.figure(figsize=(10, 6))

    # Generar el gráfico de barras estilo podio sin especificar el hue
    sns.barplot(x='Hiperparámetros', y=metrica, data=data_ordenada, palette='coolwarm')

    # Añadir los valores de la métrica en las barras
    for index, row in data_ordenada.iterrows():
        plt.text(index, row[metrica] + 0.001, round(row[metrica], 4), color='black', ha="center")

    # Títulos y etiquetas
    plt.title(f'Top 3 Modelos según {metrica}', fontsize=14)
    plt.ylabel(metrica)
    plt.xlabel('Hiperparámetros y Reglas')
    plt.xticks(rotation=45, ha='right')

    # Ajustar el gráfico
    plt.subplots_adjust(bottom=0.25)
    plt.show()


# Ruta al archivo CSV de resultados
csv_path = 'resultados/DiabetesRegLogGraficos/resultados_detallados.csv'

# Especifica la métrica que quieres visualizar
metrica = 'F1 Score'  # Cambia esto por 'Precision', 'Recall', 'F1 Score', etc.

# Número de modelos a mostrar en el podio
num_modelos = 3  # Cambia esto si quieres mostrar más o menos modelos

# Llamada a la función
mostrar_podio(csv_path, metrica, num_modelos)
