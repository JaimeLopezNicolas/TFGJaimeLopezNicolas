import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def graficar_lineas_suavizadas(csv_path, x_col, y_col, window_size=10):
    # Leer el archivo CSV
    data = pd.read_csv(csv_path)

    # Verificar si la columna seleccionada existe en el CSV
    if y_col not in data.columns:
        print(f"La columna seleccionada '{y_col}' no existe en el archivo CSV.")
        return

    # Crear una columna 'Orden' para representar el número de modelo (iteración)
    data['Orden'] = range(1, len(data) + 1)

    # Obtener los números únicos de reglas para asignar un color diferente a cada conjunto
    unique_rules = data['Numero de reglas'].unique()
    colors = cm.rainbow([i / len(unique_rules) for i in range(len(unique_rules))])

    # Crear gráfico de líneas
    plt.figure(figsize=(10, 6))

    for i, rules in enumerate(unique_rules):
        subset = data[data['Numero de reglas'] == rules].copy()  # Usar .copy() para evitar la advertencia
        # Calcular la media móvil para suavizar los datos
        subset['Smoothed'] = subset[y_col].rolling(window=window_size).mean()
        plt.plot(subset['Orden'], subset['Smoothed'], marker='o', linestyle='-', color=colors[i], label=f'{rules} reglas')

    plt.title(f'{y_col} vs Orden de los Modelos (suavizado)')
    plt.xlabel('Orden de los Modelos')
    plt.ylabel(y_col)
    plt.grid(True)
    plt.legend(title='Número de Reglas')
    plt.show()


# Ejemplo de uso
csv_path = 'resultados/DiabetesRegLogGraficos/resultados_detallados.csv'
x_col = 'Orden'  # columna x del eje
y_col = 'Tiempo'  # columna y del eje

graficar_lineas_suavizadas(csv_path, x_col, y_col, window_size=10)
