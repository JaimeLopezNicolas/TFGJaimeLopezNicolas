import pandas as pd

# Parámetros
ruta_csv = "dataset/train/weatherAUS.csv"  # Reemplaza con la ruta de tu archivo CSV
columna_objetivo = "RainTomorrow"  # Reemplaza con el nombre de la columna que deseas modificar
valor_a_convertir_en_0 = "No"  # Reemplaza con el valor que deseas convertir en 0
valor_a_convertir_en_1 = "Yes"  # Reemplaza con el valor que deseas convertir en 1
ruta_salida = "dataset/train/weather_AUS_conv.csv"  # Ruta donde se guardará el CSV modificado

# Leer el CSV
df = pd.read_csv(ruta_csv)

# Verificar que la columna objetivo existe en el dataframe
if columna_objetivo not in df.columns:
    raise ValueError(f"La columna '{columna_objetivo}' no existe en el archivo CSV.")

# Mapear los valores a 0 y 1
df[columna_objetivo] = df[columna_objetivo].map({
    valor_a_convertir_en_0: 0,
    valor_a_convertir_en_1: 1
})

# Verificar si hay valores que no han sido mapeados
valores_sin_mapear = df[columna_objetivo].isna().sum()
if valores_sin_mapear > 0:
    print(f"Advertencia: {valores_sin_mapear} valores no fueron mapeados porque no coincidían con los valores dados.")
    # Aquí puedes decidir qué hacer con esos valores: eliminarlos, convertirlos en NaN, etc.
    # df = df.dropna(subset=[columna_objetivo])

# Guardar el CSV modificado
df.to_csv(ruta_salida, index=False)

print(f"El archivo modificado se ha guardado en: {ruta_salida}")
