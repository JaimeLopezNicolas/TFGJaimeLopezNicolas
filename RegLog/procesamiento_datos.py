import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# funcion para corregir CSVs sin comas
def detectar_y_cambiar_delimitador(data_path):
    try:
        # comprobacion del delimitador con comas
        data = pd.read_csv(data_path)
        print("Archivo cargado correctamente con delimitador de comas.")
        return data
    except pd.errors.ParserError:
        # prueba con delimitador con punto y coma si falla
        print("Error de delimitación detectado, intentando con delimitador de punto y coma...")
        data = pd.read_csv(data_path, delimiter=';')
        print("Archivo cargado correctamente con delimitador de punto y coma.")
        return data


# funcion que selecciona y preprocesa las caracteristicas del dataset
def seleccionar_y_preprocesar_caracteristicas(data, num_atributos, nombre_variable_objetivo, columnas_a_ignorar):
    # elegir las caracteristicas y la variable objetivo
    features = [col for col in data.columns if col != nombre_variable_objetivo and col not in columnas_a_ignorar][
               :num_atributos]
    y = data[nombre_variable_objetivo]
    X = data[features]

    # distinguir entre variables numericas y categoricas
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # crear transformadores por tipo de datos
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # Asegurar salida densa

    # aplicar las transformaciones definidas a las caracteristicas
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # aplicar transformaciones a los datos
    X_processed = preprocessor.fit_transform(X)

    # imprimir la forma del dataset procesado para depuracion
    print("Forma de X_processed:", X_processed.shape)

    # convertir los datos procesados a dataframe si es necesario
    if isinstance(X_processed, pd.DataFrame):
        X_processed_df = X_processed
    else:
        # obtener los nombres de las caracteristicas solo si hay caracteristicas categoricas
        if categorical_features:
            cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
                categorical_features)
        else:
            cat_feature_names = []

        all_feature_names = numeric_features + list(cat_feature_names)
        print("Número de características numéricas:", len(numeric_features))
        print("Número de características categóricas (one-hot):", len(cat_feature_names))
        print("Número total de características:", len(all_feature_names))

        # convertir a df
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # dividir en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

    # verificar num de columnas generadas
    print(f"Número de columnas generadas: {X_processed_df.shape[1]}")
    print(f"Número de nombres de columnas: {len(X_processed_df.columns)}")

    return X_train, X_test, y_train, y_test, X_processed_df.columns.tolist(), preprocessor


# funcion que guarda los datos procesados en un CSV
def guardar_datos_procesados(X_train, X_test, y_train, y_test, ruta_salida, column_names):
    # asegura que el numero de columnas coincide
    if X_train.shape[1] != len(column_names):
        raise ValueError(
            f"El número de nombres de columnas ({len(column_names)}) no coincide con el número de características ({X_train.shape[1]}).")

    # convertir entrenamiento y prueba a dataframe
    train_data = pd.DataFrame(X_train, columns=column_names)
    test_data = pd.DataFrame(X_test, columns=column_names)
    train_data['target'] = y_train.reset_index(drop=True)
    test_data['target'] = y_test.reset_index(drop=True)

    # guardar en CSV
    train_data.to_csv(f'{ruta_salida}_train.csv', index=False)
    test_data.to_csv(f'{ruta_salida}_test.csv', index=False)
    print("Datos procesados guardados correctamente.")


# funcion que realiza el procesamiento completo
def procesamiento_integral(data_path, num_atributos, nombre_variable_objetivo, columnas_a_ignorar, ruta_salida):
    # detectar delimitador correcto y cargar los datos
    data = detectar_y_cambiar_delimitador(data_path)
    # seleccionar y preprocesar caracteristicas
    X_train, X_test, y_train, y_test, features, preprocessor = seleccionar_y_preprocesar_caracteristicas(data,
                                                                                                         num_atributos,
                                                                                                         nombre_variable_objetivo,
                                                                                                         columnas_a_ignorar)
    # guardar datos procesados
    guardar_datos_procesados(X_train, X_test, y_train, y_test, ruta_salida, features)
    return X_train, X_test, y_train, y_test, preprocessor, features
