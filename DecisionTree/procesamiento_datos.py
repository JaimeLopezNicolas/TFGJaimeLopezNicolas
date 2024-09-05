import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# funcion que detecta y ajusta el delimitador en el dataset
def detectar_y_cambiar_delimitador(data_path):
    try:
        data = pd.read_csv(data_path)
        print("Archivo cargado correctamente con delimitador de comas.")
        return data
    except pd.errors.ParserError:
        # si hay error, intenta con otro delimitador
        print("Error de delimitación detectado, intentando con delimitador de punto y coma...")
        data = pd.read_csv(data_path, delimiter=';')
        print("Archivo cargado correctamente con delimitador de punto y coma.")
        return data


# selecciona y preprocesa las caracteristicas de los datos
def seleccionar_y_preprocesar_caracteristicas(data, num_atributos, nombre_variable_objetivo, columnas_a_ignorar):
    # selecciona las columnas de features
    features = [col for col in data.columns if col != nombre_variable_objetivo and col not in columnas_a_ignorar][
               :num_atributos]
    y = data[nombre_variable_objetivo]
    X = data[features]

    # separa features numericos y categoricos
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    # preprocesa columnas numericas y categoricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])  # Asegurar salida densa

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # aplica las transformaciones a los datos
    X_processed = preprocessor.fit_transform(X)

    # si los datos no son dataframe, convierte a dataframe
    print("Forma de X_processed:", X_processed.shape)
    if isinstance(X_processed, pd.DataFrame):
        X_processed_df = X_processed
    else:
        cat_feature_names = preprocessor.named_transformers_['cat']['onehot'].get_feature_names_out(
            categorical_features)
        all_feature_names = numeric_features + list(cat_feature_names)
        print("Número de características numéricas:", len(numeric_features))
        print("Número de características categóricas (one-hot):", len(cat_feature_names))
        print("Número total de características:", len(all_feature_names))
        X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)

    # divide los datos en entrenamiento y test
    X_train, X_test, y_train, y_test = train_test_split(X_processed_df, y, test_size=0.2, random_state=42)

    # Verificar la cantidad de columnas generadas y la cantidad de nombres
    print(f"Número de columnas generadas: {X_processed_df.shape[1]}")
    print(f"Número de nombres de columnas: {len(X_processed_df.columns)}")

    return X_train, X_test, y_train, y_test, X_processed_df.columns.tolist(), preprocessor


# guarda los datos preprocesados en CSV
def guardar_datos_procesados(X_train, X_test, y_train, y_test, ruta_salida, column_names):
    # Convertir matrices a DataFrames
    if X_train.shape[1] != len(column_names):
        raise ValueError(
            f"El número de nombres de columnas ({len(column_names)}) no coincide con el número de características ({X_train.shape[1]}).")

    train_data = pd.DataFrame(X_train, columns=column_names)
    test_data = pd.DataFrame(X_test, columns=column_names)
    train_data['target'] = y_train.reset_index(drop=True)
    test_data['target'] = y_test.reset_index(drop=True)

    # Guardar en CSV
    train_data.to_csv(f'{ruta_salida}_train.csv', index=False)
    test_data.to_csv(f'{ruta_salida}_test.csv', index=False)
    print("Datos procesados guardados correctamente.")


# funcion principal para procesamiento de datos
def procesamiento_integral(data_path, num_atributos, nombre_variable_objetivo, columnas_a_ignorar, ruta_salida):
    data = detectar_y_cambiar_delimitador(data_path)
    X_train, X_test, y_train, y_test, features, preprocessor = seleccionar_y_preprocesar_caracteristicas(data,
                                                                                                         num_atributos,
                                                                                                         nombre_variable_objetivo,
                                                                                                         columnas_a_ignorar)
    guardar_datos_procesados(X_train, X_test, y_train, y_test, ruta_salida, features)
    return X_train, X_test, y_train, y_test, preprocessor, features
