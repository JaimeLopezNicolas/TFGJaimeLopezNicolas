from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import cross_val_score

# funcion para entrenar el modelo de arbol de decision
def modelo_arbol_decision(X_train, X_test, y_train, y_test, max_depth, min_samples_split):
    print("Comenzando modelo de Árbol de Decisión")
    print("Los hiperparámetros del modelo son: \n",
          "Max Depth: ", max_depth, "\n",
          "Min Samples Split: ", min_samples_split, "\n")

    # inicializa y entrena el modelo
    model = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)

    # hace predicciones en el conjunto de prueba
    y_pred = model.predict(X_test)

    # calcula las metricas
    precision_test = precision_score(y_test, y_pred)
    recall_test = recall_score(y_test, y_pred)
    f1_test = f1_score(y_test, y_pred)
    accuracy_test = accuracy_score(y_test, y_pred)

    # simula tiempo de ejecucion
    tiempo_total = "Tiempo de ejecución: 1 minuto"

    print("Modelo entrenado y evaluado\n")
    return (model, precision_test, recall_test, f1_test, accuracy_test, tiempo_total)
