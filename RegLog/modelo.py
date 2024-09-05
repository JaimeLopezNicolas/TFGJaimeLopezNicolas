from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# funcion que entrena un modelo de reglog y devuelve las metricas
def modelo_reglog(X_train, X_test, y_train, y_test, c, pen, solv, iteraciones):
    # crear y ajustar reglog con los hiperparam dados
    model = LogisticRegression(C=c, penalty=pen, solver=solv, max_iter=iteraciones)
    model.fit(X_train, y_train)

    # predicciones sobre el conjunto de tets
    y_pred = model.predict(X_test)

    # calculo de metricas de rendimiento
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return model, precision, recall, f1, accuracy

# funcion que aplica las reglas a los datos y entrena el modelo
def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters):
    results = []
    for params in hyperparameters:
        # entrena el modelo para cada conjunto de hiperpraametros
        print(f"\nEvaluando hiperparámetros: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}, Iteraciones={params['iteraciones']}")
        model, precision, recall, f1, accuracy = modelo_reglog(X_train, X_test, y_train, y_test, params['c'], params['pen'], params['solv'], params['iteraciones'])

        # guardar los resultados
        results.append((model, precision, recall, f1, accuracy))
    return results


# funcion que ajusta los hiperparametros con una busqueda en cuadricula
def ajustar_hiperparametros(X_train, y_train):
    # definir cuadricula de busqueda para los hiperparametros
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    # busqueda de los mejores hiperparametros con validacion cruzada
    grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # mejores valores de C, penalty y solver
    mejor_C = grid_search.best_params_['C']
    mejor_penalty = grid_search.best_params_['penalty']
    mejor_solver = grid_search.best_params_['solver']

    return mejor_C, mejor_penalty, mejor_solver, grid_search.best_score_
