from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

def modelo_is_downloaded(X_train, X_test, y_train, y_test, c, pen, solv, iteraciones):
    model = LogisticRegression(C=c, penalty=pen, solver=solv, max_iter=iteraciones)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)

    return model, precision, recall, f1, accuracy

def aplicar_reglas_y_entrenar_modelos(X_train, X_test, y_train, y_test, hyperparameters):
    results = []
    for params in hyperparameters:
        print(f"\nEvaluando hiperparámetros: C={params['c']}, Penalty={params['pen']}, Solver={params['solv']}, Iteraciones={params['iteraciones']}")
        model, precision, recall, f1, accuracy = modelo_is_downloaded(X_train, X_test, y_train, y_test, params['c'], params['pen'], params['solv'], params['iteraciones'])
        results.append((model, precision, recall, f1, accuracy))
    return results


def ajustar_hiperparametros(X_train, y_train):
    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    }

    grid_search = GridSearchCV(LogisticRegression(max_iter=5000), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    mejor_C = grid_search.best_params_['C']
    mejor_penalty = grid_search.best_params_['penalty']
    mejor_solver = grid_search.best_params_['solver']

    return mejor_C, mejor_penalty, mejor_solver, grid_search.best_score_
