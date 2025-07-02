from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV


def split_data_into_train_test_validation(X, y):
    # First split: separate test set (20% of data)
    X_train, X_test, y_train, y_test= train_test_split(X, y,
                                                    train_size=0.8,
                                                    stratify=y,
                                                    shuffle=True,
                                                    random_state=42)

    print(f"Train set: {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%} of data)")
    print(f"Test set: {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%} of data)")

    return X_train, X_test, y_train, y_test
    


def find_best_model(X_train, y_train, param_grid, model):
    grid = GridSearchCV(model, param_grid, cv=5)
    grid.fit(X_train, y_train)
    print(f"Best parameters: {grid.best_params_}")
    print(f"Best cross-validation score: {grid.best_score_:.4f}")
    return grid.best_estimator_, grid.best_params_


def find_best_knn_model(X_train, y_train):
    param_grid = {
    'n_neighbors': [1, 2, 3, 4, 5],
    'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']
    }
    model = KNeighborsClassifier()
    return find_best_model(X_train, y_train, param_grid, model)


def find_best_svm_model(X_train, y_train):
    param_grid = {'C': [10e0, 10e1],
			'gamma': ['scale'],
			'kernel': ['rbf','linear']}
    model = SVC()
    return find_best_model(X_train, y_train, param_grid, model)


def find_best_svm_model2(X_train, y_train):
    param_grid = {'C': [10e1],
			'gamma': ['scale'],
			'kernel': ['rbf']}
    model = SVC()
    return find_best_model(X_train, y_train, param_grid, model)

def evaluate_model(model, X_test, y_test):
    y_test_pred = model.predict(X_test)
    # classification report
    print()
    print(f"Classification report on test set:\n")
    print(classification_report(y_test, y_test_pred, target_names=['not_blink', 'blink']))