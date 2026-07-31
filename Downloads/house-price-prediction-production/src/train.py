# src/train.py
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import (
    KFold,
    cross_validate,
    GridSearchCV
)

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.ensemble import (
    RandomForestRegressor,
    HistGradientBoostingRegressor
)

from sklearn.metrics import (
    mean_absolute_error,
    root_mean_squared_error,
    r2_score
)

from src.config import (
    CSV_PATH,
    RANDOM_STATE
)

from src.preprocess import (
    load_data,
    split_data,
    create_preprocessor
)


def train_model():

    # ==========================
    # Load Dataset
    # ==========================

    df = load_data(CSV_PATH)

    X_train, X_test, y_train, y_test = split_data(df)

    preprocess = create_preprocessor(X_train)

    # ==========================
    # Baseline Model
    # ==========================

    baseline_pipe = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", LinearRegression())
        ]
    )

    baseline_pipe.fit(X_train, y_train)

    train_pred = baseline_pipe.predict(X_train)
    test_pred = baseline_pipe.predict(X_test)

    print("\n========== Baseline ==========")

    print("Train RMSE:",
          root_mean_squared_error(y_train, train_pred))

    print("Test RMSE:",
          root_mean_squared_error(y_test, test_pred))

    print("Train R2:",
          r2_score(y_train, train_pred))

    print("Test R2:",
          r2_score(y_test, test_pred))

    # ==========================
    # Model Comparison
    # ==========================

    models = {

        "LinearRegression": LinearRegression(),

        "Ridge":
        Ridge(random_state=RANDOM_STATE),

        "Lasso":
        Lasso(
            random_state=RANDOM_STATE,
            max_iter=10000
        ),

        "RandomForest":
        RandomForestRegressor(
            random_state=RANDOM_STATE
        ),

        "HistGB":
        HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        )
    }

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scoring = {

        "rmse":
        "neg_root_mean_squared_error",

        "mae":
        "neg_mean_absolute_error",

        "r2":
        "r2"
    }

    rows = []

    for name, model in models.items():

        pipe = Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("model", model)
            ]
        )

        scores = cross_validate(
            pipe,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1
        )

        rows.append({

            "Model": name,

            "CV_RMSE":
            -scores["test_rmse"].mean(),

            "CV_MAE":
            -scores["test_mae"].mean(),

            "CV_R2":
            scores["test_r2"].mean()
        })

    results = pd.DataFrame(rows)

    print("\n========== Model Comparison ==========")

    print(results.sort_values("CV_RMSE"))

    # ==========================
    # GridSearchCV
    # ==========================

    hgb_pipe = Pipeline(

        steps=[
            ("preprocess", preprocess),

            (
                "model",
                HistGradientBoostingRegressor(
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    param_grid = {

        "model__learning_rate":
        [0.03, 0.05, 0.1],

        "model__max_depth":
        [None, 3, 6],

        "model__max_leaf_nodes":
        [15, 31, 63],

        "model__min_samples_leaf":
        [20, 50, 100],

        "model__l2_regularization":
        [0.0, 0.1, 1.0]
    }

    grid = GridSearchCV(

        estimator=hgb_pipe,

        param_grid=param_grid,

        cv=cv,

        scoring="neg_root_mean_squared_error",

        n_jobs=1,

        verbose=2
    )

    grid.fit(X_train, y_train)

    print("\n========== Best Parameters ==========")

    print(grid.best_params_)

    print("Best RMSE:", -grid.best_score_)

    # ==========================
    # Final Model
    # ==========================

    best_model = grid.best_estimator_

    best_model.fit(X_train, y_train)

    # ==========================
    # Final Evaluation
    # ==========================

    prediction = best_model.predict(X_test)

    print("\n========== Final Evaluation ==========")

    print(
        "RMSE:",
        root_mean_squared_error(
            y_test,
            prediction
        )
    )

    print(
        "MAE:",
        mean_absolute_error(
            y_test,
            prediction
        )
    )

    print(
        "R2:",
        r2_score(
            y_test,
            prediction
        )
    )

    # ==========================
    # Save Model
    # ==========================

    joblib.dump(
        best_model,
        "model/model.pkl"
    )

    print("\nModel Saved Successfully!")


if __name__ == "__main__":

    train_model()