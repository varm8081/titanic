# %% [markdown]
# #  Libraries

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report

# %% [markdown]
# # Settings

# %%

config = {
    "target_column": "Profit_Class",
    "drop_columns" : ["Profit_Class", "CompanyYear", "Net Profit"],
    "file_path": "my_version.csv",
    "test_size": 0.2,
    "random_state": 42,
}
# every thing is changble
param_grid = {
            "classifier__random_state": [10, 20, 40],
            "classifier__min_samples_split": [2, 4, 5],
            "classifier__max_depth": [3, 5, 10]
            }
df = pd.read_csv(config["file_path"])


# %% [markdown]
# # Functions

# %%
def preapare_data(config):
        df[config["target_column"]] = df["Net Profit"].apply(
        lambda x: "High" if x > df["Net Profit"].median() else "Low"
    )
        x = df.drop(columns=config["drop_columns"])
        y = df[config["target_column"]]
        return train_test_split(x, y, test_size=config["test_size"], random_state=config["random_state"])


# %%
def build_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
        ("classifier", DecisionTreeClassifier())
    ])

# %%
def tune_model(pipeline, param_grid, X_train, y_train):
    grid = GridSearchCV(pipeline, param_grid, cv=5, scoring="f1_weighted")
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


# %%
def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    print("F1 Score:", f1_score(y_test, preds, average="weighted"))
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

# %% [markdown]
# # 🚀 Main Execution

# %%
if __name__ == "__main__":
    # Load data
    X_train, X_test, y_train, y_test = preapare_data(config)

    # Build pipeline
    pipeline = build_pipeline()

    # Tune model
    best_model, best_params, best_cv_score = tune_model(pipeline, param_grid, X_train, y_train)
    print("Best Params:", best_params)
    print("Best CV F1:", best_cv_score)

    # Evaluate
    evaluate_model(best_model, X_test, y_test)

# %%



