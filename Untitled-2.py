# %%
import pandas as pd
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df = pd.read_csv("titanic_dataset.csv")

# %%
df.info()


# %%
df = df.drop(columns=["Cabin"])
df.info()

# %%
# Histogram for each feature
df.hist(figsize=(12, 8))
plt.tight_layout()
plt.show()


# %%
df_num = df.select_dtypes(include="number").drop(columns="PassengerId")
corr_matrix = df_num.corr()
plt.figure(figsize=(12,8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Feature Correlation Matrix')
plt.show()


# %%
x = df.drop(columns=["PassengerId","Survived"])
y = df["Survived"]

# %% [markdown]
# i guess data imbalance is not that bass

# %%
print(df['Survived'].value_counts())

# %% [markdown]
# chec if stratify is neccary

# %%
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42, stratify= y)

# %%
num_cols = df_num.columns
num_cols = num_cols.to_list()
cat_cols = [c for c in df.columns if c not in num_cols]
print(cat_cols, "and", num_cols)

# %% [markdown]
# # Model 

# %%
from sklearn.preprocessing import  OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='most_frequent')

# %%
pipe_pre = Pipeline([
    ("impute", imputer),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# %%
ct = ColumnTransformer([
    ("cat", pipe_pre , cat_cols)
])

# %%
from sklearn.ensemble import RandomForestClassifier
model_rf = RandomForestClassifier()

# %%
pipe_main = Pipeline([
    ("pre", pipe_pre),
    ("clf", model_rf)
])

# %%
pipe_main.fit(x_train, y_train)

# %%
y_preb_base = pipe_main.predict(x_test)
from sklearn.metrics import accuracy_score

# %%
accuracy_score(y_test, y_preb_base)


# %%
from sklearn.metrics import average_precision_score , precision_recall_curve , f1_score

# %%
print("base F1:", f1_score(y_test, y_preb_base))

# %%
from sklearn.metrics import average_precision_score , precision_recall_curve , f1_score
proba_test = pipe_main.predict_proba(x_test)[:, 1]
ap = average_precision_score(y_test, proba_test)
print(f"Average Precision (PR-AUC): {ap:.3f} | Baseline ~ {y.mean():.3f}")


# %%
import numpy as np
prec, rec, thr = precision_recall_curve(y_test, proba_test)
f1s = (2*prec*rec)/(prec+rec+1e-12)
best_idx = np.nanargmax(f1s)
best_thr = thr[best_idx] if best_idx < len(thr) else 0.5
print(f"Best F1={f1s[best_idx]:.3f} at threshold={best_thr:.3f}")

# %%
# --- 6) تصمیم‌گیری با آستانه‌ی انتخاب‌شده
y_pred = (proba_test >= best_thr).astype(int)
print("Final F1:", f1_score(y_test, y_pred))

# %%
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)


