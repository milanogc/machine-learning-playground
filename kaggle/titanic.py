# score of 0.74641 without Embarked and using RandomForestClassifier

import os
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline

from sklearn_pandas import DataFrameMapper
from sklearn_pandas import CategoricalImputer

train_set = pd.read_csv(os.path.join("kaggle", "titanic", "train.csv"), index_col = "PassengerId")
test_set = pd.read_csv(os.path.join("kaggle", "titanic", "test.csv"), index_col = "PassengerId")

mapper = DataFrameMapper([
    ("Age", None),
    ("Fare", None),
    #("Embarked", [CategoricalImputer(), LabelBinarizer()]), # at first it didn't well
    ("Sex", LabelEncoder()),
    ("Pclass", None),
    ("SibSp", None),
], df_out = False)

pipeline = Pipeline([
    ('mapper', mapper),
    ('imputer', Imputer()),
    ('scaler', StandardScaler()),
    #('classifier', SGDClassifier(random_state = 42, n_jobs = 4)), # Stochastic Gradient Descent
    ('classifier', RandomForestClassifier(random_state = 42, n_jobs = 4))
])

train_set_labels = train_set["Survived"]
pipeline.fit(train_set, train_set_labels)
test_set["Survived"] = pipeline.predict(test_set)
test_set[["Survived"]].to_csv(os.path.join("kaggle", "titanic", "result.csv"))

