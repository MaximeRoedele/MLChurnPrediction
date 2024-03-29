from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline


# Define a BaseEstimator to drop redundant datacolumns
class ColumnDropper(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        return X.drop(["customerID"], axis=1)


# Define a class to amputate missing data
class NaNAmputator(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        X.replace(["", " "], np.nan, inplace=True)  # replace empty string with nan
        return X.dropna()  # remove all nan values


# Define a class to handle feature encoding: Text data -> Numerical data
class FeatureEncoder(BaseEstimator, TransformerMixin):

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        # encode the values in `gender`: male -> 0, female -> 1
        gender_dct = {"Male": 0.0, "Female": 1.0}
        X["gender"] = [gender_dct[g] for g in X["gender"]]

        # One Hot encode the following columns
        columns = {
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaymentMethod": [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
        }
        for col, cat in columns.items():
            # Define an encoder instance for the col returning a pandas DF
            ohe = OneHotEncoder(
                categories=[cat],
                sparse_output=False,
            ).set_output(transform="pandas")

            # Fit the encoder to the correct column of X
            ohe_transform = ohe.fit_transform(X[[col]])

            # concat X with the transformed DF and remove column 'col'
            X = pd.concat([X, ohe_transform], axis=1).drop(columns=[col])

        # Ordinally encode the following columns with the specified rankings
        columns = {"InternetService": ["No", "DSL", "Fiber optic"]}
        for col, cat in columns.items():
            # Define an ordinal encoder instance for the col
            enc = OrdinalEncoder(categories=[cat])

            # Fit the encoder to the data in X[col] and replace the original
            X[col] = enc.fit_transform(X[[col]])

        # Encode the values of columns containing yes/no -> 1/0
        for col in X.columns:
            X[col] = X[col].apply(
                lambda x: 0.0 if type(x) is str and "No" in x else x
            )  # change all substrings with 'No' to 0
            X[col] = X[col].apply(
                lambda x: 1.0 if type(x) is str and "Yes" in x else x
            )  # change all substrings with 'Yes' to 1

        return X


# Define a class to handle the final data validation
class DataValidator(BaseEstimator, TransformerMixin):
    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame):
        # go through each column of the data:
        for col in X.columns:
            # make sure every column is numeric
            X[col] = pd.to_numeric(X[col])

            # apply min-max normalization to each column using MinMaxScaler()
            # NOTE: If minmaxscaler.pkl exists, use that. Else, save a new one.
            # BASE_DIR = Path("Preprocessing.py").parent.resolve()
            # SCALER_NAME = 'minmaxscaler_' + str(col) + '.pkl'
            # SCALER_PATH = BASE_DIR / "completed_pipelines" / SCALER_NAME
            # print(SCALER_PATH)

            # if self.write_file:
            #     scaler = MinMaxScaler()
            #     scaler.fit(X[col])
            #     X[col] = scaler.transform(X[col])

            #     #joblib.dump(scaler, SCALER_PATH)
            # else:
            #     scaler = joblib.load(SCALER_PATH)
            #     X[col] = scaler.transform(X[col])

        return X
