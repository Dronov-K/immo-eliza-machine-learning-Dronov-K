import pandas as pd
import numpy as np
import sklearn.compose
from src.cleaner import DataCleaner
from src.cleaning_config import CleaningConfig
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path


class ModelTrainer:
    """
    Class for building, training and tuning ML model using a pipeline and GridSearchCV.
    """
    def __init__(self, df: pd.DataFrame, target_column: str, cleaner_config: CleaningConfig):
        """
        Initializing a class.

        :param df: pd.DataFrame.
        :param target_column: Name of the target feature (what we are predicting).
        """
        self.df = df
        self.target_column = target_column
        self.cleaner_config = cleaner_config
        self.X_train = self.X_test = self.y_train = self.y_test = None
        self.pipeline = None

    def _prepare_data(self) -> None:
        """
        Prepares the data for training and testing.
        This method uses the DataCleaner to clean the dataset with the provided parameters.
        It then splits the cleaned data into features (X) and target (y), and performs a
        train-test split with 80% for training and 20% for testing.

        :return: None
        """
        cleaner = DataCleaner(self.df)
        self.df = cleaner.clean_all(self.cleaner_config)

        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def _build_preprocessor(self) -> sklearn.compose.ColumnTransformer:
        """
        Builds a preprocessing pipeline for numerical and categorical features.
        Numerical features are imputed using the mean strategy and scaled using StandardScaler.
        Categorical features are imputed with the most frequent value and encoded using OrdinalEncoder.
        The method returns a ColumnTransformer that applies the appropriate transformations
        to each column type based on the training data.

        :return: A ColumnTransformer object for preprocessing the dataset.
        """
        numeric_cols = self.X_train.select_dtypes(include=['number']).columns.tolist()
        categorical_cols = self.X_train.select_dtypes(include=['object', 'bool', 'category']).columns.tolist()

        numeric_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
        ])

        preprocessor = ColumnTransformer([
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

        return preprocessor

    def find_best_hyperparameters(self, model, param_grid: dict, cv: int = 5,
                                  scoring: str = 'neg_mean_absolute_error') -> None:
        """
        Finding the best hyperparameters using GridSearchCV.

        :param model: Model to use in pipeline. Required if pipeline is not already built (default is None).
        :param param_grid: Dictionary of parameters to search through (default is None).
        :param cv: Number of folds for cross-validation (default is 2).
        :param scoring: Metric for optimization (default is "mae").
        :raises ValueError if cv param less than 2 and if model not.
        :return: None
        """
        if cv < 2:
            raise ValueError('cv must be at least 2 or higher')

        self._prepare_data()
        preprocessor = self._build_preprocessor()

        self.pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

        search = GridSearchCV(
            self.pipeline,
            param_grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            verbose=2
        )
        search.fit(self.X_train, self.y_train)

        print('Best parameters:', search.best_params_)
        self.pipeline = search.best_estimator_

    def train(self) -> None:
        """
        Train the model based on the best hyperparameters (if GridSearch was called).

        :raises ValueError: If no pipeline has been created via find_best_hyperparameters().
        """
        if self.pipeline is None:
            raise ValueError('No model to train. Use find_best_hyperparameters() first.')

        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self) -> None:
        """
        Evaluate the trained model on the test set using MAE.

        :return: None
        """
        y_pred = self.pipeline.predict(self.X_test)
        mae = mean_absolute_error(self.y_test, y_pred)
        r2 = r2_score(self.y_test, y_pred)
        print(f'MAE: {mae:.2f}')
        print(f'R2: {r2:.2f}')

    def save_pipeline(self, filepath: str) -> None:
        """
        Saves the trained pipeline to a file.
        If the pipeline has not been trained, raises a ValueError.
        Creates the directory if it does not exist and saves the pipeline using joblib.

        :param filepath: The path where the pipeline should be saved.
        :return: None
        """
        if self.pipeline is None:
            raise ValueError('Pipeline has not been trained.')
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"Pipeline save: {filepath}")

    def load_pipeline(self, filepath: str) -> None:
        """
        Loads a pipeline from a file.
        Uses joblib to load a previously saved pipeline and assigns it to the instance.

        :param filepath: The path to the saved pipeline file.
        :return: None
        """
        self.pipeline = joblib.load(filepath)
        print(f"Pipeline download: {filepath}")

    def analyze_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Analyzes feature importance after training the model.
        Supported models are `feature_importances_` (trees)
        and `coef_` (linear models).

        :param top_n: Number of top features to display (default 20).
        :return: DataFrame with signs and their importance
        """
        if self.pipeline is None:
            raise ValueError('Train the model first (pipeline).')

        model = self.pipeline.named_steps['model']
        preprocessor = self.pipeline.named_steps['preprocessor']

        # Get the names of the features
        def get_feature_names(preprocessor, input_features):
            output_features = []

            for name, transformer, cols in preprocessor.transformers_:
                if name == 'remainder':
                    continue
                if hasattr(transformer, 'get_feature_names_out'):
                    try:
                        feat_names = transformer.get_feature_names_out(cols)
                    except:
                        feat_names = cols
                else:
                    feat_names = cols
                output_features.extend(feat_names)

            return output_features

        feature_names = get_feature_names(preprocessor, self.X_train.columns)

        # Get importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_)
        else:
            raise ValueError('The model does not support feature importance extraction')

        if len(feature_names) != len(importances):
            raise ValueError('The size of features and importances do not match')

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values(by='importance', ascending=False)

        print(f"\nTop-{top_n} sings by importance:")
        print(importance_df.head(top_n))

        return importance_df
