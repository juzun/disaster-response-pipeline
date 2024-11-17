# import libraries
import nltk


nltk.download(["punkt", "wordnet", "averaged_perceptron_tagger"])

from typing import List, Tuple, Union
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import pickle
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
import sqlalchemy
from sqlalchemy import create_engine
import structlog
import typer

from disaster_response_pipeline.core.custom_transformers import StartingVerbExtractor


# Typer app for CLI commands
app = typer.Typer()

# Structlog logger for structured logging
log = structlog.get_logger()

LEMMATIZER = WordNetLemmatizer()
STOP_WORDS = stopwords.words("english")


def load_data(database_filepath: str) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Load data from an SQLite database.

    Args:
        database_filepath (str): Path to the SQLite database.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
            - X (pd.DataFrame): DataFrame containing message text.
            - y (pd.DataFrame): DataFrame containing target categories.
            - target_columns (List[str]): List of target category column names.
    """
    # Connect to the SQLite database
    engine = create_engine(f"sqlite:///{database_filepath}.db")

    # Load the first table from the database
    df = pd.read_sql(f"select * from {sqlalchemy.inspect(engine).get_table_names()[0]}", engine)

    # Identify target columns
    target_columns = []
    for col in df.columns:
        if col not in ["id", "message", "genre", "original"]:
            target_columns.append(col)

    # Separate features (X) and targets (y)
    X = df["message"]
    y = df[target_columns]

    return X, y, target_columns


def tokenize(text: str) -> List[str]:
    """
    Tokenize and preprocess text data.

    Args:
        text (str): Input text to be tokenized.

    Returns:
        List[str]: List of cleaned and lemmatized tokens.
    """
    # Regex to identify URLs
    url_regex = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)

    # Replace URLs with a placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Tokenize the text into words
    tokens = word_tokenize(text)

    # Clean and lemmatize tokens, excluding stopwords
    cleaned_tokens = [
        LEMMATIZER.lemmatize(token).lower().strip() for token in tokens if token not in STOP_WORDS
    ]

    return cleaned_tokens


def build_model() -> GridSearchCV:
    """
    Build a machine learning pipeline for disaster response classification.

    Returns:
        Pipeline: A scikit-learn pipeline object.
    """
    # Define the pipeline structure
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "text_pipeline",
                            Pipeline(
                                [("vect", CountVectorizer(tokenizer=tokenize)), ("tfidf", TfidfTransformer())]
                            ),
                        ),
                        ("starting_verb", StartingVerbExtractor(tokenizer=tokenize)),
                    ]
                ),
            ),
            ("clf", MultiOutputClassifier(estimator=RandomForestClassifier())),
        ],
        verbose=True,
    )

    # Define parameters to iterate through during grid search
    param_grid = {
        "clf__estimator__n_estimators": [100, 200],
        "clf__estimator__max_depth": [None, 10, 30],
    }

    # Initiate grid search.
    grid_search = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, n_jobs=-1)

    return grid_search


def evaluate_model(
    model: Union[Pipeline, GridSearchCV],
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    target_columns: List[str],
) -> None:
    """
    Evaluate a trained model using test data.

    Args:
        model (Union[Pipeline, GridSearchCV]): Trained model to evaluate.
        X_test (pd.DataFrame): Test features.
        y_test (pd.DataFrame): Test targets.
        target_columns (List[str]): List of target category names.
    """
    # Predict the categories for test data
    y_pred = model.predict(X_test)

    # Generate classification report
    report = pd.DataFrame(
        classification_report(
            y_pred=y_pred,
            y_true=y_test,
            target_names=target_columns,
            output_dict=True,
            zero_division=np.nan,
        )
    ).transpose()

    # Print the classification report
    print(report)

    # Print the best parameters found by grid search.
    print(model.best_params_)


def save_model(model: Union[Pipeline, GridSearchCV], model_filepath: str) -> None:
    """
    Save the trained model to a pickle file.

    Args:
        model (Union[Pipeline, GridSearchCV]): Trained model to save.
        model_filepath (str): Path to save the model (without extension).
    """
    with open(f"{model_filepath}.pkl", "wb") as file:
        pickle.dump(model, file)


@app.command()
def main(
    database_filepath: str = typer.Option(  # noqa B008
        help="Path where the database shall be load from. Example: data/DisasterResponse.",
    ),
    model_filepath: str = typer.Option(  # noqa B008
        help="Path where the trained model should be stored. Example: data/trained_model.",
    ),
) -> None:
    """
    Main CLI command to load data, build and train a model, evaluate, and save it.

    Args:
        database_filepath (str): Path to the database file.
        model_filepath (str): Path to save the trained model.
    """
    # Log data loading process
    log.info(f"Loading data...\n    DATABASE: {database_filepath}")
    X, y, target_columns = load_data(database_filepath=database_filepath)

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Building process
    log.info("Building model...")
    model = build_model()

    # Train the model
    log.info("Training model...")
    model.fit(X=X_train, y=y_train)

    # Evaluate the trained model
    log.info("Evaluating model...")
    evaluate_model(model=model, X_test=X_test, y_test=y_test, target_columns=target_columns)

    # Save the trained model
    log.info(f"Saving model...\n    MODEL: {model_filepath}")
    save_model(model, model_filepath)

    # Confirm successful save
    log.info(f"Trained model saved in {model_filepath}.")


if __name__ == "__main__":
    app()
