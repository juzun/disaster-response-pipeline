from typing import Callable, List, Optional
import nltk
from nltk.tokenize import sent_tokenize
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Custom transformer to extract a feature indicating whether the first word
    in a sentence of a text message is a verb or the retweet indicator ('RT').
    """

    def __init__(self, tokenizer: Callable[[str], List[str]], messages_col_name: Optional[str] = None):
        """
        Initialize the StartingVerbExtractor with a tokenizer and column name.

        Args:
            tokenizer (Callable[[str], List[str]]): A function to tokenize sentences into words.
            messages_col_name (str): The name of the column in the input DataFrame
                                     containing the text messages.
        """
        self.tokenizer = tokenizer
        self.messages_col_name = messages_col_name

    def starting_verb(self, text: str) -> bool:
        """
        Determine if the first word of any sentence in the text is a verb or 'RT'.

        Args:
            text (str): A text message to analyze.

        Returns:
            bool: True if the first word in at least one sentence is a verb or 'RT',
                  False otherwise.
        """
        # Tokenize the input text into sentences
        sentence_list = sent_tokenize(text)

        for sentence in sentence_list:
            # Tokenize the sentence into words and tag parts of speech
            pos_tags = nltk.pos_tag(self.tokenizer(sentence))
            # Skip empty sentences
            if not pos_tags:
                continue

            # Extract the first word and its part-of-speech tag
            first_word, first_tag = pos_tags[0]

            # Check if the first word is a verb or the retweet indicator ('RT')
            if first_tag in ["VB", "VBP"] or first_word == "RT":
                return True

        return False

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "StartingVerbExtractor":
        """
        Fit method (no operation) for compatibility with scikit-learn pipelines.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.DataFrame, optional): Target labels (not used).

        Returns:
            StartingVerbExtractor: The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the starting_verb feature extraction to a DataFrame column.

        Args:
            X (pd.DataFrame): Input DataFrame containing the messages column.

        Returns:
            pd.DataFrame: A DataFrame with a single column containing the feature
                          (True/False) for each message.
        """
        # Try to apply the starting_verb method to the specified column
        try:
            if not isinstance(X, pd.Series):
                X_tagged = X[self.messages_col_name].apply(self.starting_verb)
            else:
                X_tagged = X.apply(self.starting_verb)
        # Raise a KeyError if the column name is incorrect
        except KeyError as error_message:
            raise KeyError(
                f"Wrong column name for messages text was used: {error_message}. Available columns: {list(X.columns)}"
            )
        return pd.DataFrame(X_tagged)


class GenreTransformer(BaseEstimator, TransformerMixin):
    """
    Custom transformer to encode a 'genre' column into one-hot encoded features.
    """

    def __init__(self, genre_col_name: str):
        """
        Initialize the GenreTransformer with the column name.

        Args:
            genre_col_name (str): The name of the column in the input DataFrame
                                  containing genre information.
        """
        self.genre_col_name = genre_col_name

    def fit(self, X: pd.DataFrame, y: pd.DataFrame = None) -> "GenreTransformer":
        """
        Fit method (no operation) for compatibility with scikit-learn pipelines.

        Args:
            X (pd.DataFrame): Input DataFrame.
            y (pd.DataFrame, optional): Target labels (not used).

        Returns:
            GenreTransformer: The instance itself.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode the genre column.

        Args:
            X (pd.DataFrame): Input DataFrame containing the genre column.

        Returns:
            pd.DataFrame: A DataFrame with one-hot encoded genre columns.
        """
        # Generate one-hot encoded features for the genre column
        genres_encoded = pd.get_dummies(X[self.genre_col_name], drop_first=False)
        return genres_encoded
