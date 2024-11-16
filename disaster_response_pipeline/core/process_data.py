import pandas as pd
import sqlalchemy
import structlog
import typer


# Typer app for CLI commands
app = typer.Typer()

# Structlog logger for structured logging
log = structlog.get_logger()


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    """
    Load and merge messages and categories datasets.

    Args:
        messages_filepath (str): Path to the CSV file containing messages data.
        categories_filepath (str): Path to the CSV file containing categories data.

    Returns:
        pd.DataFrame: Merged DataFrame of messages and categories.
    """
    # Load datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge datasets on the common 'id' column
    data_merged = messages.merge(right=categories, on="id", how="inner")
    return data_merged


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the merged dataset by splitting categories, converting to binary,
    removing duplicates or invalid rows and dropping unnecessary columns.

    Args:
        data (pd.DataFrame): Merged DataFrame of messages and categories.

    Returns:
        pd.DataFrame: Cleaned DataFrame with expanded category columns.
    """
    # Split the 'categories' column into separate category columns
    categories_split = data["categories"].str.split(pat=";", expand=True)

    # Extract category column names from the first row
    category_colnames = categories_split.iloc[0].str.split("-").str[0]
    categories_split.columns = category_colnames

    # Convert category values to integers (e.g., 'related-1' -> 1)
    categories_split = categories_split.apply(lambda col: col.str.split("-").str[1].astype(int))

    # Combine the original data with the expanded categories
    data_expanded = pd.concat([data.drop(columns=["categories"]), categories_split], axis=1)

    # Remove duplicate rows
    data_expanded = data_expanded.drop_duplicates()

    # Drop the 'child_alone' column as it contains only zeros
    data_expanded = data_expanded.drop(columns=["child_alone"])

    # Handle 'related' column: remove rows with 'related == 2'
    # As rows with 'related == 2' have all other targets as 0
    data_expanded = data_expanded[data_expanded["related"] != 2]

    return data_expanded


def save_data(data: pd.DataFrame, database_filepath: str) -> None:
    """
    Save the cleaned data to an SQLite database.

    Args:
        data (pd.DataFrame): Cleaned DataFrame to be saved.
        database_filepath (str): File path (excluding extension) for the database.
    """
    # Create a SQLAlchemy engine to write data to SQLite
    engine = sqlalchemy.create_engine(f"sqlite:///{database_filepath}.db")
    data.to_sql("twitter_data", engine, index=False)


@app.command()
def process_data(
    messages_filepath: str = typer.Option(  # noqa B008
        help="Path to a CSV file containing messages. Example: data/messages.csv.",
    ),
    categories_filepath: str = typer.Option(  # noqa B008
        help="Path to a CSV file containing categories. Example: data/categories.csv.",
    ),
    database_filepath: str = typer.Option(  # noqa B008
        help="Path where resulting database shall be stored and its name. Example: data/DisasterResponse.",
    ),
) -> None:
    """
    CLI command to process the data: load, clean, and save it to a database.

    Args:
        messages_filepath (str): Path to the messages CSV file.
        categories_filepath (str): Path to the categories CSV file.
        database_filepath (str): Path for the resulting SQLite database file.
    """
    # Log the start of the process
    log.info(f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}")
    data_loaded = load_data(messages_filepath=messages_filepath, categories_filepath=categories_filepath)

    # Log the cleaning process
    log.info("Cleaning data...")
    data_cleaned = clean_data(data=data_loaded)

    # Log the saving process
    log.info(f"Saving data...\n    DATABASE: {database_filepath}")
    save_data(data=data_cleaned, database_filepath=database_filepath)

    # Confirm completion
    log.info(f"Cleaned data saved to database {database_filepath}.")


if __name__ == "__main__":
    app()
