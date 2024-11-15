import pandas as pd
import sqlalchemy
import structlog
import typer


app = typer.Typer()
log = structlog.get_logger()


def load_data(messages_filepath: str, categories_filepath: str) -> pd.DataFrame:
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    data_merged = messages.merge(right=categories, on="id", how="inner")
    return data_merged


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    categories_split = data["categories"].str.split(pat=";", expand=True)
    category_colnames = categories_split.iloc[0].str.split("-").str[0]
    categories_split.columns = category_colnames
    categories_split = categories_split.apply(
        lambda col: col.str.split("-").str[1].astype(int)
    )
    data_expanded = pd.concat(
        [data.drop(columns=["categories"]), categories_split], axis=1
    )
    data_expanded = data_expanded.drop_duplicates()

    # "child_alone" column contains only zeros - it will be dropped.
    data_expanded = data_expanded.drop(columns=["child_alone"])

    # "related" column contains not only zeros and ones but also 2.
    # It was discovered that when there is 2, all other target columns are 0.
    # Therefore all the rows with related ==2 will be dropped (less than 200).
    data_expanded = data_expanded[data_expanded["related"] != 2]

    return data_expanded


def save_data(data: pd.DataFrame, database_filepath: str):
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
):
    log.info(
        f"Loading data...\n    MESSAGES: {messages_filepath}\n    CATEGORIES: {categories_filepath}"
    )
    data_loaded = load_data(
        messages_filepath=messages_filepath, categories_filepath=categories_filepath
    )

    log.info("Cleaning data...")
    data_cleaned = clean_data(data=data_loaded)

    log.info(f"Saving data...\n    DATABASE: {database_filepath}")
    save_data(data=data_cleaned, database_filepath=database_filepath)

    log.info(f"Cleaned data saved to database {database_filepath}.")


if __name__ == "__main__":
    app()
