import pandas as pd
import streamlit as st
from collections import defaultdict
from src.cleaner import DataCleaner
from src.data_io import load_csv


@st.cache_data(show_spinner=False)
def build_location_structure(filepath: str) -> dict:
    """.
    Build a nested location structure from a CSV file.

    The structure is a dictionary of provinces, each mapping to another dictionary
    of localities, each containing a set of postal codes.

    Example:
        {
            'Antwerp': {
                'Berchem': {2600, 2610},
                'Mechelen': {2800}
            },
            ...
        }

    :param filepath: Path to the CSV file containing 'province', 'locality', and 'postCode' columns.
    :return: Nested dictionary of province -> locality -> set of postcodes.
    """
    df = prepare_data(filepath)

    structure = defaultdict(default_factory)

    for _, row in df.iterrows():
        province = row['province']
        locality = row['locality']
        post_code = int(row['postCode'])
        structure[province][locality].add(post_code)

    return structure


def prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the location-related data from a CSV file.
    It keeps only the necessary columns, removes duplicates, and normalizes text in the 'province' column.

    :param filepath: Path to the CSV file.
    :return: Cleaned pandas DataFrame with columns: 'province', 'locality', 'postCode'.
    """
    df = load_csv(filepath)
    df = df[['province', 'locality', 'postCode']]
    cleaner = DataCleaner(df)
    cleaner.remove_duplicates()
    cleaner.normalize_text_columns(['province'])

    return cleaner.df


def default_factory():
    """
    Factory function for creating a defaultdict of sets.
    Used to build a nested structure: province -> locality -> set of postcodes.

    :return: defaultdict with sets as default values.
    """
    return defaultdict(set)

