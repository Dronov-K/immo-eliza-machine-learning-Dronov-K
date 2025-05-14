import pandas as pd


def load_csv(filepath: str, sep: str = ',', encoding: str = 'utf-8') -> pd.DataFrame:
    """
    Loads a CSV file into a DataFrame with error handling.

    :param filepath: The path to the file.
    :param sep: The delimiter used in the file.
    :param encoding: The encoding of the file.
    :return: pd.DataFrame: pandas DataFrame with loaded data from csv file.
    """
    try:
        df = pd.read_csv(filepath, sep=sep, encoding=encoding)
        print(f"File {filepath} uploaded successfully")
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path: {filepath}")
    except pd.errors.ParserError:
        raise pd.errors.ParserError(f"Unable to parse CSV file: {filepath}")
    except UnicodeError:
        raise UnicodeError(f"Encoding error reading file: {filepath}")
    except Exception as e:
        raise Exception(f"Unknown error loading file {filepath}: {str(e)}")

    return df


def save_csv(df: pd.DataFrame, output_file: str, mode='w', sep: str = ',', encoding: str = 'utf-8') -> None:
    """
    Saves the cleaned DataFrame to a CSV file.да не

    :param df: DataFrame to save
    :param output_file: The path to the output file.
    :param mode: The write mode ('w' for overwriting, 'a' for appending) default is 'w'.
    :param sep: The delimiter to use between columns (default is ',').
    :param encoding: The encoding format to use for the output file (default is 'utf-8').
    :return: None
    """
    try:
        df.to_csv(output_file, mode=mode, sep=sep, encoding=encoding, index=False)
        print(f"File {output_file} created successfully")
    except FileNotFoundError:
        raise FileNotFoundError(f"Write path doesn't exist: {output_file}")
    except PermissionError:
        raise PermissionError(f"No permission to write to file: {output_file}")
    except Exception as e:
        raise Exception(f"Unknown error writing file {output_file}: {str(e)}")
