import os
import pandas as pd

# Helper functions for checking errors, missing data and specified strings


def check_for_strings(relative_path, text):
    # Check whether a specified text is in the text files of a specified path
    txt_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".txt")]
    count = 0
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            content = file.read()
            # Check if the string is present in the given file
            if text in content:
                print(f"{os.path.basename(txt_file)[:-4]} has the word/sentence {text} in it!")
                count += 1
    if text == "iPad Only":
        print(f"{count} applications are only available for iPads!")
    else:
        print(f"{count} applications have the word/sentence {text} in it!")


def check_errors(relative_path):
    # Check for errors in csv files in the given path
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Check for NaN values in multiple columns
        if df["app_id"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "app_id" column.')
        if df["unified_app_id"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "unified_app_id" column.')
        if df["app_category"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "app_category" column.')
        if df["worldwide_release"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "worldwide_release" column.')
        if df["initial_release"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "initial_release" column.')
        if df["app_lang"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "app_lang" column.')
        if df["multihoming"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "multihoming" column.')
        if df["paid"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "paid" column.')
        if df["in_app_purchases"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "in_app_purchases" column.')
        if df["app_version"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "app_version" column.')
        if df["app_version_date"].isnull().any():
            print(f'File {os.path.basename(csv_file)[:-4]} contains NaN values in the "app_version_date" column.')


def check_lang(relative_path):
    # Check for errors in the column app_lang
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df["app_lang"].unique() == 0:
            print(f"File {os.path.basename(csv_file)} has the value 0 in the column app_lang!")


def check_app_id(relative_path):
    # Check for errors in the column app_id
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        if df["app_id"].iloc[-1] == 0:
            print(f"File {os.path.basename(csv_file)} has the value 0 in the column app_id!")


def has_attributes(relative_path, column_to_check):
    # Check whether csv files in a given path have columns with NaN
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # Check for missing values in the specified column
        if df[column_to_check].isna().any():
            print(f"File {os.path.basename(csv_file)[:-4]} has NaN values in column {column_to_check}")


def has_row(relative_path):
    # Check for csv files with 0 rows in the given path and print them to the console
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # If there's no row (excluding the header), return True
        if len(df) == 0:
            print(f"File {os.path.basename(csv_file)[:-4]} has no rows!")

