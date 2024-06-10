import pandas as pd
from datetime import datetime, timedelta
import re
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import os
import sys
from pathlib import Path
import numpy as np


# Functions for preprocessing data for the Difference-in-Difference- and content analysis


def change_txt_files(file_path):
    txt_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(".txt")]
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            keyword = "© 2024 data.ai"
            filtered_lines = [line for line in lines if keyword not in line]
        with open(txt_file, "w", encoding="utf-8") as file:
            file.writelines(filtered_lines)


def change_txt_files_last(file_path):
    txt_files = [os.path.join(file_path, file) for file in os.listdir(file_path) if file.endswith(".txt")]
    for txt_file in txt_files:
        counter = 0
        with open(txt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()


def read_txt_files(file_path):
    # Read a text file and retrieve all app-specific attributes
    unified_app_id = 0
    multihoming = 0
    app_id = 0
    app_languages = None
    app_size = None
    worldwide_release_date = None
    initial_release_date = None
    app_category = None
    paid_finder = "DoNotUseThis"
    paid = 0
    in_app_purchases = 0

    with open(file_path, "r", encoding="utf-8") as file:
        app_name = os.path.basename(file_path)[:-4]
        # Remove certain special characters
        app_name = re.sub(r"[^\w\s!.:]", " ", app_name)
        # Remove unnecessary spaces
        app_name = (re.sub(r"\s+", " ", app_name)).strip()
        # Read the entire file as a single string
        content = file.read()
        # Split contents into lines
        lines = content.splitlines()
        previous_line = None
        # Read the text file backwards until beginning and extract relevant app attributes
        for line in reversed(lines):
            if unified_app_id == 0 and line.startswith("Unified App ID:"):
                unified_app_id = int(line.split()[3])
                multihoming = 1
            elif app_id == 0 and line.startswith("App ID:"):
                app_id = int(line.split()[2])
            elif app_languages is None and line.startswith("Languages"):
                if len(line.split()) == 1:
                    app_languages = len(previous_line.split(","))
                else:
                    app_languages = len(line.split(","))
            elif app_size is None and line.startswith("Size"):
                if len(line.split()) == 2:
                    if line.split()[0][-1] == "k":
                        app_size = float(line.split()[0][4:-1]) * 1000
                    else:
                        app_size = float(line.split()[0][4:])
                else:
                    if previous_line.split()[0][-1] == "k":
                        app_size = float(previous_line.strip().split()[0][:-1]) * 1000
                    else:
                        app_size = float(previous_line.strip().split()[0])
            elif worldwide_release_date is None and line.startswith("Worldwide Release Date"):
                if len(line.split()) == 5:
                    worldwide_release_date = datetime.strptime(line[22:], "%b %d, %Y").date()
                else:
                    worldwide_release_date = datetime.strptime(previous_line.strip(), "%b %d, %Y").date()
            elif initial_release_date is None and line.startswith("Initial Release Date"):
                if len(line.split()) == 5:
                    initial_release_date = datetime.strptime(line[20:], "%b %d, %Y").date()
                else:
                    initial_release_date = datetime.strptime(previous_line.strip(), "%b %d, %Y").date()
            elif app_category is None and line.startswith("Category"):
                if len(line) > 8:
                    app_category = line[8:]
                    paid_finder = app_category + " (Applications)"
                else:
                    app_category = previous_line.strip()
                    paid_finder = app_category + " (Applications)"
            elif line.startswith("See More"):
                if previous_line.startswith("Top In-App Purchases"):
                    in_app_purchases = 1
            elif line.startswith(paid_finder):
                if previous_line.startswith("Free"):
                    paid = 0
                else:
                    paid = 1
                break
            previous_line = line

    df = pd.DataFrame([[app_id, unified_app_id, app_name, app_category, worldwide_release_date,
                        initial_release_date, app_size, app_languages, multihoming, paid, in_app_purchases]],
                      columns=["app_id", "unified_app_id", "app_name", "app_category", "worldwide_release",
                               "initial_release", "app_size", "app_lang", "multihoming", "paid",
                               "in_app_purchases"])
    return df


def read_complete_releasenotes(txt_file):
    # Read text files and retrieve version numbers, version dates and corresponding release notes
    translator = GoogleTranslator(source="auto", target="en")
    DetectorFactory.seed = 0
    df = pd.DataFrame(columns=["app_version", "app_version_date", "release_note"])
    with open(txt_file, "r", encoding="utf-8") as file:
        # 3 patterns to match all possible combinations of app version formats
        pattern = r"(?i)^Version\s+([\w+\.\+\-]+)\s*(?:\((\d+)\)\s*)?\((\w+\s+\d+,\s+\d+)\)$"
        # pattern 2 for shorter strings of version + number
        pattern2 = r"(?i)^Version\s+([\d+\.\+\-]+)$"
        # pattern 3 for versions with "OS tested" marker
        pattern3 = r"(?i)^Version? (\d+(\.\d+)*) \((.*?) \d+(\.\d+)* Tested\)$"
        version_list = []
        include = False
        rn = ""

        for line in file:
            if line.startswith("What's New"):
                break
        for line in file:
            if line.startswith("Users may also like"):
                app_version = version_list[-1].split()[1]
                app_version_date = datetime.strptime((" ".join(version_list[-1].split()[2:5])[1:-1]),
                                                     "%b %d, %Y").date()
                rn = re.sub(r"[^\w\s!.:]", " ", rn)
                rn = re.sub(r"\s+", " ", rn).strip()
                df1 = pd.DataFrame([[app_version, app_version_date, rn]],
                                   columns=["app_version", "app_version_date", "release_note"])
                df = pd.concat([df, df1], ignore_index=True, axis=0)
                break
            if re.match(pattern, line):
                if len(line.split()) == 6:
                    line = " ".join(line.split()[:2] + line.split()[3:])
                if line in version_list:
                    include = False
                    continue
                else:
                    include = True
                    version_list.append(line)
                    if len(version_list) > 1:
                        app_version = version_list[-2].split()[1]
                        app_version_date = datetime.strptime((" ".join(version_list[-2].split()[2:5])[1:-1]),
                                                             "%b %d, %Y").date()
                        rn = re.sub(r"[^\w\s!.:]", " ", rn)
                        rn = re.sub(r"\s+", " ", rn).strip()
                        df1 = pd.DataFrame([[app_version, app_version_date, rn]],
                                           columns=["app_version", "app_version_date", "release_note"])
                        df = pd.concat([df, df1], ignore_index=True, axis=0)
                        rn = ""
            elif include:
                if re.match(pattern2, line.strip()):
                    continue
                elif re.match(pattern3, line.strip()):
                    include = False
                    continue
                else:
                    try:
                        if detect(line) != "en":
                            rn = rn + translator.translate(text=line.strip()) + " "
                        else:
                            rn = rn + line.strip() + " "
                    except:
                        continue
    return df


def create_csv_files(relative_path):
    # Read text files in the given relative path and convert them into csv files
    txt_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".txt")]
    for txt_file in txt_files:
        df1 = read_txt_files(txt_file)
        df2 = read_complete_releasenotes(txt_file)
        df1_adjusted = pd.concat([df1] * len(df2), ignore_index=True)
        df = pd.concat([df1_adjusted, df2], axis=1)
        df.to_csv(txt_file[:-4] + ".csv", index=False)
        print("File: " + os.path.basename(txt_file)[:-4] + " was successfully created!")


def label_time_dummies(core_app, relative_path):
    # Label time dummies depending on core app entry date
    match core_app.lower():
        case "journal":
            release_date = datetime(2023, 10, 26)
            folder_name = re.split(r"[\\/]", relative_path)[2] + " " + core_app.title()
            os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)
        case "translate":
            release_date = datetime(2020, 6, 22)
            folder_name = re.split(r"[\\/]", relative_path)[2] + " " + core_app.title()
            os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)
        case "freeform":
            release_date = datetime(2022, 10, 25)
            folder_name = re.split(r"[\\/]", relative_path)[2] + " " + core_app.title()
            os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)
        case _:
            print("Invalid input value!")
            sys.exit(1)

    # Get all relevant dates for time dummies
    one_month_before = release_date - timedelta(days=1 * 30)
    two_months_before = release_date - timedelta(days=2 * 30)
    three_months_before = release_date - timedelta(days=3 * 30)
    four_months_before = release_date - timedelta(days=4 * 30)
    one_month_after = release_date + timedelta(days=1 * 30)
    two_months_after = release_date + timedelta(days=2 * 30)
    three_months_after = release_date + timedelta(days=3 * 30)
    four_months_after = release_date + timedelta(days=4 * 30)

    labelled_data = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]

    # Iterate through all files and add time dummy lists
    for csv_file in labelled_data:
        print(os.path.basename(csv_file)[:-4])
        df = pd.read_csv(csv_file)

        # Skip apps that were released during observation time
        if datetime.strptime(df["app_version_date"].iloc[-1], "%Y-%m-%d") > four_months_before:
            continue

        time_dummy_lists = [[], [], [], [], [], [], [], []]

        for date_string in df["app_version_date"]:
            date = datetime.strptime(date_string, "%Y-%m-%d")
            is_within_timeframe_list = [four_months_before <= date < three_months_before,
                                        three_months_before <= date < two_months_before,
                                        two_months_before <= date < one_month_before,
                                        one_month_before <= date < release_date,
                                        one_month_after > date >= release_date,
                                        two_months_after > date >= one_month_after,
                                        three_months_after > date >= two_months_after,
                                        four_months_after > date >= three_months_after]

            # Fill time dummy lists
            for i, dummy_list in enumerate(time_dummy_lists):
                dummy_list.append(int(is_within_timeframe_list[i]))

        for i, dummy_list in enumerate(time_dummy_lists):
            df[f"dummy_{i+1}"] = dummy_list

        file_path = os.path.join(relative_path, folder_name, os.path.basename(csv_file))
        # Create and save a new csv file with all time dummies
        df.to_csv(file_path[:-4] + "_labelled.csv", index=False)
    return os.path.join(relative_path, folder_name)


def create_did_data(core_app, relative_path, affected):
    # Make each csv file in the given path ready for the Difference-in-Difference analysis
    # Add update frequencies per month, time dummies and averaged days elapsed since release (app-age)
    match core_app.lower():
        case "journal":
            release_date = datetime(2023, 10, 26)
        case "translate":
            release_date = datetime(2020, 6, 22)
        case "freeform":
            release_date = datetime(2022, 10, 25)
        case _:
            print("Invalid input value!")
            sys.exit(1)

    folder_name = "DiD Data"
    os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)

    # Get all relevant date times in the observation period and save them in date_list
    date_list = [0, 0, 0, 0, 0, 0, 0, 0]

    for i, j in zip(range(3, -1, -1), range(4, 8)):
        date_list[i] = release_date - timedelta(days=(j - 3) * 30)
        date_list[j] = release_date + timedelta(days=(j - 3) * 30)

    print(date_list)

    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    columns = ["dummy_1", "dummy_2", "dummy_3", "dummy_4", "dummy_5", "dummy_6", "dummy_7", "dummy_8"]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        release = df["app_version_date"].iloc[-1]
        print(release)
        # Count days since app release
        avg_days = (((date_list[0] - datetime.strptime(release, "%Y-%m-%d")) +
                     (date_list[1] - datetime.strptime(release, "%Y-%m-%d"))) / 2).days
        avg_days_list = [avg_days]
        for i in range(1, 8):
            avg_days_list.append(avg_days+30*i)

        update_frequencies = [0, 0, 0, 0, 0, 0, 0, 0]
        function_frequencies = [0, 0, 0, 0, 0, 0, 0, 0]

        # Include function update frequencies
        if "app_function" in df.columns:
            for i, row in df.iterrows():
                # Add up update frequencies (and function update frequencies) if in the observation period
                if date_list[7] > datetime.strptime(row["app_version_date"], "%Y-%m-%d") >= date_list[0]:
                    for index, column in enumerate(columns):
                        update_frequencies[index] = update_frequencies[index] + row[column]
                        if row[column] == 1:
                            function_frequencies[index] = function_frequencies[index] + row["app_function"]
                else:
                    continue
        else:
            for i, row in df.iterrows():
                # Add up update frequencies when an update lies in the observation period
                if date_list[7] > datetime.strptime(row["app_version_date"], "%Y-%m-%d") >= date_list[0]:
                    for index, column in enumerate(columns):
                        update_frequencies[index] = update_frequencies[index] + row[column]
                else:
                    continue

        # Select column names from static app content
        selected_columns = df.columns[:11]

        # Select static app content
        selected_row = df.iloc[0, 0:11]

        # Create a dataframe for the observation period (8 months) with static app content
        df_adjusted = pd.concat([pd.DataFrame([selected_row], columns=selected_columns)] * 8, ignore_index=True)

        # Add updates per month
        df_adjusted["updates_per_month"] = update_frequencies

        # Add functional updates per month
        df_adjusted["function_updates_per_month"] = function_frequencies

        identity_matrix = np.eye(8).astype(int)
        month_indicator = []
        for i in range(0, 8):
            # Add time dummies for each month
            df_adjusted[f"time_dummy_{i + 1}"] = identity_matrix[i]
            month_indicator.append(f"month_{i + 1}")

        # Add month indicator
        df_adjusted["month_indicator"] = month_indicator

        # Add treatment indicator
        if affected:
            df_adjusted["affected"] = [1] * 8
        else:
            df_adjusted["affected"] = [0] * 8

        # Add after-entry indicator
        df_adjusted["after_entry"] = [0, 0, 0, 0, 1, 1, 1, 1]

        # Add averaged elapsed days since app release
        df_adjusted["elapsed_avg_days"] = avg_days_list

        # Add avg days over all recorded avg days
        df_adjusted["avg_age_days"] = np.mean(avg_days_list)

        file_path = os.path.join(relative_path, folder_name, os.path.basename(csv_file))

        # Create and save a new csv file with DiD data
        df_adjusted.to_csv(file_path[:-4] + "_DiD.csv", index=False)
    return os.path.join(relative_path, folder_name)


def label_function_updates(relative_path):
    # Manually label each release note entry of an application during observation period.
    # Read and export csv files only!
    # Labels: 0 = no functional changes, 1 = new functional changes
    folder_name = "Labelled Function Updates"
    os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]

    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        print(df["app_name"].iloc[0])
        function = []
        for index, row in df.iterrows():
            if all(row["dummy_1":"dummy_8"] == 0):
                function.append(0)
                continue
            else:
                while True:
                    print(dict(row[["app_version", "release_note"]]))
                    user_input = input("Enter a label for 'functional changes' (0 = False or 1 = True): ")
                    label = user_input.strip()

                    # Check if the input is either 0 or 1
                    if label in ["0", "1"]:
                        # Convert labels to int
                        function.append(int(label))
                        print("\n")
                        break
                    else:
                        print("Invalid input! Please enter a label as '0' or '1'.")

        df["app_function"] = function
        file_path = os.path.join(relative_path, folder_name, os.path.basename(csv_file))
        df.to_csv(file_path[:-4] + "_function.csv", index=False)


def create_single_dataframe(relative_path):
    # Concatenate all available csv files in the current directory and save them in one file
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]
    folder_name = "Merged Data"
    os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)
    path_object = Path(relative_path)

    # Check if there are csv files and save the first csv file for further processing
    if csv_files:
        first_csv_file = csv_files[0]
        first_df = pd.read_csv(first_csv_file)

        # Concatenate all dataframes and export them as a single csv file for the DiD analysis
        for csv_file in csv_files[1:]:
            df = pd.read_csv(csv_file)
            first_df = pd.concat([first_df, df], ignore_index=True)

        # Export concatenated data
        first_df.to_csv(os.path.join(relative_path, folder_name, path_object.parts[3] + ".csv"), index=False)
        return os.path.join(relative_path, folder_name, path_object.parts[3] + ".csv")
    else:
        print("No data available!")


def create_single_dataframe_descriptions(relative_path):
    # Concatenate all available csv files in the current directory and save them in a single file
    csv_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".csv")]

    folder_name = "Merged App Descriptions"
    os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)

    # Check if there are .csv files and save the first csv file for further processing
    if csv_files:
        first_csv_file = csv_files[0]
        first_df = pd.read_csv(first_csv_file)

        # Concatenate all dataframes and export them as a single csv file for the DiD analysis
        for csv_file in csv_files[1:]:
            df = pd.read_csv(csv_file)
            first_df = pd.concat([first_df, df], ignore_index=True)

        # Export concatenated data
        first_df.to_csv(os.path.join(relative_path, folder_name, "Merged Descriptions.csv"), index=False)
        return os.path.join(relative_path, folder_name, "Merged Descriptions.csv")
    else:
        print("No data available!")


def analyze_merged_did(app_group1, app_group2, core_app):
    df1 = pd.read_csv("Groups/Merged Data/" + app_group1 + " " + core_app.title() + ".csv")
    df2 = pd.read_csv("Groups/Merged Data/" + app_group2 + " " + core_app.title() + ".csv")
    df3 = pd.concat([df1, df2], ignore_index=True)
    df3.to_csv(os.path.join("Groups/Merged Data/", app_group1 + " " + app_group2 + " " +
                            core_app.title() + ".csv"), index=False)


def label_data(core_app, relative_path, affected):
    # Add time dummies and create DiD-data depending on the given core app
    next_path = label_time_dummies(core_app, relative_path)
    next_path = create_did_data(core_app, next_path, affected)
    next_path = create_single_dataframe(next_path)


def read_descriptions(relative_path):
    # Read text files in a given path and retrieve corresponding descriptions
    txt_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".txt")]
    folder_name = "App Descriptions"
    os.makedirs(os.path.join(relative_path, folder_name), exist_ok=True)

    for txt_file in txt_files:
        df = read_txt_files(txt_file)
        description = ""
        with open(txt_file, "r", encoding="utf-8") as file:
            for line in file:
                if line.startswith("App Description"):
                    break
            for line in file:
                if line.strip() == "Displayed Avg Ratings":
                    if len(description) <= 8:
                        break
                    else:
                        description = description[:-8].strip()
                        break
                if len(line.strip()) == 0:
                    continue
                description = description + line.strip() + " "
            # Remove special characters
            description = re.sub(r"[^\w\s!.:]", " ", description)
            # Remove unnecessary spaces
            description = (re.sub(r"\s+", " ", description)).strip()
            df1 = pd.DataFrame({"app_description": [description]})
            df = pd.concat([df, df1], axis=1)
            file_path = os.path.join(relative_path, folder_name, os.path.basename(txt_file))
            df.to_csv(file_path[:-4] + "_description.csv", index=False)

