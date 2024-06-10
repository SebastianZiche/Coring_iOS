import pandas as pd
import os
import preprocessing
from pathlib import Path
import analysis
import shutil

pd.set_option("display.width", 10000)
pd.set_option("display.max_columns", 21)
pd.set_option("display.max_colwidth", None)

# ---Quantitative Analysis----------------------------------------------------------------------------------------------

# 1. Preprocessing

# Set up treatment- and control group directories
treatment_group_directory = "Groups/Treatment Groups/"
control_group_directory = "Groups/Control Groups/"
treatment_groups = ["Freeform", "Journal", "Translate"]
control_groups = ["Messenger", "Video", "Streaming", "Voice"]
core_apps = ["freeform", "journal", "translate"]

# Parse text files and convert them to csv files for further analysis.
# !!! CSV CREATION PROCESS TAKES SEVERAL HOURS DUE TO GOOGLE TRANSLATOR ENGINE !!!

for directory in treatment_groups:
    path = os.path.join(treatment_group_directory, directory)
    preprocessing.create_csv_files(path)

for directory in control_groups:
    path = os.path.join(control_group_directory, directory)
    preprocessing.create_csv_files(path)


# Add time dummies and edit all previously created .csv files for the DiD-Analysis
for core_app in core_apps:
    for directory in treatment_groups:
        path = os.path.join(treatment_group_directory, directory)
        if directory.lower() == core_app:
            preprocessing.label_data(core_app, path, True)
        else:
            preprocessing.label_data(core_app, path, False)

for core_app in core_apps:
    for directory in control_groups:
        path = os.path.join(control_group_directory, directory)
        preprocessing.label_data(core_app, path, False)


# 2. DiD-Analysis
# Perform all necessary steps for the DiD-analysis
analysis.full_analysis("Translate Translate", "Voice Translate")
analysis.full_analysis("Freeform Freeform", "Translate Freeform")
analysis.full_analysis("Journal Journal", "Freeform Journal")


# 3. Function-related Analysis
# Label affected applications in terms of function updates
for directory in treatment_groups:
    next_directory = directory + " " + directory
    path = os.path.join(treatment_group_directory, directory, next_directory)
    preprocessing.label_function_updates(path)
    next_path = os.path.join(path, "Labelled Function Updates")
    preprocessing.create_did_data(directory, next_path, True)
    next_path = os.path.join(next_path, "DiD Data")
    preprocessing.create_single_dataframe(next_path)
    last_path = os.path.join(next_path, "Merged Data")
    shutil.copy(f"{last_path}/{directory} {directory}.csv", "Merged Function Data")
    analysis.print_avg_function_updates(directory + " " + directory)
    analysis.plot_function_update_freq(directory + " " + directory)


# ---Qualitative Analysis-----------------------------------------------------------------------------------------------

# Several steps to analyze reviews, descriptions and to decide which applications to pick for the qualitative analysis

# 1. Create n-grams of review files of affected applications
analysis.analyze_reviews_file("Reviews/Journal Review Files/Journal_Reviews.csv", 100, 5)
analysis.analyze_reviews_file("Reviews/Freeform Review Files/Freeform_Reviews.csv", 100, 5)
analysis.analyze_reviews_file("Reviews/Translate Review Files/Translate_Reviews.csv", 100, 5)

# 2. Read app descriptions of affected applications and create n-grams
for directory in treatment_groups:
    path = os.path.join(treatment_group_directory, directory)
    preprocessing.read_descriptions(path)
    next_path = os.path.join(path, "App Descriptions")
    last_path = preprocessing.create_single_dataframe_descriptions(next_path)
    analysis.analyze_descriptions_file(last_path, 50, 5)

# 3. Print out number of ratings of all treatment group applications
for directory in treatment_groups:
    path = os.path.join(treatment_group_directory, directory)
    analysis.parse_txt_files(path)
    print("\n")

