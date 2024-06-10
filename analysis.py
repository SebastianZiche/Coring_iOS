import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk import word_tokenize
from collections import Counter
import statsmodels.formula.api as smf
from nltk.collocations import BigramCollocationFinder, TrigramCollocationFinder, QuadgramCollocationFinder
import re
import string
import os
import numpy as np
import sys

# Analytical functions for the quantitative and qualitative analysis


def print_avg_function_updates(app_cluster):
    # Print the total and average function updates before and after treatment for the given affected cluster
    if app_cluster.split()[0] != app_cluster.split()[1]:
        print("Invalid input data!")
        sys.exit(1)

    df = pd.read_csv("Merged Function Data/" + app_cluster + ".csv")
    new_dict = {0: "before", 1: "after"}
    for i in range(0, 2):
        filtered_df = df[(df["after_entry"] == i)]
        total_updates = filtered_df["function_updates_per_month"].sum()
        total_apps = len(filtered_df)/4
        avg_updates_month = total_updates/total_apps
        print(f"Total Updates = {total_updates}, total "
              f"apps = {total_apps} and average function updates {new_dict[i]} {app_cluster.split()[1]} "
              f"entry = {avg_updates_month}")


def plot_function_update_freq(app_cluster):
    # Plot average function updates before and after treatment for affected applications and save plot as png file
    if app_cluster.split()[0] != app_cluster.split()[1]:
        print("Invalid input data!")
        sys.exit(1)

    df = pd.read_csv("Merged Function Data/" + app_cluster + ".csv")
    plot_path = "Images/Function Plots/"
    avg_function_updates = []

    for j in range(1, 9):
        # Filter out all affected entries for a given month
        filtered_df_affected = df[(df["month_indicator"] == f"month_{j}")]
        total_updates_month_affected = filtered_df_affected["function_updates_per_month"].sum()
        total_apps_affected = len(filtered_df_affected)
        avg_function_updates.append(total_updates_month_affected / total_apps_affected)

    print(f"List of average updates of affected apps per month: {avg_function_updates}")
    plt.figure(figsize=(10, 6))
    x = list(range(1, 9))
    sns.lineplot(x=x, y=avg_function_updates, label=f"{app_cluster.split()[1]} Apps")
    plt.ylim(0, 1)
    x_min, x_max = plt.xlim()
    x_mid_point = (x_min + x_max) / 2
    plt.axvline(x=x_mid_point, color="red", linestyle="--", label=f"Apple {app_cluster.split()[1]} Entry")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title("Average Function Updates per Month during Observation Period")
    plt.legend()
    plt.savefig(os.path.join(plot_path, app_cluster + "_function_updates.png"), dpi=300)
    plt.show()


def print_avg_updates(app_cluster1, app_cluster2):
    # Print the average updates before or after treatment for treatment and control groups
    if app_cluster1 == app_cluster2 or app_cluster1.split()[1] != app_cluster2.split()[1]:
        print("Invalid input data!")
        sys.exit(1)
    unaffected_app_cluster = [s for s in (app_cluster1, app_cluster2) if s.split()[0] != s.split()[1]]
    df1 = pd.read_csv("Groups/Merged Data/" + app_cluster1 + ".csv")
    df2 = pd.read_csv("Groups/Merged Data/" + app_cluster2 + ".csv")
    df = pd.concat([df1, df2], ignore_index=True)
    new_dict = {0: "unaffected", 1: "affected"}
    new_dict2 = {0: "Before", 1: "After"}
    for i in range(0, 2):
        for j in range(0, 2):
            filtered_df = df[(df["affected"] == i) & (df["after_entry"] == j)]
            total_updates = filtered_df["updates_per_month"].sum()
            total_apps = len(filtered_df)/4
            avg_updates_month = total_updates/total_apps
            print(f"{new_dict2[j]} entry for {new_dict[i]} {unaffected_app_cluster[0].split()[i]} apps: "
                  f"Total Updates = {total_updates}, total "
                  f"apps = {total_apps} and average updates = {avg_updates_month}")


def plot_update_freq(app_cluster1, app_cluster2):
    # Plot average updates before and after treatment for both affected and unaffected apps and save plots as .png files
    if app_cluster1 == app_cluster2 or app_cluster1.split()[1] != app_cluster2.split()[1]:
        print("Invalid input data!")
        sys.exit(1)
    df1 = pd.read_csv("Groups/Merged Data/" + app_cluster1 + ".csv")
    df2 = pd.read_csv("Groups/Merged Data/" + app_cluster2 + ".csv")
    df = pd.concat([df1, df2], ignore_index=True)
    avg_updates_unaffected = []
    avg_updates_affected = []
    plot_path = "Images/Plots/"
    unaffected_app_cluster = [s for s in (app_cluster1, app_cluster2) if s.split()[0] != s.split()[1]]

    for j in range(1, 9):
        # Filter out all unaffected entries for a given month
        filtered_df_unaffected = df[(df["month_indicator"] == f"month_{j}") & (df["affected"] == 0)]
        total_updates_month_unaffected = filtered_df_unaffected["updates_per_month"].sum()
        total_apps_unaffected = len(filtered_df_unaffected)
        avg_updates_unaffected.append(total_updates_month_unaffected / total_apps_unaffected)

        # Filter out all affected entries for a given month
        filtered_df_affected = df[(df["month_indicator"] == f"month_{j}") & (df["affected"] == 1)]
        total_updates_month_affected = filtered_df_affected["updates_per_month"].sum()
        total_apps_affected = len(filtered_df_affected)
        avg_updates_affected.append(total_updates_month_affected / total_apps_affected)

    print(f"List of average updates of unaffected apps per month: {avg_updates_unaffected}")
    print(f"List of average updates of affected apps per month: {avg_updates_affected}")
    plt.figure(figsize=(10, 10))
    x = list(range(1, 9))
    sns.lineplot(x=x, y=avg_updates_unaffected, label=f"{unaffected_app_cluster[0].split()[0]} Apps")
    sns.lineplot(x=x, y=avg_updates_affected, label=f"{app_cluster1.split()[1]} Apps")
    plt.ylim(0, 2)
    x_min, x_max = plt.xlim()
    x_mid_point = (x_min + x_max) / 2
    plt.axvline(x=x_mid_point, color="red", linestyle="--", label=f"{app_cluster1.split()[1]} Entry")
    plt.xlabel("Month")
    plt.ylabel("Value")
    plt.title("Average Updates per Month during Observation Period")
    plt.legend()
    plt.savefig(os.path.join(plot_path, app_cluster1 + " " + app_cluster2 + ".png"), dpi=300)
    plt.show()


def describe_clusters(app_cluster1, app_cluster2):
    # Calculate descriptive statistics of each cluster and save them in text and png files
    if app_cluster1 == app_cluster2 or app_cluster1.split()[1] != app_cluster2.split()[1]:
        print("Invalid input data!")
        sys.exit(1)
    df1 = pd.read_csv("Groups/Merged Data/" + app_cluster1 + ".csv")
    df2 = pd.read_csv("Groups/Merged Data/" + app_cluster2 + ".csv")
    df1_new = df1[(df1["month_indicator"] == "month_1")]
    df2_new = df2[(df2["month_indicator"] == "month_1")]
    columns = ["app_size", "app_lang", "multihoming", "paid", "in_app_purchases", "updates_per_month",
               "elapsed_avg_days", "avg_age_days"]
    print(f"Description of {app_cluster1}:")
    print("\n")
    description1 = df1_new[columns].describe()
    print(description1)
    description1.to_csv("Descriptive Statistics/" + app_cluster1 + ".txt", sep="\t", float_format="%.2f")
    plt.figure(figsize=(14, 8))
    plt.table(cellText=description1.values,
              colLabels=description1.columns,
              rowLabels=description1.index,
              loc="center")
    plt.axis("off")
    plt.title(f"Descriptive Statistics of {app_cluster1.split()[0]} Applications "
              f"during Apple {app_cluster1.split()[1]} Observation Period")
    plt.savefig("Descriptive Statistics/" + app_cluster1 + ".png")
    plt.close()

    print("\n")
    print(f"Description of {app_cluster2}:")
    print("\n")
    description2 = df2_new[columns].describe()
    print(description2)
    description2.to_csv("Descriptive Statistics/" + app_cluster2 + ".txt", sep="\t", float_format="%.2f")
    plt.figure(figsize=(14, 8))
    plt.table(cellText=description2.values,
              colLabels=description2.columns,
              rowLabels=description2.index,
              loc="center")
    plt.axis("off")
    plt.title(f"Descriptive Statistics of {app_cluster2.split()[0]} Applications "
              f"during Apple {app_cluster2.split()[1]} Observation Period")
    plt.savefig("Descriptive Statistics/" + app_cluster2 + ".png")
    plt.close()


def analyze_reviews_file(file, threshold_unigram, threshold_ngram):
    # Create uni-, bi-, -tri and quad grams for app reviews according to the specified threshold
    stopword_set = set(stopwords.words("english"))
    df = pd.read_csv(file, skiprows=1, header=1)
    df["Title"] = df["Title"].astype(str)
    df["Review"] = df["Review"].astype(str)
    new_df = pd.DataFrame()

    # Remove stopwords, special characters and numbers from titles and reviews
    new_df["cleaned_titles"] = df["Title"].apply(lambda x: re.sub(r"[^\w\s]|\d", "", x))

    new_df["filtered_titles"] = new_df["cleaned_titles"].apply(lambda x: " ".join([item.lower() for item in x.split()
                                                                      if item.lower() not in stopword_set and
                                                                      item.lower() not in string.punctuation]))

    new_df["cleaned_reviews"] = df["Review"].apply(lambda x: re.sub(r"[^\w\s]|\d", "", x))

    new_df["filtered_reviews"] = new_df["cleaned_reviews"].apply(lambda x: " ".join([item.lower() for item in x.split()
                                                                                if item.lower() not in stopword_set and
                                                                                item.lower() not in string.punctuation]))

    # Tokenization
    tokenized_titles = [word_tokenize(text) for text in new_df["filtered_titles"]]
    tokenized_reviews = [word_tokenize(text) for text in new_df["filtered_reviews"]]

    word_counts_titles = Counter(word_tokenize(" ".join(new_df["filtered_titles"])))
    filtered_word_counts_titles = {word: count for word, count in word_counts_titles.items() if count >= threshold_unigram}

    word_counts_reviews = Counter(word_tokenize(" ".join(new_df["filtered_reviews"])))
    filtered_word_counts_reviews = {word: count for word, count in word_counts_reviews.items() if count >= threshold_unigram}

    # Write single word counter to csv and xlsx files
    counter_word_tokenize_titles_df = pd.DataFrame(filtered_word_counts_titles.items(), columns=["word", "frequency"])
    counter_word_tokenize_titles_df = counter_word_tokenize_titles_df.sort_values(by="frequency", ascending=False)
    counter_word_tokenize_titles_df.to_csv(file[:-4] + "_1-Gram_Titles.csv", index=False)
    counter_word_tokenize_titles_df.to_excel(file[:-4] + "_1-Gram_Titles.xlsx", index=False)

    counter_word_tokenize_reviews_df = pd.DataFrame(filtered_word_counts_reviews.items(), columns=["word", "frequency"])
    counter_word_tokenize_reviews_df = counter_word_tokenize_reviews_df.sort_values(by="frequency", ascending=False)
    counter_word_tokenize_reviews_df.to_csv(file[:-4] + "_1-Gram_Reviews.csv", index=False)
    counter_word_tokenize_reviews_df.to_excel(file[:-4] + "_1-Gram_Reviews.xlsx", index=False)

    for n in range(2, 5):
        if n == 2:
            collocation_finder = BigramCollocationFinder
        elif n == 3:
            collocation_finder = TrigramCollocationFinder
        else:
            collocation_finder = QuadgramCollocationFinder

        # Create ngrams for titles
        finder = collocation_finder.from_documents(tokenized_titles)
        finder.apply_freq_filter(threshold_ngram)
        title_ngram = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))

        # Create ngrams for reviews
        finder = collocation_finder.from_documents(tokenized_reviews)
        finder.apply_freq_filter(threshold_ngram)
        review_ngram = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))

        # Write ngrams to csv and xlsx files
        ngrams_title_df = pd.DataFrame(title_ngram, columns=["ngram", "frequency"])
        ngrams_title_df.to_csv(file[:-4] + "_" + str(n) + "-Gram_Title.csv", index=False)
        ngrams_title_df.to_excel(file[:-4] + "_" + str(n) + "-Gram_Title.xlsx", index=False)

        ngrams_review_df = pd.DataFrame(review_ngram, columns=["ngram", "frequency"])
        ngrams_review_df.to_csv(file[:-4] + "_" + str(n) + "-Gram_Review.csv", index=False)
        ngrams_review_df.to_excel(file[:-4] + "_" + str(n) + "-Gram_Review.xlsx", index=False)


def analyze_descriptions_file(file, threshold_unigram, threshold_ngram):
    # Count keywords in app-descriptions as n-grams
    stopword_set = set(stopwords.words("english"))
    df = pd.read_csv(file)
    # Remove stopwords, special characters and numbers from titles and reviews
    df["cleaned_descriptions"] = df["app_description"].apply(lambda x: re.sub(r"[^\w\s]|\d", "", str(x)))

    df["filtered_descriptions"] = (df["cleaned_descriptions"].
                                       apply(lambda x: " ".join([item.lower() for item in x.split() if item.lower()
                                                                 not in stopword_set and item.lower() not
                                                                 in string.punctuation])))

    # Tokenize descriptions
    tokenized_descriptions = [word_tokenize(text) for text in df["filtered_descriptions"]]

    word_counts_descriptions = Counter(word_tokenize(" ".join(df["filtered_descriptions"])))
    filtered_word_counts_descriptions = {word: count for word, count in word_counts_descriptions.items() if
                                         count >= threshold_unigram}

    # Write single word counter (unigram) to csv and xlsx file
    counter_word_tokenize_descriptions_df = pd.DataFrame(filtered_word_counts_descriptions.items(),
                                                         columns=["word", "frequency"])
    counter_word_tokenize_descriptions_df = counter_word_tokenize_descriptions_df.sort_values(by="frequency",
                                                                                              ascending=False)
    counter_word_tokenize_descriptions_df.to_csv(file[:-4] + "_1-Gram_Descriptions.csv", index=False)
    counter_word_tokenize_descriptions_df.to_excel(file[:-4] + "_1-Gram_Descriptions.xlsx", index=False)

    for n in range(2, 5):
        if n == 2:
            collocation_finder = BigramCollocationFinder
        elif n == 3:
            collocation_finder = TrigramCollocationFinder
        else:
            collocation_finder = QuadgramCollocationFinder

        # Create n-grams for descriptions
        finder = collocation_finder.from_documents(tokenized_descriptions)
        finder.apply_freq_filter(threshold_ngram)
        descriptions_ngram = sorted(finder.ngram_fd.items(), key=lambda t: (-t[1], t[0]))

        # Write n-grams to csv and xlsx files
        ngrams_descriptions_df = pd.DataFrame(descriptions_ngram, columns=["ngram", "frequency"])
        ngrams_descriptions_df.to_csv(file[:-4] + "_" + str(n) + "-Gram_Descriptions.csv", index=False)
        ngrams_descriptions_df.to_excel(file[:-4] + "_" + str(n) + "-Gram_Descriptions.xlsx", index=False)

    return df, tokenized_descriptions


def parse_txt_files(relative_path):
    # Parse text files and retrieve total displayed ratings for each app in the specified path
    rating_dict = {}
    txt_files = [os.path.join(relative_path, file) for file in os.listdir(relative_path) if file.endswith(".txt")]
    for txt_file in txt_files:
        with open(txt_file, "r", encoding="utf-8") as file:
            lines = file.readlines()
            for i, line in enumerate(lines):
                if "Total Displayed Ratings" in line:
                    ratings = lines[i + 1].strip()
                    if ratings.endswith("k"):
                        ratings = ratings[:-1]
                        ratings = float(ratings)*1000
                    elif ratings.endswith("m"):
                        ratings = ratings[:-1]
                        ratings = float(ratings)*1000000
                    elif ratings == "N/A":
                        ratings = 0
                    else:
                        ratings = float(ratings)
                    rating_dict[os.path.basename(txt_file)[:-4]] = ratings
                    break
    print({k: v for k, v in sorted(rating_dict.items(), key=lambda item: item[1])})


def check_update_freq(data):
    # Compare frequencies of updates between treatment and control groups
    # Filter update frequency for treatment group based on affected and after_entry columns
    update_freq_treatment = data[(data["affected"] == 1) & (data["after_entry"] == 0)]["update_frequency"]

    # Filter update frequency for control group based on affected and after_entry columns
    update_freq_control = data[(data["affected"] == 0) & (data["after_entry"] == 0)]["update_frequency"]

    # Calculate descriptive statistics to check for balanced data before treatment
    print("Update frequency for treatment group before treatment:")
    print(update_freq_treatment.describe())
    print("\nUpdate frequency for control group before treatment:")
    print(update_freq_control.describe())


def did_analysis(app_cluster1, app_cluster2):
    # Define DiD formula with control variables and time + app-fixed effects and perform DiD analysis
    if app_cluster1 == app_cluster2 or app_cluster1.split()[1] != app_cluster2.split()[1]:
        print("Invalid input data!")
        sys.exit(1)
    df1 = pd.read_csv("Groups/Merged Data/" + app_cluster1 + ".csv")
    df2 = pd.read_csv("Groups/Merged Data/" + app_cluster2 + ".csv")
    did_data = pd.concat([df1, df2], ignore_index=True)
    did_data["app_unique"] = did_data["app_id"].astype("category")
    did_data["month_unique"] = did_data["month_indicator"].astype("category")

    # Log transform skewed variables
    did_data["log_avg_age_days"] = np.log(did_data["avg_age_days"])
    did_data["log_updates_per_month"] = np.log(did_data["updates_per_month"] + 1).round(6)
    did_data["log_elapsed_avg_days"] = np.log(did_data["elapsed_avg_days"])
    did_data.to_csv(f"DiD Merged Data/{app_cluster1}_{app_cluster2}_DiD_control_fixed.csv", index=False)

    # Define Difference-in-Difference formula with control variables and fixed-effects
    formula = ("log_updates_per_month ~ affected * after_entry + log_elapsed_avg_days + "
               "C(app_unique) + C(month_unique)")

    # Fit the regression model with ordinary least square (OLS) method
    model = smf.ols(formula=formula, data=did_data).fit(cov_type="cluster", cov_kwds={"groups": did_data["app_id"]})
    summary_text = model.summary().as_text()
    # Filter out the fixed effects from the summary output
    lines = summary_text.split("\n")
    filtered_lines = [line for line in lines if "C(app_unique)" not in line and "C(month_unique)" not in line]
    filtered_summary_text = "\n".join(filtered_lines)
    print(filtered_summary_text)
    # Save results in a text file
    output_filename = f"DiD Results/{app_cluster1}_{app_cluster2}_DiD_control_fixed.txt"
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(filtered_summary_text)


def corr_matrix(data):
    # Create a correlation matrix
    df = pd.read_csv(data)
    correlation_matrix = df[["app_lang", "multihoming", "paid", "in_app_purchases", "affected", "after_entry"]].corr()
    print(correlation_matrix)


def plot_hist(path, column):
    did_data = pd.read_csv(path)
    # Create a boxplot for the distribution of updates_per_month for each app
    plt.figure(figsize=(10, 6))
    sns.histplot(did_data[column], bins=15, kde=False)
    plt.title("Distribution of Updates Per Month")
    plt.xlabel("Updates Per Month")
    plt.ylabel("Frequency")
    plt.show()


def plot_hist_log(path, column):
    did_data = pd.read_csv(path)
    did_data[f"log_{column}"] = np.log(did_data[column] + 1)
    # Create a boxplot for the distribution of updates_per_month for each app
    plt.figure(figsize=(10, 6))
    sns.histplot(did_data[f"log_{column}"], bins=15, kde=False)
    plt.title(f"Distribution of log {column}")
    plt.xlabel(f"{column}")
    plt.ylabel("Frequency")
    plt.show()


def full_analysis(app_cluster1, app_cluster2):
    # Run essential analysis functions
    describe_clusters(app_cluster1, app_cluster2)
    plot_update_freq(app_cluster1, app_cluster2)
    print_avg_updates(app_cluster1, app_cluster2)
    did_analysis(app_cluster1, app_cluster2)

