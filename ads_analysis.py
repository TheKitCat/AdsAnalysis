import pandas as pd
import nltk
import string
import re
import matplotlib.pyplot as plt
import logging as log

from datetime import datetime
from textblob import TextBlob
from textblob import Word
from math import log


#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

def remove_emoji(input):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', input)


def check_if_number(input):
    try:
        if input.isdigit() or input.isnumeric():
            return True
        val = int(input)
        return True

    except ValueError:
        return False


def load_data(path):
    # read necessary data from csv file
    data = pd.read_csv(path, sep=",", dtype=str, nrows=60, usecols=["political", "not_political", "title", "message", "created_at", "advertiser", "entities"])

    return data


def process_data_cleaning(data):
    # remove undesired unicode characters
    print("start data cleaning")

    data["message"] = data["message"].apply(lambda x: x.replace("\\/", "/").encode("ascii", "ignore").decode("ascii"))

    # remove html tags, urls etc.
    tag_cleaner = re.compile('<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});')
    url_cleaner = re.compile('http\S+|http\S+|www\S+')

    data["message"] = [re.sub(tag_cleaner, " ", x) for x in data["message"]]
    data["message"] = [re.sub(url_cleaner, " ", x) for x in data["message"]]

    # transform into lowercase
    data["message"] = data["message"].apply(lambda x: x.lower())

    # remove smileys, symbols and all the crap
    data["message"] = [remove_emoji(x) for x in data["message"]]

    # remove all numbers
    data["message"] = data["message"].apply(lambda x: " ".join(x for x in x.split() if check_if_number(x) is not True))

    # remove all special characters, which might be left
    spec = set(string.punctuation)
    data["message"] = data["message"].apply(lambda x: x.translate({ord(i): None for i in spec}))

    # remove stopwords - check on lowercase level
    stop = nltk.corpus.stopwords.words('english')
    data["message"] = data["message"].apply(lambda x: " ".join(x for x in x.split() if x not in stop))

    # remove rows which include empty messages
    empty_string_filter = data["message"] != ""
    data = data[empty_string_filter]

    print("end data cleaning")
    print("start spelling correction")
    # spelling correction
    data["message"] = data["message"].apply(lambda x: (str(TextBlob(x).correct())))
    print("end spelling correction")
    print("start lemmatisation")
    # perform lemmatisation
    data["message"] = data["message"].apply(lambda x: " ".join([Word(word).lemmatize() for word in x.split()]))
    print("start lemmatisation")

    return data


def advanced_data_cleaning(data):
    # cnt and show frequent words - remove afterwards
    freq = pd.Series(" ".join(data["message"]).split()).value_counts()[:15]
    freq = list(freq.index)
    data["message"] = data["message"].apply(lambda x: " ".join(x for x in x.split() if x not in freq))

    # cnt and show rare words - remove afterwards
    rare = pd.Series(" ".join(data["message"]).split()).value_counts()
    to_remove = []
    for val, cnt in rare.iteritems():
        if cnt <= 1:
            to_remove.append(val)

    data["message"] = data["message"].apply(lambda x: " ".join(x for x in x.split() if x not in to_remove))

    # write result into new file
    data.to_csv("preprocessed.csv", header=True)

    return data


def n_grams_analysis(path, n):
    data = pd.read_csv(path, sep=",", dtype=str, usecols=["political", "not_political", "title", "message", "created_at", "advertiser", "entities"])

    ngram_dict = {}

    for x in data["message"]:
        word_list = list(TextBlob(str(x)).ngrams(n))
        for word in word_list:
            composed_word = " ".join(word)
            if composed_word in ngram_dict:
                ngram_dict[composed_word] = ngram_dict[composed_word] + 1
            else:
                ngram_dict[composed_word] = 1

    return pd.DataFrame.from_dict(ngram_dict, orient='index')


def determine_composed_word_frequency(df, n):
    word_series = df.reset_index()

    # calculate TF for each word in the document
    tf_data = word_series.rename(columns={"index": "word", 0: "count"})

    total_word_cnt = tf_data["count"].sum()

    tf_data["relative_word_frequency"] = tf_data["count"].apply(lambda x: int(x) / total_word_cnt)
    tf_data["tf_frequency"] = tf_data["count"].apply(lambda x: 1 + log(int(x)))
    tf_data["idf_frequency"] = tf_data["count"].apply(lambda x: log(len(tf_data) / int(x)))
    tf_data["tf x idf"] = tf_data.apply(lambda row: row["tf_frequency"] * row["idf_frequency"], axis=1)

    # write result into csv file
    tf_data.to_csv(str(n)+"_term_frequency.csv", header=True)


def plot_tf_frequency(path, nrows, n, size_x, size_y):
    df = pd.read_csv(path, sep=",", dtype=str, usecols=["word", "tf_frequency"])
    df = df.sort_values(by="tf_frequency", ascending=False)
    df = df.head(nrows)

    word_list = df["word"].tolist()
    frequency_list = df["tf_frequency"].tolist()
    frequency_list = [float(i) for i in frequency_list]
    frequency = [round(x, 2) for x in frequency_list]

    plt.figure(figsize=(30, 20))
    plt.bar(word_list, frequency)
    plt.ylim(frequency[len(frequency)-1]-size_x, frequency[0]+size_y)
    plt.ylabel("words")
    plt.ylabel("tf_frequency")
    plt.title("Top " + str(nrows) + " Word Frequency")
    plt.xticks(rotation=90)

    plt.savefig(str(n)+'_tf_frequency.png')


def plot_idf_frequency(path, nrows, n, size_x, size_y):
    df = pd.read_csv(path, sep=",", dtype=str, usecols=["word", "idf_frequency"])
    df = df.sort_values(by="idf_frequency", ascending=False)
    df = df.head(nrows)

    word_list = df["word"].tolist()
    frequency_list = df["idf_frequency"].tolist()
    frequency_list = [float(i) for i in frequency_list]
    frequency = [round(x, 4) for x in frequency_list]

    plt.figure(figsize=(30, 20))
    plt.bar(word_list, frequency)
    plt.ylim(frequency_list[len(frequency)-1]-size_x, frequency[0]+size_y)
    plt.ylabel("words")
    plt.ylabel("idf_frequency")
    plt.title("IDF " + str(nrows) + " Word Frequency")
    plt.xticks(rotation=90)

    plt.savefig(str(n) + '_idf_word_frequency.png')


def plot_tf_idf_frequency(path, nrows, n, size_x, size_y):
    df = pd.read_csv(path, sep=",", dtype=str, usecols=["word", "tf x idf"])
    df = df.sort_values(by="tf x idf", ascending=False)
    df = df.head(nrows)

    word_list = df["word"].tolist()
    frequency_list = df["tf x idf"].tolist()
    frequency_list = [float(i) for i in frequency_list]
    frequency = [round(x, 4) for x in frequency_list]

    plt.figure(figsize=(30, 20))
    plt.bar(word_list, frequency)
    plt.ylim(frequency_list[len(frequency)-1]-size_x, frequency[0]+size_y)
    plt.ylabel("words")
    plt.ylabel("tf_x_idf_frequency")
    plt.title("Overall " + str(nrows) + " Word Frequency")
    plt.xticks(rotation=90)

    plt.savefig(str(n)+'_tf_x_idf_word_frequency.png')


def calculate_term_frequency(data):
    # group words and cnt and their occurrence
    word_series = pd.Series(" ".join(data["message"]).split()).value_counts()
    word_series = word_series.reset_index()

    # calculate TF for each word in the document
    tf_data = word_series.rename(columns={"index": "word", 0: "count"})
    total_word_cnt = tf_data["count"].sum()

    tf_data["relative_word_frequency"] = tf_data["count"].apply(lambda x: int(x) / total_word_cnt)
    tf_data["tf_frequency"] = tf_data["count"].apply(lambda x: 1 + log(int(x)))
    tf_data["idf_frequency"] = tf_data["count"].apply(lambda x: log(len(tf_data)/int(x)))

    tf_data["tf x idf"] = tf_data.apply(lambda row: row["tf_frequency"] * row["idf_frequency"], axis=1)
    tf_data = tf_data.sort_values(by="tf x idf", ascending=False)

    # write result into csv file
    tf_data.to_csv("1_term_frequency.csv", header=True)


def __main__():
    print("start data import at ", datetime.now())
    # raw_data = load_data(nrows=5000)
    print("end data import at ", datetime.now())

    print("start data cleaning at ", datetime.now())

    # pre_processed_data = process_data_cleaning(data=raw_data)
    print("end data cleaning at ", datetime.now())

    print("start plotting one word frequency at ", datetime.now())
    # calculate_term_frequency(pre_processed_data)
    plot_tf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.02, size_y=0.02)
    plot_idf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.002, size_y=0.002)
    plot_tf_idf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.2, size_y=0.2)
    print("end plotting one word frequency at ", datetime.now())

    print("start building ngrams at", datetime.now())
    result = n_grams_analysis(path="./preprocessed.csv", n=2)

    determine_composed_word_frequency(df=result, n=2)
    plot_tf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.02, size_y=0.02)
    plot_idf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.02, size_y=0.02)
    plot_tf_idf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.2, size_y=0.2)

    result = n_grams_analysis(path="./preprocessed.csv", n=3)
    determine_composed_word_frequency(df=result, n=3)
    plot_tf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.02, size_y=0.02)
    plot_idf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.02, size_y=0.02)
    plot_tf_idf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.2, size_y=0.2)

    print("end building ngrams at ", datetime.now())
