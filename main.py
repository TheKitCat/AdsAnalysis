import ads_analysis as analysis
import pandas as pd
import logging as log

from multiprocessing import Process, Queue
from datetime import datetime


def data_cleaning(path, q):
    df = analysis.load_data(path)
    cleaned = analysis.process_data_cleaning(data=df)
    q.put(cleaned)


if __name__ == '__main__':
    cleaned_data_results = Queue()
    processes = []

    log.info("start creating processes at ", datetime.now())

    p1 = Process(target=data_cleaning, args=("./0_raw_data.csv", cleaned_data_results))
    p2 = Process(target=data_cleaning, args=("./1_raw_data.csv", cleaned_data_results))

    processes.extend([p1, p2])

    log.info("start processes at ", datetime.now())

    for p in processes:
        p.start()

    log.info("start join processes at ", datetime.now())

    for p in processes:
        p.join()
    log.info("end join processes at ", datetime.now())

    log.info("start concatenating data frames at ", datetime.now())

    # join data frame, save it and calculate term frequency
    cleaned_df = pd.DataFrame()

    while not cleaned_data_results.empty():
        cleaned_df = cleaned_df.append(cleaned_data_results.get(), ignore_index=True)

    log.info("end concatenating data frames at ", datetime.now())
    print(len(cleaned_df))

    log.info("start calculating word frequency at ", datetime.now())
    analysis.calculate_term_frequency(cleaned_df)
    log.info("end calculating word frequency at ", datetime.now())

    analysis.plot_tf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.02, size_y=0.02)
    analysis.plot_idf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.02, size_y=0.02)
    analysis.plot_tf_idf_frequency(path="./1_term_frequency.csv", nrows=30, n=1, size_x=0.2, size_y=0.2)

    log.info("end plotting one word frequency at ", datetime.now())

    log.info("start building n-grams at", datetime.now())
    result = analysis.n_grams_analysis(path="./preprocessed.csv", n=2)

    analysis.determine_composed_word_frequency(df=result, n=2)
    analysis.plot_tf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.02, size_y=0.02)
    analysis.plot_idf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.02, size_y=0.02)
    analysis.plot_tf_idf_frequency(path="./2_term_frequency.csv", nrows=30, n=2, size_x=0.2, size_y=0.2)

    result = analysis.n_grams_analysis(path="./preprocessed.csv", n=3)
    analysis.determine_composed_word_frequency(df=result, n=3)
    analysis.plot_tf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.02, size_y=0.02)
    analysis.plot_idf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.02, size_y=0.02)
    analysis.plot_tf_idf_frequency(path="./3_term_frequency.csv", nrows=30, n=3, size_x=0.2, size_y=0.2)

    log.info("end building n-grams at ", datetime.now())

    log.info("end program at ", datetime.now())