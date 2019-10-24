import pandas as pd


def split_document():
    df = pd.read_csv("./raw_ads_US.csv", sep=",", dtype=str, usecols=["political", "not_political", "title", "message", "created_at", "advertiser",
                                "entities"])
    size = 32920
    list_of_dfs = [df.loc[i:i + size - 1, :] for i in range(0, len(df), size)]
    thread_no = 0

    for dataframe in list_of_dfs:
        dataframe.to_csv(str(thread_no)+"_raw_data.csv", header=True)
        thread_no = thread_no + 1

    return True


split_document()
