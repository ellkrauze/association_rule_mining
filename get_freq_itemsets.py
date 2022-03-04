import pandas as pd
import numpy as np
import os
import csv
import argparse
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import timeit
from memory_profiler import memory_usage

def encode_ratings(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

def load_data():
    ratings_df = pd.read_csv(os.path.join('datasets','ratings_small.csv'))
    movies_df = pd.read_csv(os.path.join('datasets','movies_metadata.csv'))
    title_mask = movies_df['title'].isna()
    movies_df = movies_df.loc[title_mask == False]
    movies_df = movies_df.astype({'id': 'int64'})
    df = pd.merge(ratings_df, movies_df[['id', 'title']], left_on='movieId', right_on='id')
    df.drop(['timestamp', 'id'], axis=1, inplace=True)
    df = df.drop_duplicates(['userId','title'])
    df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0)
    df_pivot = df_pivot.astype('int64')
    df_pivot = df_pivot.applymap(encode_ratings)
    return df_pivot


def load_csv(path):
    data = []
    with open(os.path.join(path), 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        data = list(reader)
        # data = list(filter(None, data))

    for row in data:
        while('' in row):
            row.remove('')

    # Transform the input dataset into a one-hot encoded NumPy boolean array
    te = TransactionEncoder()
    data = pd.DataFrame(te.fit(data).transform(data), columns=te.columns_)

    return data

# @profile
def find_frequent_itemset(data, minsup, mode="None"):
    # by passing minsup we mean the number of times in total number
    # of transactions the item should be present
    frequent_itemsets = apriori(data, min_support=minsup, use_colnames=True)

    match mode:
        case "descending-support":
            return frequent_itemsets.sort_values(by=['support'], ascending=False)
        case "itemsets":
            return frequent_itemsets.sort_values(by=['itemsets'], ascending=True)
        case _:
            return frequent_itemsets


if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Get frequent itemsets from single dataset file.')
    parser.add_argument('-p', type = str, default ='', help = 'An required destination \
        path to read dataset.')
    parser.add_argument('-minsup', type = float, default = 0.3, help = 'An optional min support. Default: 0.3.')
    parser.add_argument('-mode', type = str, default = "None", help = 'An optional a way to order the resulting list of sets. Default: None.')
    
    args = parser.parse_args()

    path = args.p
    minsup = args.minsup
    mode = args.mode
    
    if not ((0 <= minsup) & (minsup <= 1)):
        parser.error("Min support value must be in interval [0, 1]")
    
    print('[INFO] Running get_freq_itemsets.py script with these parameters: \
        \n\tpath = {path},\n\tminsup = {minsup},\n\tmode = {mode}.'.format(path = path, 
        minsup = minsup, mode = mode))

    df = load_data()
    frequent_itemset = find_frequent_itemset(df, minsup, mode)
    # print(frequent_itemset)
    # num_runs = 5
    # duration = timeit.Timer("find_frequent_itemset(df, minsup)", globals=locals()).timeit(number = num_runs)
    # avg_duration = duration/num_runs

    # print(find_frequent_itemset(df, minsup, mode).size)

    # print(f'On average it took {avg_duration} seconds')
    