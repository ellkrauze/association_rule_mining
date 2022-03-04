import pandas as pd
import numpy as np
import os
import csv
import argparse
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
import timeit
from memory_profiler import memory_usage

# Map for converting data to 0 and 1
# @x is element of map to be converted
def encode_data(x):
    if x<=0:
        return 0
    if x>=1:
        return 1

# Load and convert datasets to format 
# which is needed for Apriori algorithm
def load_data():
    ratings_df = pd.read_csv(os.path.join('datasets','ratings_small.csv'))
    movies_df = pd.read_csv(os.path.join('datasets','movies_metadata.csv'))

    # Clean data
    title_mask = movies_df['title'].isna()
    movies_df = movies_df.loc[title_mask == False]

    # Convert the string datatype of id column 
    # of movies dataframe to int as that in the 
    # ratings dataframe
    movies_df = movies_df.astype({'id': 'int64'})

    # Merge movies and ratings dataframes
    df = pd.merge(ratings_df, movies_df[['id', 'title']], left_on='movieId', right_on='id')

    # Id column is repeated and the timestamp is 
    # not important for this problem.
    # Drop the two
    df.drop(['timestamp', 'id'], axis=1, inplace=True)

    # Make sure there are no duplicate records 
    # for the combination of userId and title
    df = df.drop_duplicates(['userId','title'])

    # The apriori model needs data in a format 
    # such that the userId forms the index
    df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0)

    # You need to convert the ratings to 0 or 1 
    # and also convert all float values to int
    df_pivot = df_pivot.astype('int64')
    df_pivot = df_pivot.applymap(encode_data)
    return df_pivot

# Read csv file as list of strings
# @path is directory of file
# @sep is csv separator (default=',')
def load_csv(path, sep=","):
    data = []
    with open(os.path.join(path), 'r') as file:
        reader = csv.reader(file, delimiter=sep)
        data = list(reader)
    # Remove empty strings from list
    for row in data:
        while('' in row):
            row.remove('')
    return data

# Transform the input dataset into a one-hot encoded NumPy boolean array
# @data is input dataset in list-format
def transform_onehot(data):
    te = TransactionEncoder()
    data = pd.DataFrame(te.fit(data).transform(data), columns=te.columns_)
    return data

# Get frequent itemsets from single dataset 
# with Apriori algorithm
#
# @data is input dataset
# @minsup is minimal support for 
# @mode is optional way to order the resulting list of itemsets
## @profile
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
    # Parse arguments from command line
    parser = argparse.ArgumentParser(description='Get frequent itemsets from single dataset file.')
    parser.add_argument('-p', type = str, default ='', help = 'An required destination \
        path to read dataset.')
    parser.add_argument('-minsup', type = float, default = 0.3, help = 'An optional min support. Default: 0.3.')
    parser.add_argument('-mode', type = str, default = "None", help = 'An optional a way to order the resulting list of sets. Default: None.')
    
    args = parser.parse_args()

    path = args.p
    minsup = args.minsup
    mode = args.mode
    
    # Check if given min support value is in interval [0, 1]
    if not ((0 <= minsup) & (minsup <= 1)):
        parser.error("Min support value must be in interval [0, 1]")
    
    print('[INFO] Running get_freq_itemsets.py script with these parameters: \
        \n\tpath = {path},\n\tminsup = {minsup},\n\tmode = {mode}.'.format(path = path, 
        minsup = minsup, mode = mode))

    df = load_data()
    df = transform_onehot(df)
    frequent_itemset = find_frequent_itemset(df, minsup, mode)

    # Get average execution time of function
    # num_runs = 5
    # duration = timeit.Timer("find_frequent_itemset(df, minsup)", globals=locals()).timeit(number = num_runs)
    # avg_duration = duration/num_runs

    # print(frequent_itemset)
    # print(find_frequent_itemset(df, minsup, mode).size)
    # print(f'On average it took {avg_duration} seconds')
    