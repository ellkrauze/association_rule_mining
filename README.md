# Frequent Itemsets

```
$ python3.10 -m get_freq_itemsets -h
usage: get_freq_itemsets.py [-h] [-p P] [-minsup MINSUP] [-mode MODE]

Get frequent itemsets from single dataset file.

options:
  -h, --help      show this help message and exit
  -p P            An required destination path to read dataset.
  -minsup MINSUP  An optional min support. Default: 0.3.
  -mode MODE      An optional a way to order the resulting list of sets. Default: None.
```