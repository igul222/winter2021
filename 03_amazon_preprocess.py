"""
Preprocess the Amazon reviews dataset into a big text file with all reviews
concatenated. You can download the dataset from:
http://snap.stanford.edu/data/amazon/productGraph/kcore_5.json.gz
"""

import os
import json

OUTPUT_PATH = os.path.expanduser('~/data/amazon_reviews/all_reviews.txt')
INPUT_PATH = os.path.expanduser('~/data/amazon_reviews/kcore_5.json')

with open(OUTPUT_PATH, 'w') as f_out:
    with open(INPUT_PATH, 'r') as f_in:
        for i, line in enumerate(f_in):
            text = json.loads(line[:-1])['reviewText'].replace("\n"," ")
            f_out.write(text + "\n")
            if i % 100000 == 0:
                print(f'Processed {i+1} records (of approx. 41M)...')

print('Done!')