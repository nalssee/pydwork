""" Web scraping framework using selenium
Most of the time I had to use selenium (with firefox or phantomjs)
not 'requests'
"""

import os
import logging
import pickle
import pandas as pd
from datetime import datetime
from pydwork import cnp


__all__ = ['fetch_items', 'result_files_to_df', 'RESULT_FILE_PREFIX',
           'load_items_dict', 'show_failed_items',
           'LOG_FILE', 'PICKLE_FILE']


RESULT_FILE_PREFIX = 'result'
LOG_FILE = 'logfile.txt'
PICKLE_FILE = 'items_dict.pkl'


def fetch_items(drivers, items_dict, fetchfn,
                max_items=1000000, save_every_n=100, max_trials=3):
    """
    drivers: A list of selenium web drivers
    items_dict: to-fetch-items dictionary.
                values are number of trials.
    fetch_fn: driver, item(a key of items_dict) -> pd.DataFrame
    max_trials: If fetching fails more than max_trials just skip it
    Others are obvious.

    search for files like RESULT_FILE_PREFIX +
    "2015-06-17 10:04:54.560384.csv" files
    """
    # logging setup
    logging.basicConfig(filename=LOG_FILE,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler())

    def dict_to_list(items_dict):
        result = []
        count = 0
        for item, ntrial in items_dict.items():
            if count >= max_items:
                return result
            if ntrial >= 0 and ntrial <= max_trials:
                result.append((item, ntrial))
                count += 1
        return result

    items = dict_to_list(items_dict)
    logging.info('{} items to fetch'.format(min(len(items), max_items)))

    failure_string = cnp.random_string(20)

    # counts how many items have been fetched
    count = 0
    producers = []
    results = []

    def make_producer(driver, items):
        for item, ntrials in items:
            try:
                df = fetchfn(driver, item)
                yield "", item, df
            except:
                yield failure_string, item, ()
        driver.close()

    def consumer(result1):
        nonlocal results, count
        status, item, df = result1
        if status == failure_string:
            logging.warning("Failed To Fetch: " + str(item))
            items_dict[item] += 1
        else:
            items_dict[item] = -1
            results.append(df)
            if len(results)  == save_every_n:
                count += save_every_n
                logging.info('Fetched ' + str(count) + ' items')
                save_results()
                results = []

    def save_results():
        rfile = RESULT_FILE_PREFIX + str(datetime.now()) + '.csv'
        pd.concat(results).to_csv(rfile, index=False)

    for driver, chunk in zip(drivers, cnp.nchunks(items, len(drivers))):
        producers.append(make_producer(driver, chunk))

    cnp.npc(producers, consumer)

    if results:
        save_results()

    # save items_dict for later in case you haven't finished fetching
    # and want to do it later.
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(items_dict, f)


def load_items_dict():
    with open(PICKLE_FILE, 'rb') as fport:
        items_dict = pickle.load(fport)
    print('total: ', len(items_dict))
    print('failed: ', len([_ for _, v in items_dict.items() if v != -1]))
    print('succeeded: ', len([_ for _, v in items_dict.items() if v == -1]))
    return items_dict


def show_failed_items(items_dict):
    for k, v in items_dict.items():
        if v != -1:
            print(k, v)


def result_files_to_df(path=os.getcwd()):
    result_dfs = []
    for rfile in os.listdir(path):
        if rfile.startswith(RESULT_FILE_PREFIX) and rfile.endswith('.csv'):
            result_dfs.append(pd.read_csv(rfile))
    return pd.concat(result_dfs)
