""" Web scraping framework using selenium
Most of the time I had to use selenium (with firefox or phantomjs)
not 'requests'
"""


import os
import pickle
import pandas as pd
from . import npc
from datetime import datetime


__all__ = ['fetch_items', 'result_files_to_df']


RESULT_FILE_PREFIX = 'result'


# This function should never raise an error
# unless you ctrl-c
def fetch_items(drivers, items, fetchfn,
                max_items=1000000, save_every_n=100, max_trials=3,
                base_dir=os.getcwd(), reset_dir=False):
    """
    ===================================
    READ CAREFULLY!!!
    ===================================

    drivers: A list of selenium web drivers
    items: list of strings to fetch
       (if reset_dir is False, items is simply ignored and items_dict picke file
       is used for fetching items)
    fetch_fn: driver, item -> pd.DataFrame
    max_trials: If fetching fails more than max_trials just skip it
    max_items: you can pass lots of items and just part of them are fetched
                so you can do the work later
    save_every_n: save every n items
    base_dir: folder to save results and an items_dict pickle file
    reset_dir: if True, remove results and and items_dict pickle file
         and items_dict pickle file is initiated with items

    search for files like RESULT_FILE_PREFIX +
    "2015-06-17 10:04:54.560384.csv" files
    """

    PICKLE_FILE = os.path.join(base_dir, 'items_dict.pkl')

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

    def load_items_dict():
        with open(PICKLE_FILE, 'rb') as fport:
            items_dict = pickle.load(fport)
        print("Loading items_dict.pkl file ...")
        print('total: %d' % len(items_dict))
        print('failed: %d' % len([_ for _, v in items_dict.items() if v != -1]))
        print('succeeded: %d' % len([_ for _, v in items_dict.items() if v == -1]))
        return items_dict


    if reset_dir:
        # remove all files related to reset them all
        for rfile in os.listdir(base_dir):
            if rfile.startswith(RESULT_FILE_PREFIX):
                os.remove(rfile)
        try:
            os.remove(PICKLE_FILE)
        except:
            pass


    if os.path.isfile(PICKLE_FILE):
        items_dict = load_items_dict()
    else:
        # initiate items_dict with items
        items_dict = {}
        for item in items:
            items_dict[item] = 0

    # it is a bit silly to turn a list to dict and back to a list again
    # but to keep consistency with other cases and to keep it simple
    # I will just leave it as is
    items = dict_to_list(items_dict)

    print('{} items to fetch'.format(min(len(items), max_items)))

    failure_string = npc.random_string(20)

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
            print("Failed To Fetch: " + str(item))
            items_dict[item] += 1
        else:
            items_dict[item] = -1
            results.append(df)
            if len(results)  == save_every_n:
                count += save_every_n
                print('Fetched ' + str(count) + ' items')
                save_results()
                results = []

    def save_results():
        rfile = os.path.join(base_dir, RESULT_FILE_PREFIX + str(datetime.now()) + '.csv')
        pd.concat(results).to_csv(rfile, index=False)

    for driver, chunk in zip(drivers, npc.nchunks(items, len(drivers))):
        producers.append(make_producer(driver, chunk))

    npc.npc(producers, consumer)

    if results:
        save_results()

    # Show failed items
    print("Failed items to fetch")
    max_failed_items = 20
    for k, v in items_dict.items():
        if v != -1:
            max_failed_items -= 1
            print(k)
        if max_failed_items < 0:
            print("More than %d items failed" % max_failed_items)
            break


    # save items_dict for later in case you haven't finished fetching
    # and want to do it later.
    with open(PICKLE_FILE, 'wb') as f:
        pickle.dump(items_dict, f)

    print("Done fetching")


def result_files_to_df(path=os.getcwd()):
    result_dfs = []
    for rfile in os.listdir(path):
        if rfile.startswith(RESULT_FILE_PREFIX) and rfile.endswith('.csv'):
            result_dfs.append(pd.read_csv(os.path.join(path, rfile)))

    return pd.concat(result_dfs)
