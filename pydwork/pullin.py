"""Web scraping tool

It is highly unlikely to download 100,000 pages without a single failure
So you have to split them into small pieces and save each of them
as soon as it's done.

And of course you have to record failures and successes
so you can try again later.
If you fail in more than say 10 times, probably the page is broken,
which means you also have to record the number of trials


Most of the time I had to use selenium (with firefox or phantomjs).
But you can also use 'requests'

Printing the process out to stdout is better than log files because
web scraping can't be fully automatic.
You'll learn about the sites you want to scrap through experience.
"""

import os
import pickle
import pandas as pd

from datetime import datetime
from collections import OrderedDict
from . import npc

__all__ = ['fetch_items', 'result_files_to_df']


RESULT_FILE_PREFIX = 'result'

print(os.getcwd())

def fetch_items(drivers, items, fetchfn,
                max_items=1000000, save_every_n=100, max_trials=10,
                base_dir=os.getcwd(), reset_dir=False):
    """
    ===================================
    READ CAREFULLY!!!
    ===================================

    drivers: A list of selenium web drivers
    items: list of strings(of course it can be JSON) to fetch
       (if reset_dir is False, items is simply ignored and items_dict
       picke file is used for fetching items)
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
        show_items_dict(items_dict)
        return items_dict

    def show_items_dict(items_dict):
        print('total: %d' % len(items_dict))
        print('succeeded: %d'
              % len([_ for _, v in items_dict.items() if v == -1]))
        print('unfetched: %d'
              % len([_ for _, v in items_dict.items() if v == 0]))
        print('failed: %d' % len([_ for _, v in items_dict.items() if v > 0]))

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
        items_dict = OrderedDict()
        for item in items:
            # if duplication is detected, something must have gone wrong
            items_dict[item] = 0
        print("Given items size: ", len(items))
        print("After duplication removal: ", len(items_dict))

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
    succeeded_items = []

    def make_producer(driver, items):
        subsequent_failures = 0
        for item, ntrials in items:
            try:
                df = fetchfn(driver, item)
                subsequent_failures = 0
                yield "", item, df
            except Exception as e:
                print(e)
                subsequent_failures += 1
                # must be network error or something
                if subsequent_failures > 10:
                    print("Too many subsequent failures..")
                    break
                yield failure_string, item, ()
        print("Closing the driver...")
        try:
            # you may just want to just reqeusts
            driver.close()
        except:
            pass

    def consumer(result1):
        nonlocal count
        status, item, df = result1
        if status == failure_string:
            print("Failed To Fetch: " + str(item))
            items_dict[item] += 1
        else:
            results.append(df)
            succeeded_items.append(item)
            if len(results) == save_every_n:
                count += save_every_n
                print('Fetched ' + str(count) + ' items')
                save_results()

    def save_results():
        nonlocal results, succeeded_items
        rfile = os.path.join(base_dir,
                             RESULT_FILE_PREFIX + str(datetime.now()) + '.csv')
        pd.concat(results).to_csv(rfile, index=False)
        for succeeded_item in succeeded_items:
            # Only when it's successfully saved, you can say that it's FETCHED.
            items_dict[succeeded_item] = -1
        succeeded_items = []
        results = []

    for driver, chunk in zip(drivers, npc.nchunks(items, len(drivers))):
        producers.append(make_producer(driver, chunk))

    # TODO: not working as intended, No idea what's wrong
    # Don't ever ctrl-c while running
    try:
        npc.npc(producers, consumer)
    finally:
        if results:
            save_results()

        # Show failed items
        print("Items failed to fetch:")
        max_failed_items = 0
        for k, v in items_dict.items():
            if v > 0:
                max_failed_items += 1
                print(k)
            if max_failed_items >= 20:
                print("More than %d items failed" % max_failed_items)
                break

        # save items_dict for later in case you haven't finished fetching
        # and want to do it later.
        with open(PICKLE_FILE, 'wb') as f:
            pickle.dump(items_dict, f)
        print("Fetching status")
        show_items_dict(items_dict)
        print("Done fetching")


def result_files_to_df(path=os.getcwd()):
    result_dfs = []
    for rfile in os.listdir(path):
        if rfile.startswith(RESULT_FILE_PREFIX) and rfile.endswith('.csv'):
            print(rfile)
            result_dfs.append(pd.read_csv(os.path.join(path, rfile)))

    return pd.concat(result_dfs)
