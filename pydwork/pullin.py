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
import re

from datetime import datetime
from . import npc


__all__ = ['fetch', 'result_files_to_df']


RESULT_FILE_PREFIX = 'result'
RESULTS_DIR = os.path.join(os.getcwd(), 'results')
if not os.path.isdir(RESULTS_DIR):
    print('Created', RESULTS_DIR)
    os.makedirs(RESULTS_DIR)


def fetch(drivers, items_list, fetch1,
          max_items=1000000, save_every_n=100, max_trials=10):
    """
    Args:
        drivers (Type): A list of selenium web drivers, or requests.get
        items_list (List[str]): list of strings(of course it can be JSON)
            to fetch if there is a 'items' file in the RESULTS_DIR folder
            the list will be merged to it

            'items' file is a dumped pickle file of a dictionary as followes:
                {'item1': -1, 'item2': 0, 'item3': 2}
                Each number means the number of trials of fetching
                -1 represents success

        fetch1 (FN[driver, item (str) -> pd.DataFrame]): fetch just one
            item and returns a data frame. An item may contain a list of
            results you want
        max_trials (int): If fetching fails more than max_trials just skip it
        max_items (int): you can pass lots of items and just part of them
            are fetched so you can do the work later
        save_every_n: save every n items
    """

    ITEMS_FILE = os.path.join(RESULTS_DIR, 'items')

    def load_items_file():
        with open(ITEMS_FILE, 'rb') as fport:
            items_dict = pickle.load(fport)
        return items_dict

    def dict_to_csv(items_dict):
        with open(os.path.join(RESULTS_DIR, 'items.csv'), 'w') as f:
            f.write('item,trials\n')
            for i, n in [x for x in sorted(items_dict.items(),
                         key=lambda x: x[1], reverse=True)]:
                f.write('%s,%s\n' % (i, n))

    def show_items_dict(items_dict):
        print('total: %d' % len(items_dict))
        print('succeeded: %d'
              % len([_ for _, v in items_dict.items() if v == -1]))
        print('unfetched: %d'
              % len([_ for _, v in items_dict.items() if v == 0]))
        print('failed: %d' % len([_ for _, v in items_dict.items() if v > 0]))

    items_dict = load_items_file() if os.path.isfile(ITEMS_FILE) else {}

    # merge items list to items_dict
    for item in items_list:
        if item not in items_dict:
            items_dict[item] = 0

    show_items_dict(items_dict)

    # filter items to fetch
    items_to_fetch_list = [i for i, n in sorted(list(items_dict.items()),
                           key=lambda x: x[1]) if n >= 0][:max_items]

    print('%d items to fetch' % len(items_to_fetch_list))

    failure_string = npc.random_string(20)

    # counts how many items have been fetched
    count = 0
    producers = []
    results = []
    succeeded_items = []

    def make_producer(driver, items):
        subsequent_failures = 0
        for item in items:
            try:
                df = fetch1(driver, item)
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
            # In case you use requests.get, it doesn't have 'close'
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
        rfile = os.path.join(RESULTS_DIR, _gen_result_file_name())

        pd.concat(results).to_csv(rfile, index=False)
        for succeeded_item in succeeded_items:
            items_dict[succeeded_item] = -1
        # empty the bins
        succeeded_items = []
        results = []

    for driver, chunk in zip(drivers,
                             npc.nchunks(items_to_fetch_list, len(drivers))):
        producers.append(make_producer(driver, chunk))

    # TODO: not working as intended, No idea what's wrong
    # Don't ever ctrl-c while running
    try:
        npc.npc(producers, consumer)
    finally:
        if results:
            save_results()

        dict_to_csv(items_dict)

        # save items_dict for later in case you haven't finished fetching
        # and want to do it later.
        with open(ITEMS_FILE, 'wb') as f:
            pickle.dump(items_dict, f)
        print("Fetching status")
        show_items_dict(items_dict)
        print("Done fetching")


def result_files_to_df():
    result_dfs = []
    for rfile in os.listdir(RESULTS_DIR):
        if rfile.startswith(RESULT_FILE_PREFIX) and rfile.endswith('.csv'):
            result_dfs.append(pd.read_csv(os.path.join(RESULTS_DIR, rfile)))
    return pd.concat(result_dfs)


def _gen_result_file_name():
    """Generate a result file name(csv)

    Returns:
        str
    """
    return RESULT_FILE_PREFIX + \
        re.sub(r'[^\w]+', '-', str(datetime.now())) + \
        '.csv'
