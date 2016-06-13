import os, sys

TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from pydwork.pullin import *
from selenium import webdriver
import unittest
import time
import pandas as pd


class PullInTest(unittest.TestCase):
    # no setUp and tearDown

    def test_seeking_alpha(self):
        def init_drivers(n, client):
            drivers = []
            for _ in range(n):
                drivers.append(client())
            return drivers

        def fetchfn(driver, item):
            driver.get(item)
            box = driver.find_element_by_css_selector('ul.stripes_list')
            links = box.find_elements_by_css_selector("li > div > a")
            return pd.DataFrame({'article_addr': [link.get_attribute("href") for link in links]})

        drivers = init_drivers(2, webdriver.PhantomJS)

        base_addr = "http://seekingalpha.com/articles?page="

        items = [base_addr + str(i) for i in range(1, 9)]

        # Normally reset_dir should be False
        fetch_items(drivers, items, fetchfn, save_every_n=2, reset_dir=True)
        for d in drivers:
            d.quit()

        count = 0
        while len(result_files_to_df().index) != 75 * 8 and count < 3:
            count += 1
            fetch_items(drivers, items, fetchfn, save_every_n=2)
            for d in drivers:
                d.quit()

        # seeking alpha show 75 articles per page
        self.assertEqual(len(result_files_to_df().index), 75 * 8)


if __name__ == '__main__':
    unittest.main()
