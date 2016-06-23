import os
import sys
from selenium import webdriver
import unittest
import pandas as pd


TESTPATH = os.path.dirname(os.path.realpath(__file__))
PYPATH = os.path.join(TESTPATH, '..', '..')
sys.path.append(PYPATH)

from pydwork.pullin import *


class PullInTest(unittest.TestCase):
    # no setUp and tearDown

    def test_seeking_alpha(self):
        def init_drivers(n, client):
            drivers = []
            for _ in range(n):
                drivers.append(client())
            return drivers

        def fetch1(driver, item):
            driver.get(item)
            box = driver.find_element_by_css_selector('ul.stripes_list')
            links = box.find_elements_by_css_selector("li > div > a")
            return pd.DataFrame({'article_addr':
                                 [link.get_attribute("href")
                                  for link in links]})

        drivers = init_drivers(2, webdriver.PhantomJS)
        # drivers = init_drivers(2, webdriver.Firefox)

        base_addr = "http://seekingalpha.com/articles?page="

        items = [base_addr + str(i) for i in range(1, 9)]

        fetch(drivers, items, fetch1, save_every_n=2)

        # seeking alpha show 75 articles per page
        self.assertEqual(result_files_to_df().shape, (75 * 8, 1))


unittest.main()
