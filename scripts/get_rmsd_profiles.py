#!/usr/bin/env python3

from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver import FirefoxOptions


MAXLENGTH = 1024
df = pd.read_csv('../data/cluster_ids_out.csv', index_col=[0])

df = df.sort_values(by='rmsd', ascending=False)
df = df.reset_index(drop=True)
df2 = pd.DataFrame(data=[[-1 for column in range(1024)] for row in range(df.shape[0])], columns=[str(c) for c in range(MAXLENGTH)])
df = pd.concat([df, df2], axis=1)
print(df.shape)

opts = FirefoxOptions()
opts.add_argument("--headless")
browser = webdriver.Firefox(options=opts)

for i in range(df.shape[0]):
    ID = df.loc[i, 'pdb_id']
    print(ID)
    pdbID = ID[:4]
    chainID = ID[-1]
    url = 'https://pdbflex.org/php/api/rmsdProfile.php?pdbID=' + pdbID + '&chainID=' + chainID
    browser.get(url)
    browser.implicitly_wait(20)

    element = browser.find_elements(By.XPATH, '/html/body/div/div/div/div[1]/div/div/div[2]/table/tbody/tr[3]/td[2]/span/span')
    residue_rmsd = element[0].text.split('[')[-1].split(']')[0].split(',')
    start = df.columns.get_loc("0")
    series = pd.Series(residue_rmsd[:MAXLENGTH])
    df.loc[i, 'length'] = len(series)

    df.iloc[i, start:start+len(series)] = series

    if i % 1000 == 0:
        df.to_csv('../data/cluster_rmsds_out_' + str(i / 1000) + '.csv')
        browser.close()
        browser = webdriver.Firefox(options=opts)

    if i == 9:
        df.to_csv('../data/cluster_rmsds_out_top_10.csv')
        print(10)
        browser.close()
        browser = webdriver.Firefox(options=opts)


df.to_csv('../data/cluster_rmsds_out.csv')
browser.quit()

