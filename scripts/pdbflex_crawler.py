from selenium import webdriver
from selenium.webdriver.common.by import By
import pandas as pd
from selenium.webdriver import FirefoxOptions


opts = FirefoxOptions()
opts.add_argument("--headless")
browser = webdriver.Firefox(options=opts)

url = 'https://pdbflex.org/clusters.html'

browser.get(url)
browser.implicitly_wait(30)

show_100_button = browser.find_elements(By.XPATH, '//*[@id="clsTable_length"]/label/select/option[4]')
show_100_button[0].click()

d = {'pdb_id': [], 'rmsd': [], 'members': []}
df = pd.DataFrame(data=d)

for i in range(384):
    if i % 10 == 0:
        print(i)
        df.to_csv('../data/cluster_ids_out' + str(i) + '.csv')
    pdb_cards = browser.find_elements(By.CLASS_NAME, 'sorting_1')
    row = 1
    for card in pdb_cards:
        pdb_id = card.find_element(By.CLASS_NAME, 'clusterLink').get_attribute('data-pdbid')
        rmsd = card.find_elements(By.XPATH, '//*[@id="clsTable"]/tbody/tr[' + str(row) + ']/td[2]')[0].text
        members = card.find_elements(By.XPATH, '//*[@id="clsTable"]/tbody/tr[' + str(row) + ']/td[3]')[0].text
        df.loc[i*100+row,'pdb_id'] = pdb_id
        df.loc[i * 100 + row, 'rmsd'] = rmsd
        df.loc[i * 100 + row, 'members'] = members
        print(i*100+row, pdb_id, rmsd, members)
        row += 1
    element = browser.find_elements(By.XPATH, '//*[@id="clsTable_next"]/a')
    element[0].click()

df.to_csv('../data/cluster_ids_out.csv')
