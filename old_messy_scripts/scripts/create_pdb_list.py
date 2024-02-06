import pandas as pd

# Load in the dataframe.
df = pd.read_csv('cluster_ids_out.csv')

# Create a text file of pdb ids.
with open('pdb_ids.txt', 'w') as f:
    f.write(' '.join(df.loc[:, 'pdb_id'].values.tolist()))
