import sys
import pandas as pd

MAX_LENGTH = 400
MIN_LENGTH = 50
MIN_RMSD = 1


def main(csv_path, output_file):
    """
    Filter the pdbflex dataset to only include proteins with length between 50 and 400 and rmsd > 1.
    
    Input: csv file with columns: pdb_id, length, rmsd
    output: list of pdb ids that meet the criteria
    """
    

    df = pd.read_csv(csv_path, index_col=[0])
    df = df.drop(df[df["length"] > MAX_LENGTH].index)
    df = df.drop(df[df["length"] < MIN_LENGTH].index)
    df = df.drop(df[df["rmsd"] < MIN_RMSD].index)

    pdb_list = df.sort_values(by='length', ascending=False)['pdb_id'].tolist()
    pdb_list = [pdb[:4] + '_' + pdb[4].lower() for pdb in pdb_list]

    with open(output_file, "w") as pdb_file:
        pdb_file.write('\n'.join(pdb_list))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
