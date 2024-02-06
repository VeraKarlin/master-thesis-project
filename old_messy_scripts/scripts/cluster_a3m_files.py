
import os
import glob
import subprocess


def main():
    path = "/home/vera/projects/masters_project/data/benchmark_data/"
    script_path = "/home/vera/projects/masters_project/AFcluster/scripts/ClusterMSA.py"

    if not os.path.exists(script_path):
        print("ERROR: Script path", script_path, "does not exist")

    for category in ['af_cluster_monomeric', 'af_cluster_oligomeric', 'classic']:
        category_path = path + category + '/'
        for file in os.listdir(category_path):
            if file[0] == '.':
                continue
            msa_path = [glob.glob(category_path + file + '/alignments/colab*.a3m')][0][0]
            if not os.path.exists(msa_path):
                print("ERROR: MSA path", msa_path, "does not exist")
            cluster_path = category_path + file + '/af_clusters/'
            if not os.path.exists(cluster_path):
                print("ERROR: Cluster path", cluster_path, "does not exist")
            
            python_command = ' '.join(["python", script_path, "cluster_" + file.split('/')[-1], "-i", msa_path, "-o", cluster_path])
            print(python_command)
            subprocess.call(python_command, shell=True)


if __name__ == "__main__":
    main()