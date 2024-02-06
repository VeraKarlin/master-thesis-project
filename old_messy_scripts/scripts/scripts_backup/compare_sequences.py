


#!/usr/bin/python

import string
from sys import argv, stdout
from os import popen, system
from os.path import exists
from amino_acids import longer_names
from Bio import SeqIO


def fasta_to_seq(fasta):
    with open(fasta, "r") as f:
        record = SeqIO.read(f, "fasta")
        fasta_sequence = str(record.seq)
    return fasta_sequence


def pdb_to_seq(pdbnames, chainid):
    chainids = list(chainid)
    for pdbname in pdbnames:
        sequence = []
        for chainid in chainids:
            removechain = 0
            if argv.count('-nochain'):
                removechain = 1

            netpdbname = pdbname

            lines = open(netpdbname, 'r').readlines()
            fastaid = stdout

            oldresnum = '   '
            count = 0
            for line in lines:
                if (len(line) > 20) and (chainid == line[21]):
                    line_edit = line
                    if line[0:3] == 'TER':
                        break
                    elif (line[0:6] == 'HETATM') & (line[17:20] == 'MSE'):  # Selenomethionine
                        line_edit = 'ATOM  ' + line[6:17] + 'MET' + line[20:]
                        if (line_edit[12:14] == 'SE'):
                            line_edit = line_edit[0:12] + ' S' + line_edit[14:]
                        if len(line_edit) > 75:
                            if (line_edit[76:78] == 'SE'):
                                line_edit = line_edit[0:76] + ' S' + line_edit[78:]

                    if line_edit[0:4] == 'ATOM':
                        resnum = line_edit[23:26]
                        if not resnum == oldresnum:
                            count = count + 1
                            longname = line_edit[17:20]
                            # if longer_names.has_key(longname):
                            if longname in longer_names:
                                # fastaid.write(longer_names[longname])
                                sequence.append(longer_names[longname])
                            else:
                                #fastaid.write('X')
                                sequence.append('X')
                        oldresnum = resnum

                        newnum = '%3d' % count
                        line_edit = line_edit[0:23] + newnum + line_edit[26:]
                        if removechain:
                            line_edit = line_edit[0:21] + ' ' + line_edit[22:]
            fastaid.write('\n')
        return "".join(sequence)


if __name__ == "__main__":
    inputs = argv[1:]
    fasta = [p for p in inputs if ".seq" in p or ".fasta" in p][0]
    pdbnames = [p for p in inputs if ".pdb" in p]
    chainid = ' '
    if len(argv) > 3:
        chainid = argv[-1]
    pdb_sequence = pdb_to_seq(pdbnames, chainid)
    fasta_sequence = fasta_to_seq(fasta)
    print(pdb_sequence)
    print(fasta_sequence)

