#!/usr/bin/env python

import gzip
import numpy as np
from sample import run_sampler

# expected order
# "TTT","TTC","TTA","TTG","TCT","TCC","TCA","TCG","TAT","TAC","TGT","TGC"
# "TGG","CTT","CTC","CTA","CTG","CCT","CCC","CCA","CCG","CAT","CAC","CAA"
# "CAG","CGT","CGC","CGA","CGG","ATT","ATC","ATA","ATG","ACT","ACC","ACA"
# "ACG","AAT","AAC","AAA","AAG","AGT","AGC","AGA","AGG","GTT","GTC","GTA"
# "GTG","GCT","GCC","GCA","GCG","GAT","GAC","GAA","GAG","GGT","GGC","GGA"
# "GGG"
# mapped to
col_order = np.array([63, 61, 60, 62, 55, 53, 52, 54, 51, 49, 59, 57, 58, 31, 29, 28, 30,
                      23, 21, 20, 22, 19, 17, 16, 18, 27, 25, 24, 26, 15, 13, 12, 14,  7,
                       5,  4,  6,  3,  1,  0,  2, 11,  9,  8, 10, 47, 45, 44, 46, 39, 37,
                      36, 38, 35, 33, 32, 34, 43, 41, 40, 42, 48, 50, 56, 64])
#stop codons 48 50 56
#Ns 64

def read_fasta(fp):
    name, seq = None, []
    for line in fp:
        line = line.rstrip()
        if line.startswith(">"):
            if name: yield (name, ''.join(seq))
            name, seq = line[1:], []
        else:
            seq.append(line)
    if name: yield (name, ''.join(seq))

def count_codons(file_name):
    with open(file_name, 'rb') as test_f:
        zipped = test_f.read(2) == b'\x1f\x8b'
    if zipped:
        fh = gzip.open(file_name, 'rt')
    else:
        fh = open(file_name, 'rt')
    with fh as fasta:
        X = None
        for h, s in read_fasta(fasta):
            s = np.frombuffer(s.lower().encode(), dtype=np.int8)
            if X is None:
                X = np.array((65, s.shape[0]), dtype=np.int32)
            # Set ambiguous bases
            ambig = s[(s!=97) & (s!=99) & (s!=103) & (s!=116)]
            if ambig.any():
                s[ambig] = 65
            codon_s = s.reshape(-1, 3).copy()
            # Convert to usual binary encoding
            codon_s[codon_s==97] = 0
            codon_s[codon_s==99] = 1
            codon_s[codon_s==103] = 2
            codon_s[codon_s==116] = 3
            # Bit shift
            codon_s[:, 1] = np.left_shift(codon_s[:, 1], 2)
            codon_s[:, 2] = np.left_shift(codon_s[:, 2], 4)
            codon_map = np.fmin(np.sum(codon_s, 1), 65)
            # slow? Alternative would be to make X have shape (samples, n_codons)
            # and copy codon map into each row, then run np.bincount along columns
            for idx, count in enumerate(codon_map):
                X[count, idx] += 1

    # reorder and cut off stops, ambiguous
    X = X[col_order,:]
    X = X[0:61, :]

    return X

def main():
    # Need to read in data
    # see https://github.com/gtonkinhill/pairsnp-python/blob/master/pairsnp/pairsnp.py
    # and https://github.com/gtonkinhill/panaroo/blob/master/panaroo/prokka.py
    # but should be easy enough, just use a static lookup from codon to idx
    # (eventually reduce sum/merge would work if wanting to parallelise)
    print(run_sampler(X, pi_eq, warmup, samples))

if __name__ == "__main__":
    main()
