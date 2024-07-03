import re
import matplotlib.pyplot as plt
from collections import defaultdict


def seeds_8mer(miRNA):
    return [
        'A' + rev_compl(miRNA[1:8]) # 8mer - full complementarity on positions 2-8 and A on the position 1
    ]

def seeds_7mer(miRNA):
    return [
        rev_compl(miRNA[1:8]), # 7mer-m8 - full complementarity on positions 2-8
        'A' + rev_compl(miRNA[1:7]) # 7mer-A1 - full complementarity on positions 2-7 and A on the position 1
    ]

def seeds_6mer(miRNA):
    return [
        rev_compl(miRNA[1:7]), # 6mer - full complementarity on positions 2-7
        rev_compl(miRNA[2:8]), # 6mer-m8 - full complementarity on positions 3-8
        'A' + rev_compl(miRNA[1:6]) # 6mer-A1 - full complementarity on positions 2-6 and A on the position 1
    ]

def seeds_6mer_bulge(miRNA):
    mers = []
    mers.append(rev_compl(miRNA[1:7]))
    for pos in range(1, 7):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                rev_compl(miRNA[1:7])[:pos] + nt + rev_compl(miRNA[1:7])[pos:]
            )
    mers.append(rev_compl(miRNA[2:8]))
    for pos in range(2, 8):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                rev_compl(miRNA[2:8])[:pos] + nt + rev_compl(miRNA[2:8])[pos:]
            )
    mers.append('A' + rev_compl(miRNA[1:6]))
    for pos in range(1, 6):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                'A' + rev_compl(miRNA[1:6])[:pos] + nt + rev_compl(miRNA[1:6])[pos:]
            )

    return list(set(mers))

def seeds_6mer_bulge_or_mismatch(miRNA):
    mers = []
    mers.append(rev_compl(miRNA[1:7]))
    for pos in range(1, 7):
        for nt in ['A', 'C', 'G', 'T']:
            # bulges
            mers.append(
                rev_compl(miRNA[1:7])[:pos] + nt + rev_compl(miRNA[1:7])[pos:]
            )
            # mismatches
            mers.append(
                rev_compl(miRNA[1:7])[:pos] + nt + rev_compl(miRNA[1:7])[pos+1:]
            )
    mers.append(rev_compl(miRNA[2:8]))
    for pos in range(2, 8):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                rev_compl(miRNA[2:8])[:pos] + nt + rev_compl(miRNA[2:8])[pos:]
            )
            mers.append(
                rev_compl(miRNA[2:8])[:pos] + nt + rev_compl(miRNA[2:8])[pos+1:]
            )
    mers.append('A' + rev_compl(miRNA[1:6]))
    for pos in range(1, 6):
        for nt in ['A', 'C', 'G', 'T']:
            mers.append(
                'A' + rev_compl(miRNA[1:6])[:pos] + nt + rev_compl(miRNA[1:6])[pos:]
            )
            mers.append(
                'A' + rev_compl(miRNA[1:6])[:pos] + nt + rev_compl(miRNA[1:6])[pos+1:]
            )

    return list(set(mers))


def rev_compl(st):
    nn = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}
    return "".join(nn[n] for n in reversed(st))


def classify_site(gene, miRNA, get_seeds, thresh):
    seeds = get_seeds(miRNA)
    seed_counts = 0
    for seq in seeds:
        seed_counts += gene.count(seq)
        if seed_counts > thresh:
            return 1
    return 0


def get_seed_locations(gene, miRNA, get_seeds):
    seeds = get_seeds(miRNA)
    seed_loci_flags = [0] * len(gene)
    for seq in seeds:
        start_indexes = [m.start() for m in re.finditer(seq, gene)]
        for i in start_indexes:
            seed_loci_flags[i:i+len(seq)] = [1] * len(seq)
    return seed_loci_flags


def count_miRNA_seeds(mRNA, miRNA):
    # Initialize dictionary to keep track of positions
    positions_covered = defaultdict(bool)

    def count_seeds(seeds):
        count = 0
        for seed in seeds:
            start_positions = [i for i in range(len(mRNA)) if mRNA.startswith(seed, i)]
            for start in start_positions:
                if all(not positions_covered[pos] for pos in range(start, start + len(seed))):
                    count += 1
                    for pos in range(start, start + len(seed)):
                        positions_covered[pos] = True
        return count

    count_8mer = count_seeds(SEED_TYPE_TO_EXTRACTION_FUNCTION['kmer8'](miRNA))
    count_7mer = count_seeds(SEED_TYPE_TO_EXTRACTION_FUNCTION['kmer7'](miRNA))
    count_6mer = count_seeds(SEED_TYPE_TO_EXTRACTION_FUNCTION['kmer6'](miRNA))
    
    bulge_6mers = SEED_TYPE_TO_EXTRACTION_FUNCTION['kmer6_bulge'](miRNA)
    mismatch_6mers = SEED_TYPE_TO_EXTRACTION_FUNCTION['kmer6_bulge_or_mismatch'](miRNA)
    
    bulge_6mers.sort(key=lambda s: len(s), reverse=True)
    mismatch_6mers.sort(key=lambda s: len(s), reverse=True)
    
    count_6mer_bulge = count_seeds(bulge_6mers)
    count_6mer_mismatch = count_seeds(mismatch_6mers)

    return {
        'kmer8_count': count_8mer,
        'kmer7_count': count_7mer,
        'kmer6_count': count_6mer,
        'kmer6_bulge_count': count_6mer_bulge,
        'kmer6_bulge_or_mismatch_count': count_6mer_mismatch,
    }



SEED_TYPE_TO_EXTRACTION_FUNCTION = {
    'kmer8': seeds_8mer,
    'kmer7': seeds_7mer,
    'kmer6': seeds_6mer,
    'kmer6_bulge': seeds_6mer_bulge,
    'kmer6_bulge_or_mismatch': seeds_6mer_bulge_or_mismatch
}

