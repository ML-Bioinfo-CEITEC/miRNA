import re
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support, PrecisionRecallDisplay


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
    # TODO: allow imperfect match

    seeds = get_seeds(miRNA)
    seed_loci_flags = [0] * len(gene)
    for seq in seeds:
        start_indexes = [m.start() for m in re.finditer(seq, gene)]
        for i in start_indexes:
            seed_loci_flags[i:i+len(seq)] = [1] * len(seq)
    return seed_loci_flags


SEED_TYPES = {
    'kmer8': seeds_8mer,
    'kmer7': seeds_7mer,
    'kmer6': seeds_6mer,
    'kmer6_bulge': seeds_6mer_bulge,
    'kmer6_bulge_or_mismatch': seeds_6mer_bulge_or_mismatch
}


def plot_prc_with_seeds(data, seed_types, methods, title=''):
    fig, ax = plt.subplots(nrows = 1, ncols=1, figsize=(5, 5))
    #for index, name in enumerate(data.keys()):

    markers = {
        'kmer8': 'x',
        'kmer7': 'o',
        'kmer6': 'v',
        'kmer6_bulge': '*',
        'kmer6_bulge_or_mismatch': '^'
    }

    colors = ['brown', 'blue', 'orange', 'pink', 'gray', 'black']

    for seed_name in seed_types.keys():
        marker = markers[seed_name]
        for threshold in [1,2,3,4]:
        # for threshold in [1,2,3,4,5,6]:
            prec, rec, _, _ = precision_recall_fscore_support(data['label'].values, data[seed_name + "_count_" + str(threshold)].values, average='binary')

            ax.plot(rec, prec, marker, color=colors[threshold - 1],  label=seed_name + "_count_" + str(threshold))

    for meth in methods.keys():
        PrecisionRecallDisplay.from_predictions(
            methods[meth]['actual'], methods[meth]['predicted'], ax=ax,
            label=str(meth)
        )
        
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))
    return fig, ax