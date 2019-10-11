RAWDATA = '../../data/raw'
PROCDATA = '../../data/processed'
MODS = '../../data/models'

rule benchmark:
    output:
        ['../../data/models/{cell_type}/' + benchfile for benchfile in ['benchmark_sample_traces]]
