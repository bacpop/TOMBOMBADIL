# TOMBOMBADIL <img src="https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png" alt="" width="200"/>
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci 

Estimate dN/dS directly from alignments using codon counts, no tree required.

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil 


## Installation

python -m pip install git+https://github.com/bacpop/TOMBOMBADIL.git@v0.1.0 

## Running tests

With the project environment active and pytest available, run the complete test
suite with:

```bash
python -m pytest
```

## Documentation

docs/

## Basic usage

You will first need to have an alignment of DNA sequences, aligned into codons. These can be produced by (e.g. revtrans)

Estimate dN/dS using TOMBOMBADIL by running one of the following commands from within the folder:

Fit one omega estimate for the whole alignment (scalar omega) with maximum a posteriori (MAP) optimisation (default)
```bash
python -m tombombadil --alignment alignment.fas.aln --fit-replicates 4 --output-jax output.txt
```

MAP fitting stops early when convergence is reached by default. Use
`--fixed-iterations` to run all `--max-it` optimizer steps.

Fit one omega estimate for per codon position in the alignment with maximum a posteriori (MAP) optimisation
```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --output-jax output
```

For more examples see the [documentation](https://bacpop.github.io/TOMBOMBADIL/)



