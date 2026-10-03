# TOMBOMBADIL <img src="https://github.com/bacpop/TOMBOMBADIL_jax/blob/main/TOMBOMBADIL_logo.png" alt="" width="200"/>
**T**ree-free **O**mega **M**apping **B**y **O**bserving **M**utations of **B**ases and **A**mino acids **D**istributed **I**nside **L**oci 

Estimate dN/dS directly from alignments using codon counts, no tree required

>    "Old Tom Bombadil is a merry fellow! Bright Blue his jacket is, and his boots are yellow!"
    —Tom Bombadil 

## Installation

TODO: unpin versions < in poetry, make one per line
create conda package
give single install command

## Running tests

With the project environment active and pytest available, run the complete test
suite with:

```bash
python -m pytest
```

## Documentation

TODO: link
use docs/ and use sphinx (see poppunk)

## Basic usage

You will first need to have an alignment of DNA sequences, aligned into codons. These can be produced by (e.g. revtrans)

Estimate dN/dS using TOMBOMBADIL by running one of the following commands from within the folder:

Fit one omega estimate for the whole alignment (scalar omega) with maximum a posteriori (MAP) optimisation (default)
```bash
python -m tombombadil --alignment alignment.fas.aln --fit-replicates 4 --fit-until-convergence --output-jax output.txt
```

Fit one omega estimate for per codon position in the alignment with maximum a posteriori (MAP) optimisation
```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --output-jax output
```

For more examples see the [documentation](https://tombombadil.bacpop.org/)

Optional domain JSON annotations can colour per-site omega plots. A reference
protein FASTA is required for mapping alignment columns to protein positions:

TODO: everything below here in main docs
main docs has:
- Intro page explaining algorithm briefly
- Usage page giving the different modes of running
- A worked example/tutorial, which also includes plots

```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --domains domains.json --reference reference.faa
```


- Scalar omega parameter inference with MCMC (NUTS) using blackJax

```bash
python -m tombombadil --alignment alignment.fas.aln --fit-method nuts --num-warmup 100 --num-samples 100 --num-chains 4 --output-jax output.txt --cpus 4 --nuts-chain-mode pmap
```


- Fit one omega estimate for per codon position in the alignment with MCMC (NUTS) using blackJax 
```bash
python -m tombombadil --alignment alignment.fas.aln --omega-mode per-site --fit-method nuts --num-warmup 100 --num-samples 100 --num-chains 4 --output-jax output.txt --cpus 4 --nuts-chain-mode pmap
```


## Mini tutorial

The repository includes the example codon alignment of porin porB of *Neisseria meningitidis* `porB3_aligned.fasta`.

### 1. Scalar omega with maximum a posteriori (MAP) optimisation
(takes around 30 seconds on one cpu core)

This is the default model: one omega is estimated for the complete alignment.
`--sample-it` controls the number of MAP optimisation steps.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode scalar \
  --fit-method map \
  --sample-it 500 \
  --output-jax porB3_map
```

This writes `scalar_porB3_map_Allparams.csv`.

### 2. Per-site omega with maximum a posteriori (MAP) optimisation
(takes about four minutes on one cpu core)

This estimates one omega value for each codon site.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode per-site \
  --fit-method map \
  --sample-it 500 \
  --output-jax porB3_map
```

This writes `per_site_porB3_map_GTRparams.csv`,
`per_site_porB3_map_omega.csv`, and a per-site omega plot.

### 3. Scalar omega with NUTS sampling
(takes about 2.5 minutes on four cpu cores)

NUTS estimates a posterior distribution for one alignment-wide omega.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode scalar \
  --fit-method nuts \
  --num-warmup 100 \
  --num-samples 100 \
  --num-chains 4 \
  --output-jax porB3_nuts \
  --cpus 4 \
  --nuts-chain-mode pmap
```

### 4. Per-site omega with NUTS sampling
(takes about one hour on four cpu cores)

This samples a posterior distribution for every codon-site omega. It is more
computationally demanding because the parameter dimension grows with the
alignment length.

```bash
python -m tombombadil \
  --alignment porB3_aligned.fasta \
  --omega-mode per-site \
  --fit-method nuts \
  --num-warmup 100 \
  --num-samples 100 \
  --num-chains 4 \
  --output-jax porB3_nuts \
  --cpus 4 \
  --nuts-chain-mode pmap
```

For CPU parallel chains, add `--nuts-chain-mode pmap --cpus 4`. Otherwise,
chains run sequentially by default. NUTS writes posterior samples and summary
files with the selected `scalar_` or `per_site_` prefix.

On the CPU backend, `--cpus` sets the JAX worker count for both MAP and NUTS
(default: 4). In NUTS `pmap` mode it also sets the number of local CPU devices.
This configures JAX workers; process CPU use can exceed the selected count due
to runtime helper threads and compilation work.

Output files are prefixed by omega mode. Without `--output-jax`, results use
the stem `output`; supplying `--output-jax STEM` replaces that stem. Scalar
MAP fitting writes `scalar_output_Allparams.csv`; per-site MAP fitting writes
`per_site_output_GTRparams.csv` and `per_site_output_omega.csv`, plus
`per_site_output_omega_plot.pdf`. NUTS writes posterior samples and summaries
with the same `scalar_` or `per_site_` prefix. MAP final estimates and NUTS
summaries are logged, and parameter estimates are saved in these CSV files.

MAP optimisation reports the log-likelihood after every optimizer step. In a
terminal this appears in the progress bar for each replicate; when output is
redirected, each value is written to the log. The initial objective evaluation
and first optimizer update are timed as startup, including JAX compilation when
needed; the remaining optimization time is reported separately. Every MAP run
also saves a likelihood-versus-iteration PDF with the best replicate
highlighted. Without `--output-jax`, the plot is saved in the
current directory as `scalar_likelihood_plot.pdf` or
`per_site_likelihood_plot.pdf`. With an output stem, it is saved alongside the
other results, for example `per_site_fit_likelihood_plot.pdf`. The per-site
omega plot uses a linear y-axis starting at zero and keeps the omega=1 guide
visible.

# More options
--convergence-tol x default=1e-6

--sample-it x number of sampling steps

--platform gpu/cpu/tpu

--cpus x default=4 JAX CPU worker setting for MAP and NUTS; also sets local CPU devices for NUTS pmap

--pi uniform/empirical/F3x4 default=uniform codon equilibrium frequencies

--pi-pseudocount x default=0.5 pseudocount for empirical and F3x4 codon frequencies
