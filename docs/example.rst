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

Output files are prefixed by omega mode. With `--output-jax output`, scalar
MAP fitting writes `scalar_output_Allparams.csv`; per-site MAP fitting writes
`per_site_output_GTRparams.csv` and `per_site_output_omega.csv`. NUTS files
use the same `scalar_` or `per_site_` prefix.
