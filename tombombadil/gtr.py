"""Codon GTR rate construction and nonsynonymous-rate updates."""

import jax
import jax.numpy as jnp


# Rows/columns follow alignment.CODON_LIST. a..f select alpha..eta
# (AG, AC, AT, CG, GT, CT); uppercase marks nonsynonymous changes.
# '.' denotes structural zeros. Decode this fixed topology outside JIT tracing.
_SUBSTITUTIONS = (
    ".fCEF...C.E..F...............C...............E...............",
    "f.BD.F...C.E..F...............C...............E..............",
    "CB.a..F........f...............C...............E.............",
    "EDa....F....E...f...............C...............E............",
    "F....fceB.D......F...............C...............E...........",
    ".F..f.bd.B.D......F...............C...............E..........",
    "..F.cb.a...........F...............C...............E.........",
    "...Feda.....D.......F...............C...............E........",
    "C...B....fA..........F...............C...............E.......",
    ".C...B..f..A..........F...............C...............E......",
    "E...D...A..fE............F...............C...............E...",
    ".E...D...Af.D.............F...............C...............E..",
    "...E...D..ED................F...............C...............E",
    "F.............fceF...C...E...B...............D...............",
    ".F...........f.bd.F...C...E...B...............D..............",
    "..f..........cb.a..F...C...E...B...............D.............",
    "...f.........eda....F...C...E...B...............D............",
    "....F........F....fceB...D.......B...............D...........",
    ".....F........F..f.bd.B...D.......B...............D..........",
    "......F........F.cb.a..B...D.......B...............D.........",
    ".......F........Feda....B...D.......B...............D........",
    "........F....C...B....fCEA...........B...............D.......",
    ".........F....C...B..f.BD.A...........B...............D......",
    "...............C...B.CB.a..A...........B...............D.....",
    "................C...BEDa....A...........B...............D....",
    "..........F..E...D...A....fce............B...............D...",
    "...........F..E...D...A..f.bd.............B...............D..",
    "...............E...D...A.cb.a..............b...............D.",
    "............F...E...D...Aeda................b...............D",
    "C............B................fcEF...C...E...A...............",
    ".C............B..............f.bD.F...C...E...A..............",
    "..C............B.............cb.A..F...C...E...A.............",
    "...C............B............EDA....F...C...E...A............",
    "....C............B...........F....fceB...D.......A...........",
    ".....C............B...........F..f.bd.B...D.......A..........",
    "......C............B...........F.cb.a..B...D.......A.........",
    ".......C............B...........Feda....B...D.......A........",
    "........C............B.......C...B....fCEA...........A.......",
    ".........C............B.......C...B..f.BD.A...........A......",
    ".......................B.......C...B.CB.a..A...........A.....",
    "........................B.......C...BEDa....A...........A....",
    "..........C..............B...E...D...A....fCE............A...",
    "...........C..............B...E...D...A..f.BD.............A..",
    "...........................b...E...D...A.CB.a..............A.",
    "............C...............b...E...D...AEDa................A",
    "E............D...............A................fceF...C...E...",
    ".E............D...............A..............f.bd.F...C...E..",
    "..E............D...............A.............cb.a..F...C...E.",
    "...E............D...............A............eda....F...C...E",
    "....E............D...............A...........F....fceB...D...",
    ".....E............D...............A...........F..f.bd.B...D..",
    "......E............D...............A...........F.cb.a..B...D.",
    ".......E............D...............A...........Feda....B...D",
    "........E............D...............A.......C...B....fCEA...",
    ".........E............D...............A.......C...B..f.BD.A..",
    ".......................D...............A.......C...B.CB.a..A.",
    "........................D...............A.......C...BEDa....A",
    "..........E..............D...............A...E...D...A....fce",
    "...........E..............D...............A...E...D...A..f.bd",
    "...........................D...............A...E...D...A.cb.a",
    "............E...............D...............A...E...D...Aeda.",
)
_RATE_INDEX = jnp.array([
    ["abcdef.".index(code.lower()) for code in row] for row in _SUBSTITUTIONS
], dtype=jnp.int32)
_NONSYNONYMOUS = jnp.asarray([[code.isupper() for code in row] for row in _SUBSTITUTIONS])
_DIAGONAL = jnp.eye(61, dtype=bool)


@jax.profiler.annotate_function
@jax.jit
def diag_update(M, pimult):
    """Replace the diagonal with negative frequency-weighted off-diagonal sums."""
    offdiag = jnp.where(_DIAGONAL, 0.0, M)
    diagonal = -jnp.sum(offdiag * pimult, axis=1)
    return jnp.where(_DIAGONAL, diagonal[:, None], offdiag)


@jax.jit
def build_GTR(alpha, beta, gamma, delta, epsilon, eta, omega, pimat, pimult):
    """Build the symmetric rate matrix; pimat is diag(sqrt(codon frequencies))."""
    rates = jnp.stack((alpha, beta, gamma, delta, epsilon, eta))
    rates = jnp.concatenate((rates, jnp.zeros(1, dtype=rates.dtype)))
    M = rates[_RATE_INDEX] * jnp.where(_NONSYNONYMOUS, omega, 1.0)
    sqrt_pi = jnp.diag(pimat)
    M = (M * sqrt_pi[None, :]) * sqrt_pi[:, None]
    return diag_update(M, pimult)


@jax.jit
def update_GTR(M, omega, pimult):
    """Scale nonsynonymous off-diagonal entries, then replace the diagonal."""
    return diag_update(M * jnp.where(_NONSYNONYMOUS, omega, 1.0), pimult)
