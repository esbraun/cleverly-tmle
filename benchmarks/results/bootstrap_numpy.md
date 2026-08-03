# The multiplier bootstrap, in numpy

`findings.md` §2.1 calls the Rademacher multiplier bootstrap "the clearest adoption" for a
compiled kernel: 2.4–2.5× serial, 7.4–7.6× on four cores, and a 264× reduction in allocation.
Those ratios are correct and the conclusion drawn from them was not, because the baseline they
were measured against was the shipped implementation rather than a competent one.

This is the numpy change and what it leaves for a compiler.

> Measured on the four-core Intel Xeon @ 2.80 GHz container this repository's cloud sessions
> run in, `/proc/loadavg` under 0.6, Python 3.11, numpy 2.4.6, OpenBLAS 0.3.31. Medians of
> interleaved repetitions; memory is `tracemalloc` peak over one untimed call.

## 1. The profile the previous one pointed the wrong way

`candidate_inventory.md` §2.7 and `docs/roadmap.md` both record this path as "92–95%
multiplier *generation*", which reads as an argument about the random draw. Splitting the
generation at `n = 100,000` with a 256-replicate block:

| step | ms | share |
| --- | ---: | ---: |
| `rng.integers(0, 256, (256, n/8), uint8)` — **the seeded draw** | 3.5 | 2% |
| `np.unpackbits` — bits to `uint8` | 1.7 | 1% |
| `_SIGNS[bits]` — **expanding to a 205 MB float64 matrix** | **159.0** | **89%** |
| `xi @ centred` — the `dgemm` | 12.6 | 7% |

The draw is 2%. The expansion is 89%, and it is a 25-million-element gather that writes a
whole second array. Both of the things a compiled kernel was going to remove — the array and
the expansion — are removable in numpy, and the third thing it was going to change, the
generator, is 2% of the cost and did not need to change at all.

## 2. What changed

Two edits to `cleverly/inference/multiplier.py`, and neither is clever:

**Expand in place.** `_SIGNS[bits]` becomes `np.copyto(out, bits); out *= 2; out -= 1` into a
buffer allocated once and reused across blocks. `±1` is exactly representable, so this is the
same arithmetic and not an approximation of it. `mammen` gets the same treatment through
`np.copyto(..., where=flags)`; `normal` never materialised a multiplier matrix and is
untouched.

**Size the block in bytes, not replicates.** The optimum tracks the buffer's footprint
against the cache, not the replicate count, and a block that is comfortable at `n = 10,000`
allocates two gigabytes at `n = 1,000,000`. Measured at `n = 100,000`, `B = 512`:

| block | buffer | ms |
| ---: | ---: | ---: |
| 4 | 3.2 MB | 257.8 |
| 16 | 12.8 MB | 182.6 |
| 32 | 25.6 MB | 138.5 |
| **64** | **51.2 MB** | **135.4** |
| 128 | 102.4 MB | 152.8 |
| 256 | 204.8 MB | 292.3 |

At `n = 1,000,000` the same curve peaks at an 8-replicate block — 64 MB, the same footprint
again. So the block is `32 MB / (8n)`, clamped to `[4, 256]` and rounded down to a multiple of
four.

## 3. What it bought

| configuration | before | after | speed-up | before | after | memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| n = 20,000, m = 7, B = 1,000 | 0.152 s | 0.045 s | **3.35×** | 88.9 MB | 39.1 MB | 0.44× |
| n = 100,000, m = 7, B = 1,000 | 0.706 s | 0.203 s | **3.48×** | 444.1 MB | 42.1 MB | **0.09×** |
| n = 1,000,000, m = 7, B = 200 | 4.350 s | 1.114 s | **3.90×** | 1,881.1 MB | 92.5 MB | **0.05×** |

Against `findings.md`'s compiled kernel at **2.4–2.5× serial**: the numpy path is now faster
than the numba one was, on the same box, with no dependency, no 2.0 s compilation and no
change to the seeded stream. The compiled kernel's four-core arm (7.4–7.6×) is still ahead of
this, by about 2× rather than by 3×, and that is the number a decision about adopting numba
here has to be worth.

The memory column is the more durable half. `docs/roadmap.md` names the `(chunk, n)`
multiplier matrix as one of two allocations that break before any arithmetic does — 9.5 GB at
`n = 5,000,000`. What is left is a buffer with a **fixed 32 MB budget at any `n`**, so the
allocation no longer grows with the sample at all. That is the same capability change
`findings.md` attributed to the compiled kernel, obtained without it.

## 4. The seeded stream is unchanged, and the block does not move it

The draw was not touched, so a fixed `random_state` produces the multipliers it always did.
Checked directly: the new critical value is **bit-identical** to the old implementation across
`n ∈ {3, 500, 2,000, 10,000, 100,000}`, `m ∈ {2, 3, 5, 7}`, `B ∈ {50, 200, 500, 997, 1,000}`,
both two-point kinds and two seeds — including replicate counts that leave a partial final
block.

The block size is free to vary because `rng.integers(0, 256, ..., uint8)` fills from buffered
32-bit words: a block that is a multiple of four consumes whole words and leaves the byte
stream where the next block picks it up, so replicate `b` gets the same multipliers at every
block. That is pinned on the multipliers themselves
(`test_the_block_size_does_not_change_the_multipliers`), where the claim is exact.

The *critical value* follows to rounding rather than exactly: `xi @ centred` is a `dgemm`
whose blocking depends on its operand shape, so the sum over `n` can accumulate in a different
order at a different block. Measured at a relative 1e-15, and pinned at 1e-12 rather than at
equality — a test asserting equality there would be pinning OpenBLAS's blocking heuristics.
Every configuration in the paragraph above happens to be exactly equal at the block the byte
budget picks; one configuration at `B = 997` was not, which is why the claim is stated this
way.

## 5. What was measured and not taken: float32

Expanding to `float32` and running the `dgemm` at `float32` measures **7.18×** against the
shipped path where the float64 rewrite measures 3.4–3.9×, because the expansion writes half
the bytes.

| n | shipped | float64 blocked | float32 blocked |
| ---: | ---: | ---: | ---: |
| 20,000 | 229.9 ms | 63.2 ms (3.64×) | 29.1 ms (7.89×) |
| 100,000 | 2,487.5 ms | 524.6 ms (4.74×) | 139.7 ms (17.81×) |

(That table is from a cold first pass and its absolute times are noisier than §3's; the
ordering is the point.) The critical values agreed to 6 decimal places — 2.645745 against
2.645744 at `n = 100,000` — against a resampling error at `B = 1,000` of order 1e-2.

**Not taken here.** It is a change to the arithmetic of a reported quantity, and "the error is
much smaller than the Monte Carlo error" is an argument that has to be made at the tails and
under leverage, not at one well-behaved configuration: the whole reason `"rademacher"` is the
default rather than `"normal"` is that this package cares what happens to a critical value
when the influence curve has a `1/g` in it. Deciding it is its own change, with its own
tolerance analysis and its own tests under weak overlap.

## 6. What this leaves for a compiled kernel

A fused kernel that never expands the bits at all — reading the packed bytes and accumulating
signs into an `m`-vector — remains the only way to remove the last 89%. The prototype written
for `production_plan.md` §1.2 measured **275 ms serial and 69 ms on four threads** per 256
replicates against this path's ~135 ms, because extracting the bit in the inner loop defeats
the vectoriser. The kernel `benchmarks/numba/kernels/bootstrap.py` measures does better — it
draws its own signs from a counter hash, which fuses the sign into the accumulate — but that
is the design that gives up the seeded stream, for a 2% share of the cost it does not need to
touch.

So the remaining question is narrow and worth stating precisely: **is there a compiled kernel
that consumes numpy's packed bytes, unpacks eight signs at a time into registers, and beats
0.203 s at `n = 100,000`?** Until one is written and measured, the honest classification for
this kernel is *unresolved*, not *strong production candidate*.
