# Y-chromosome measurements for the ABC lab

The source is the [Pritchard lab Y-chromosome archive](https://web.stanford.edu/group/pritchardlab/dataArchive/ydata.html), associated with Pritchard et al. (1999). `ydata-source.html` is a local copy retrieved 2026-10-07. The archive contains 445 named male samples at eight markers. These are measurements from a pooled worldwide sample, not a random sample from one unstructured population.

Run `Rscript prepare_data.R` from the lab directory to regenerate the cleaned files. The script parses all 445 source records, checks for noninteger repeat changes, and writes:

| File | Contents |
|:--|:--|
| `observed_repeat_counts.csv` | 440 rows × 8 integer repeat counts; no ID column |
| `retained_source_ids.csv` | Source ID in the same row order |
| `excluded_rows.csv` | The five exclusions and reason |

The archive columns `389a` and `389c` are already separated DYS389 components. The script does **not** subtract one from the other again. Fragment sizes are converted using these repeat-unit lengths and fixed offsets:

| Marker | DYS19 | DYS388 | DYS389a | DYS389c | DYS390 | DYS391 | DYS392 | DYS393 |
|:--|--:|--:|--:|--:|--:|--:|--:|--:|
| Bases per repeat | 4 | 3 | 4 | 4 | 4 | 4 | 3 | 4 |
| Offset subtracted before division | 0 | 2 | 1 | 0 | 0 | 0 | 1 | 0 |

The offsets define arbitrary zero points for each marker and do not affect the lab summaries. Only five records fail the repeat lattice: their DYS389c fragment is 99 bases, three bases from the other observations' four-base lattice. They are listed in `excluded_rows.csv`. The resulting 440-person summaries are:

| Summary | Computed | Beaumont et al. (2002), rounded |
|:--|--:|--:|
| Mean sample variance of repeat counts | 1.122617 | 1.123 |
| Mean diversity, $1-\sum_a \widehat p_a^2$ | 0.634672 | 0.635 |
| Distinct complete profiles | 312 | 312 |

The variance uses R's `var` (denominator $n-1$). Diversity is uncorrected (denominator $n$ in allele frequencies); applying the finite-sample correction does **not** reproduce 0.635. The summary function in `population_growth_sim.R` applies these conventions to both real and simulated datasets.

The cleaned 440-person comparison and its prior are in [Beaumont, Zhang, and Balding (2002), p. 2033](https://people.eecs.berkeley.edu/~jordan/sail/readings/beaumont-zhang-balding.pdf). Their Table 1 describes a different, 445-person natural-parameter comparison and must not be used as a direct benchmark for this 440-person analysis.
