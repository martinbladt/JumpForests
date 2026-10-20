# Coverage outputs and Git

The full study writes to `results/coverage/`. The manuscript `.gitignore`
keeps raw datasets, fitted objects, per-replication files, checkpoints and
figures out of Git. It allows these paper/tuning outputs:

- `results/coverage/*_by_group.tsv`;
- `results/coverage/simultaneous_by_profile.tsv`;
- `results/coverage/n*/pilot/*.tsv`;
- LaTeX/TSV files in `results/coverage/Tables/`.

Other studies' `Tables/*.tex`, `Tables/*.tsv` and `tuning_*.txt` files are also
allowed. The four bundled biomedical CSV inputs are also allowed; other local
inputs under `data/` are ignored.

To regenerate coverage summaries and paper outputs, use the reporting section
of `coverage.r` with `run_study <- FALSE` and `make_reports <- TRUE`. The helper
`plot_coverage_study()` also maintains a gzip copy of the large detailed
profile summary; that runtime archive remains ignored in this code distribution.

These rules apply to the manuscript copy. The original project's files and
Git rules are unchanged. Ignore rules do not untrack already committed files.
