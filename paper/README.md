# Manuscript

One text, two journal formats:

| File | Target | Class |
|---|---|---|
| `ieee_main.tex` / `ieee_main.pdf` | IEEE Internet of Things Journal | IEEEtran, two columns |
| `jnca_main.tex` / `jnca_main.pdf` | Journal of Network and Computer Applications (Elsevier) | elsarticle, preprint |

Sections live in `body/` and are shared by both versions. `body/rev_*.tex` (tables and the
data-driven sentences of the results section) and `figs/rev_*.pdf` are generated from the raw
results in `results/revision/`; do not edit them by hand.

## Rebuilding everything (from the repository root)

```sh
.venv/bin/python test_revision.py                          # model, oracle, routers, lossy model
sh results/revision/run_all.sh                             # test-phase experiments (hours, 4 cores)
.venv/bin/python -m revision.madii_diag                    # learned-router reward diagnosis
.venv/bin/python -m revision.analyze                       # statistics, tables, figures
cd paper && latexmk -pdf ieee_main.tex && latexmk -pdf jnca_main.tex
```

Validation tuning (already done; settings fixed in `revision/routers.py` TUNED):
`.venv/bin/python -m revision.run --exp tune` and `--exp tune2`.

`RESPONSE_TO_REVIEWERS.md` maps every review comment to the change that addresses it.
`rebuild_publication.py` belongs to the previous revision round and is superseded by `revision/`.

## Before submission

- Corresponding-author e-mail, CRediT roles, funding statement.
- Confirm the AI-assistance statements name every tool that was used.
