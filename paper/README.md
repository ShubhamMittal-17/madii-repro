# Manuscript

One text, two journal formats:

| File | Target | Class |
|---|---|---|
| `jnca_main.tex` / `jnca_main.pdf` | Journal of Network and Computer Applications (Elsevier) | elsarticle, preprint |
| `ieee_main.tex` / `ieee_main.pdf` | IEEE Internet of Things Journal | IEEEtran, two columns |

The sections live in `body/` and are shared by both versions. `preamble.tex` holds the common
packages and macros, and `refs.bib` the references.

## Rebuilding

```sh
.venv/bin/python wilcoxon_tables.py        # results_wilcoxon.{md,json}  (from the repo root)
.venv/bin/python paper/make_tables.py      # paper/body/tab_*.tex
.venv/bin/python paper/make_figures.py     # paper/figs/*.pdf
cd paper && latexmk -pdf jnca_main.tex && latexmk -pdf ieee_main.tex
```

Every number in the generated tables comes from the saved result files; no simulation is rerun.

## Before submission

- Fill the author, affiliation and e-mail placeholders, the CRediT statement, the repository URL
  and the acknowledgements (`body/declarations.tex`).
- Complete the generative-AI declaration truthfully (Elsevier requires it whenever such tools were used).
- Complete the bibliography entries marked `AUTHORS:` in `refs.bib` (Tang et al., ICMCSI-2026, the two arXiv papers).
- JNCA: upload the highlights (also in `jnca_main.tex`) as a separate file.
