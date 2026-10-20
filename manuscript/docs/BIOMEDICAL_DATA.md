# Biomedical applications: data inputs

The dataset sections in `biomedical_applications.r` are copied from the four
original analyses. They use the following inputs, relative to the
`manuscript/` working directory:

| Dataset | Input |
|---|---|
| Copenhagen Holter | `data/biomedical_applications/cphholter.csv` |
| BMT | `data/biomedical_applications/bmt.csv` |
| PROVA | `data/biomedical_applications/prova.csv` |
| EBMT3 | `data/biomedical_applications/ebmt3.csv` |

All four CSV files are bundled and read directly by the study script.
The Copenhagen Holter, BMT and PROVA files come from the companion material for *Models for
Multi-State Survival Data*, Chapter 1:
<https://multi-state-book.github.io/companion/Ch1.html>.
EBMT3 is exported from `mstate::ebmt3` (mstate 0.3.3). The files retain the
wide format used in these analyses; no fitted objects or score tables are needed.
The script restores EBMT3's original factor levels after reading its CSV,
including their order and reference categories.

Several original preparation steps select columns by position. Preserve
these column names **and their order**, without adding a row-number column:

**cphholter.csv**

```text
id,timedeath,death,timeafib,afib,timestroke,stroke,sex,age,smoker,esvea,chol,diabet,bmi,aspirin,probnp,sbp
```

**bmt.csv**

```text
id,team,timedeath,death,timerel,rel,timegvhd,gvhd,timeanc500,anc500,sex,age,all,bmonly
```

**prova.csv**

```text
id,timedeath,death,timebleed,bleed,beta,scle,sex,age,bili,coag,varsize
```

**ebmt3.csv**

```text
id,prtime,prstat,rfstime,rfsstat,dissub,age,drmatch,tcd
```

The scripts retain the original missing-value imputation, event ordering
and negligible adjustments for tied within-subject event times. BMT
excludes `team`; its four features are `sex`, `age`, `all` and `bmonly`.
Output directories are created by `helpers/setup.r`.

Section 7 of `biomedical_applications.r` builds the combined absolute and
relative error grids after all four datasets have harmonised score tables.
