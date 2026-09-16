# Data Dictionary

Four public datasets, each registered as a `DatasetSpec` in
`src/dr_bench/datasets.py`. Every dataset is reduced to one binary classification
target so the leaderboard is comparable across them. Targets use a median split,
so classes are balanced by construction.

Variable meanings below are transcribed from each source's official codebook.

---

## nscg_2023 — NSF National Survey of College Graduates

- **Source:** National Center for Science and Engineering Statistics (NCSES), NSF.
- **URL:** `https://ncses.nsf.gov/822/assets/0/files/college_grads_2023.zip` → member `pcg23Public/epcg23.csv`
- **Cycle:** 2023 (reference week Feb 2023; released Jan 13, 2025). Newest NSCG cycle; the survey is biennial.
- **Grain:** individual college graduate. ~94,606 records; ~79,800 after the employed filter.
- **Terms:** US Government public-use file, public domain.
- **Loader:** `nscg_zip` (read all 548 columns as strings).

**Target:** `SALARY` (annual salary), binarized at the median. Filter `LFSTAT == 1`
(employed). The salary sentinel `9999998` corresponds exactly to non-employed
respondents and is excluded.

**Numeric features:** `AGE`, `HRSWK` (principal-job weekly hours), `WKSWK`
(weeks worked).

**Categorical features:**

| Variable | Meaning |
|---|---|
| `SEX_2023` | Sex |
| `RACECAT` | Race category |
| `HISPANIC` | Hispanic origin indicator |
| `MARSTA` | Marital status |
| `CTZUSIN` | US citizenship indicator |
| `BTHUS` | Birthplace US / non-US |
| `DGRDG` | Highest degree type |
| `MRDG` | Most recent degree type |
| `BADG` | Bachelor's degree type |
| `N2BAMED` | Bachelor's major field (detailed) |
| `HDCRN21C` | Highest-degree school Carnegie classification (2021) |
| `BACRN21C` | Bachelor's school Carnegie classification (2021) |
| `EMSECSM` | Employer sector (summary) |
| `EMSIZE` | Employer size |
| `EMTP` | Employer type |
| `N2OCPRMG` | Principal-job occupation group (major) |
| `JOBSATIS` | Overall job satisfaction |
| `SATADV`, `SATSAL`, `SATSEC` | Satisfaction with advancement / salary / security |
| `ACTMGT`, `ACTRD`, `ACTTCH`, `ACTCAP` | Work activity: management-sales / research-dev / teaching / computer |

---

## scorecard_2026 — US Dept. of Education College Scorecard

- **Source:** US Department of Education.
- **URL:** `https://ed-public-download.scorecard.network/downloads/Most-Recent-Cohorts-All-Data-Elements.csv`
- **Cycle:** most-recent cohorts, released June 2026 (covers through 2025–26).
- **Grain:** institution. ~6,806 institutions; ~4,817 with a non-suppressed earnings value.
- **Terms:** US Government public domain.
- **Loader:** `scorecard_csv` (1,986 columns read as strings; suppressed values coerce to missing).

**Target:** `MD_EARN_WNE_P10` (median earnings of graduates 10 years after entry),
binarized at the median. Fallbacks if absent: `MD_EARN_WNE_P6`, `MN_EARN_WNE_P10`.

**Numeric features:** `ADM_RATE` (admission rate), `SAT_AVG`, `ACTCMMID` (ACT
midpoint), `UGDS` (undergraduate enrollment), `COSTT4_A` (average annual cost),
`TUITIONFEE_IN`, `TUITIONFEE_OUT`, `PCTPELL` (Pell share), `C150_4` (completion
rate), `UGDS_WHITE`, `UGDS_BLACK`, `UGDS_HISP`, `UGDS_ASIAN` (enrollment shares).

**Categorical features:** `CONTROL` (public / private nonprofit / for-profit),
`REGION`, `LOCALE` (urbanicity), `PREDDEG` (predominant degree), `HIGHDEG`
(highest degree), `CCSIZSET` (Carnegie size/setting), `STABBR` (state).

---

## acs_co_2024 — Census ACS PUMS 2024 (Colorado)

- **Source:** US Census Bureau, American Community Survey Public Use Microdata Sample.
- **URL:** `https://www2.census.gov/programs-surveys/acs/data/pums/2024/1-Year/csv_pco.zip` → member `psam_p08.csv`
- **Cycle:** 2024 1-year PUMS (newest available). Colorado; change the state file in the spec to switch.
- **Grain:** individual person.
- **Terms:** US Government public domain.
- **Loader:** `acs_zip` (person records read as strings).

**Target:** `WAGP` (wages/salary income, past 12 months), binarized at the median.
Filters: `SCHL ∈ {21,22,23,24}` (bachelor's or higher) and `ESR ∈ {1,2}` (employed).

**Numeric features:** `AGEP` (age), `WKHP` (usual hours worked per week).

**Categorical features:** `SCHL` (educational attainment), `FOD1P` (field of
degree), `OCCP` (occupation), `SEX`, `RAC1P` (race), `MAR` (marital status),
`CIT` (citizenship), `COW` (class of worker), `HISP` (Hispanic origin).

---

## pisa_2022 — OECD PISA 2022 student questionnaire

- **Source:** OECD Programme for International Student Assessment.
- **URL:** `https://webfs.oecd.org/pisa2022/STU_QQQ_SPSS.zip` (SPSS `.sav`, ~600 MB).
- **Cycle:** 2022 (newest; PISA is triennial).
- **Grain:** student (~600k, across ~80 education systems).
- **Terms:** OECD, freely available for research use.
- **Loader:** `pisa_sav` (requires `pyreadstat`; reads metadata first and keeps
  only requested columns that exist, so it tolerates codebook differences).

**Target:** `BSMJ` (expected occupational status at age 30, ISEI index),
binarized at the median. Fallback if absent: `PV1MATH`.

**Numeric features:** `ESCS` (economic/social/cultural status), `HOMEPOS` (home
possessions), `HISCED` (highest parental education), `BMMJ1` / `BFMJ2`
(mother's / father's occupational status), `PV1MATH` / `PV1READ` / `PV1SCIE`
(proficiency plausible values), `BELONG` (sense of belonging), `GROSAGR`
(growth mindset), `WORKMAST` (work mastery).

**Categorical features:** `ST004D01T` (gender), `IMMIG` (immigrant status),
`GRADE`, `REPEAT` (grade repetition), `CNT` (country/economy).

Variable names follow the OECD PISA 2022 codebook. This is the largest download
in the project; remove `pisa_2022` from `conf/experiments.yml` to skip it.
