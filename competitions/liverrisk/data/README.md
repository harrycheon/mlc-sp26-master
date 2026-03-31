# LiverRisk (ANNITIA Challenge) Data

Download from Trustii: https://app.trustii.io/datasets/1551

Place the following files in this directory (`competitions/liverrisk/data/`):

```
data/
  train.csv              # 1253 patients with features + outcome columns
  test.csv               # 423 patients with features only (+ trustii_id)
  sample_submission.csv  # Expected output: trustii_id, risk_hepatic_event, risk_death
  dictionary.csv         # Column descriptions
```

Outcome columns in train.csv: evenements_hepatiques_majeurs,
evenements_hepatiques_age_occur, death, death_age_occur.

Score = 0.3 * C-index(death) + 0.7 * C-index(hepatic)

Per competition rules, all data is strictly confidential.
Do NOT commit data files to the repository.
