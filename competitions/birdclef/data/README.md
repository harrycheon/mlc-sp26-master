# BirdCLEF 2026 Data

Download from Kaggle: `kaggle competitions download -c birdclef-2026`

After downloading and unzipping, this directory should contain:

```
data/
  train_soundscapes/                 # Labeled soundscape audio (.ogg, 32kHz)
  train_soundscapes_labels.csv       # Labels: filename, start, end, primary_label
  taxonomy.csv                       # Species list (primary_label column)
  train_audio/                       # Individual species recordings (.ogg)
  train.csv                          # Metadata for train_audio
  sample_submission.csv              # Submission format reference
```

Audio files are OGG format at 32kHz. Test soundscapes are 1 minute each,
scored in 5-second windows. This is a multi-label task.

Do NOT commit data files to the repository.
