import numpy as np
import pandas as pd
from hashlib import sha1
from sklearn.model_selection import train_test_split

# Load
df = pd.read_csv('k562_crispr_eg_features.csv')

# Replace reference with reference ID
for ref in np.unique(df["Reference"].values):
    hashid = str(sha1(ref.encode('utf-8')).hexdigest()[:8])
    df = df.replace(ref, hashid)

df = df.rename(columns={"Reference": "ReferenceID"})

# Split by label
df_pos = df[df["Regulated"]]
df_neg = df[~df["Regulated"]]

# Stratified train/test split equally by chromosome
X_pos_train, X_pos_test = train_test_split(df_pos, test_size=0.1, random_state=0, stratify=df_pos['chr'])
print("y=+1 ", "train shape: ", X_pos_train.shape, "test shape: " , X_pos_test.shape)

X_neg_train, X_neg_test = train_test_split(df_neg, test_size=0.1, random_state=1, stratify=df_neg['chr'])
X_neg_train.shape, X_neg_test.shape
print("y=-1 ", "train shape: ", X_neg_train.shape, "test shape: " , X_neg_test.shape)

# Recombine dataframes
X_train = pd.concat((X_pos_train, X_neg_train))
X_test = pd.concat((X_pos_test, X_neg_test))

y_train = X_train["Regulated"]
y_test = X_test["Regulated"]

# Drop labels from X
del X_train["Regulated"]
del X_test["Regulated"]

# Write to csv
X_train.to_csv("X_train.csv", index=False)
X_test.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
