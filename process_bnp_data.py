import argparse
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder
from feature_engine.imputation import CategoricalImputer
from feature_engine.selection import (
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures,
    SmartCorrelatedSelection,
    SelectByShuffling,
    SelectBySingleFeaturePerformance,
    RecursiveFeatureElimination,
)


# Function to convert to hexavigesimal base
def az_to_int(az, nanVal=None):
    if az == az:  #catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az


def clean_data(df):
    df.dropna(subset=['target'], inplace=True)
    df.drop(columns='ID', inplace=True)
    df['v22'] = df['v22'].apply(az_to_int)
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    con_cols = train.select_dtypes(include=['number']).columns.tolist()
    num_missing_imputer = SimpleImputer(strategy='median')
    cat_missing_imputer = CategoricalImputer(fill_value='__MISS__')
    rare_label_encoder = RareLabelEncoder(tol=0.01, n_categories=10, replace_with='__OTHER__')
    cat_freq_encoder = CountFrequencyEncoder(encoding_method="frequency")
    df[con_cols] = num_missing_imputer.fit_transform(df[con_cols])
    df[cat_cols] = cat_missing_imputer.fit_transform(df[cat_cols])
    df[cat_cols] = rare_label_encoder.fit_transform(df[cat_cols])
    df[cat_cols] = cat_freq_encoder.fit_transform(df[cat_cols])
    return df


if __name__ == '__main__':
    train = pd.read_csv("./dataset/bnp/train.csv.zip", compression="zip")
    print(f"Train shape {train.shape}")
    train_filtered = clean_data(train)
    train_filtered.to_csv("./dataset/bnp/data_cleaned.csv.zip", index=False, compression="zip")
