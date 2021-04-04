import argparse
import os
import pandas as pd
from sklearn.impute import SimpleImputer
from feature_engine.encoding import RareLabelEncoder, CountFrequencyEncoder
from feature_engine.imputation import CategoricalImputer
from sklearn.preprocessing import MinMaxScaler
from feature_engine.selection import (
    DropFeatures,
    DropConstantFeatures,
    DropDuplicateFeatures,
    DropCorrelatedFeatures
)
from feature_engine.outliers import OutlierTrimmer, Winsorizer
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt


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


def clean_data(X):
    X.dropna(subset=['target'], inplace=True)
    y = X.pop('target')
    X.drop(columns='ID', inplace=True)
    X['v22'] = X['v22'].apply(az_to_int)
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    con_cols = X.select_dtypes(include=['number']).columns.tolist()
    num_missing_imputer = SimpleImputer(strategy='median')
    cat_missing_imputer = CategoricalImputer(fill_value='__MISS__')
    rare_label_encoder = RareLabelEncoder(tol=0.01, n_categories=10, replace_with='__OTHER__')
    cat_freq_encoder = CountFrequencyEncoder(encoding_method="frequency")
    X[con_cols] = num_missing_imputer.fit_transform(X[con_cols])
    X[cat_cols] = cat_missing_imputer.fit_transform(X[cat_cols])
    X[cat_cols] = rare_label_encoder.fit_transform(X[cat_cols])
    X[cat_cols] = cat_freq_encoder.fit_transform(X[cat_cols])
    # more cleaning
    trimmer = Winsorizer(capping_method='quantiles', tail='both', fold=0.005)
    X = trimmer.fit_transform(X)
    undersampler = RandomUnderSampler(sampling_strategy=0.7, random_state=1234)
    X, Y = undersampler.fit_resample(X, y)
    quasi_constant = DropConstantFeatures(tol=0.998)
    X = quasi_constant.fit_transform(X)
    print(f"Quasi Features to drop {quasi_constant.features_to_drop_}")
    # Remove duplicated features¶
    duplicates = DropDuplicateFeatures()
    X = duplicates.fit_transform(X)
    print(f"Duplicate feature sets {duplicates.duplicated_feature_sets_}")
    print(f"Dropping duplicate features {duplicates.features_to_drop_}")
    drop_corr = DropCorrelatedFeatures(method="pearson", threshold=0.95, missing_values="ignore")
    X = drop_corr.fit_transform(X)
    print(f"Drop correlated feature sets {drop_corr.correlated_feature_sets_}")
    print(f"Dropping correlared features {drop_corr.features_to_drop_}")
    X['target'] = Y
    return X


if __name__ == '__main__':
    train = pd.read_csv("./dataset/bnp/train.csv.zip", compression="zip")
    print(f"Train shape {train.shape}")
    train_filtered = clean_data(train)
    train_filtered.to_csv("./dataset/bnp/data_cleaned.csv.zip", index=False, compression="zip")
