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



def process_data(X):
    #clean data
    y = X.pop("TARGET")
    X['var3'].replace("-999999", -1, inplace=True)
    #remove constant feature
    trimmer = Winsorizer(capping_method='quantiles', tail='both', fold=0.005)
    X = trimmer.fit_transform(X)
    undersampler = RandomUnderSampler(sampling_strategy=0.7, random_state=1234)
    X, Y = undersampler.fit_resample(X, y)
    drop_features = DropFeatures(features_to_drop=['ID'])
    X = drop_features.fit_transform(X)
    quasi_constant = DropConstantFeatures(tol=0.998)
    X = quasi_constant.fit_transform(X)
    print(f"Quasi Features to drop {quasi_constant.features_to_drop_}")
    # Remove duplicated features¶
    duplicates = DropDuplicateFeatures()
    X = duplicates.fit_transform(X)
    print(f"Duplicate feature sets {duplicates.duplicated_feature_sets_}")
    print(f"Dropping duplicate features {duplicates.features_to_drop_}")
    drop_corr = DropCorrelatedFeatures(method="pearson", threshold=0.9, missing_values="ignore")
    X = drop_corr.fit_transform(X)
    print(f"Drop correlated feature sets {drop_corr.correlated_feature_sets_}")
    print(f"Dropping correlared features {drop_corr.features_to_drop_}")
    X['target'] = Y
    return X


if __name__ == '__main__':
    df = pd.read_csv("./dataset/santander/train.csv.zip", compression="zip")
    print(f"Train shape {df.shape}")
    df_filtered = process_data(df)
    print(f"Processed Train shape {df_filtered.shape}")
    df_filtered.to_csv("./dataset/santander/data_cleaned.csv.zip", index=False, compression="zip")
