import pandas as pd
import argparse
import os
from boruta import BorutaPy


from sklearn.ensemble import RandomForestClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-o', '--output', type=str, help="Output dir for images and compressed data", required=True)
    parser.add_argument('-t', '--train_data_name', type=str, help='Training dataset name', required=True)
    args = parser.parse_args()

    train = pd.read_csv(args.train_data_name, compression="zip")
    y_train = train.pop('target')
    print(f"Train shape {train.shape}")

    rf = RandomForestClassifier(n_jobs=-2, class_weight='balanced', max_depth=10, random_state=1234)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=1234)
    feat_selector.fit(train.values, y_train.values)

    selected_features = train.columns[feat_selector.support_].tolist() + train.columns[feat_selector.support_weak_].tolist()
    train_filtered = train[selected_features]
    train_filtered['target'] = y_train
    train_filtered.to_csv(os.path.join(args.output, "data_boruta.csv.zip"), index=False, compression="zip")

# -t ./dataset/bnp/data_cleaned.csv.zip -o ./dataset/bnp