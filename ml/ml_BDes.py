# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:47:50 2026

@author: weiru
"""

import os
import joblib
import multiprocessing
import numpy as np
import pandas as pd
from scipy import stats
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.combine import SMOTEENN

import warnings
warnings.filterwarnings('ignore') 

RANDOM_SEED = 3154
np.random.seed(RANDOM_SEED)

des_all = ['A', 'B', 'C', 'D', 'E']
methods = ['rf', 'xg', 'lg', 'svm']
target = 'AKT3'
library = 'FDA'

N_JOBS = -1

BASE_OUTPUT_DIR = ""

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def generate_scaffold(smiles, include_chirality=False):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except:
        return ""

def scaffold_split(df, smiles_col='SMILES', test_size=0.2):
    scaffolds = {}
    for idx, row in df.iterrows():
        scaffold = generate_scaffold(row[smiles_col])
        scaffolds.setdefault(scaffold, []).append(idx)
    
    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    
    train_indices, test_indices = [], []
    total_samples = len(df)
    test_target_count = total_samples * test_size
    
    for group in scaffold_sets:
        if len(test_indices) + len(group) <= test_target_count:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
            
    return df.iloc[train_indices].index, df.iloc[test_indices].index

def getData(des):
    target_path = f"..\\Descriptors\\{target}\\"
    target_name = f"{target}_{des}.csv"
    target_data = pd.read_csv(target_path + target_name, encoding="gb18030")
    target_data['label'] = (target_data['Standard Value'] < 10000).astype(int)
    target_data.reset_index(drop=True, inplace=True)
    
    X = target_data.iloc[:, -641:-1]
    y = target_data['label']
    
    library_path = f"..\\Descriptors\\{library}\\"
    library_name = f"{library}_{des}.csv"
    library_data = pd.read_csv(library_path + library_name, encoding="gb18030")
    
    return X, y, target_data, library_data

def get_pipeline_and_param_grid(model_name):
    if model_name == 'rf':
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('resample', SMOTEENN(random_state=RANDOM_SEED)),
            ('classifier', RandomForestClassifier(random_state=RANDOM_SEED))
        ])
        param_grid = {
            'classifier__n_estimators': [100, 300, 500],
            'classifier__max_depth': [None, 10, 20]
        }
    elif model_name == 'xg':
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('resample', SMOTEENN(random_state=RANDOM_SEED)),
            ('classifier', XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss'))
        ])
        param_grid = {
            'classifier__n_estimators': [200, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [4, 6]
        }
    elif model_name == 'lg':
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('resample', SMOTEENN(random_state=RANDOM_SEED)),
            ('classifier', LGBMClassifier(random_state=RANDOM_SEED, verbose=-1))
        ])
        param_grid = {
            'classifier__n_estimators': [200, 500],
            'classifier__learning_rate': [0.01, 0.05, 0.1]
        }
    elif model_name == 'svm':
        pipe = ImbPipeline([
            ('scaler', StandardScaler()),
            ('resample', SMOTEENN(random_state=RANDOM_SEED)),
            ('classifier', SVC(probability=True, random_state=RANDOM_SEED))
        ])
        param_grid = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto']
        }
    return pipe, param_grid

def get_confidence_interval_val(data, confidence=0.95):
    data = np.array(data)
    mean = np.mean(data)
    n = len(data)
    if n < 2:
        return mean, 0.0, mean, mean
    
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2., n - 1)
    return mean, h, mean - h, mean + h

def evaluate_model(model, X_test, y_test):
    y_predict = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    confusion = metrics.confusion_matrix(y_test, y_predict)
    TP = confusion[1, 1] if confusion.shape == (2, 2) else 0
    TN = confusion[0, 0] if confusion.shape == (2, 2) else 0
    FP = confusion[0, 1] if confusion.shape == (2, 2) else 0
    FN = confusion[1, 0] if confusion.shape == (2, 2) else 0

    sen = TP / (TP + FN) if (TP + FN) > 0 else 0
    spe = TN / (TN + FP) if (TN + FP) > 0 else 0
    auc = metrics.roc_auc_score(y_test, y_proba)
    acc = metrics.accuracy_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict, zero_division=0)
    pre = metrics.precision_score(y_test, y_predict, zero_division=0)
    log_loss_val = metrics.log_loss(y_test, y_proba)

    res = {
        'ACC': acc, 'SPE': spe, 'SEN': sen, 'PRE': pre, 
        'F1': f1, 'AUC': auc, 'Log_Loss': log_loss_val
    }
    return res, y_predict, y_proba

def run_pipeline(model_name, retrain=False):
    score_dir = os.path.join(BASE_OUTPUT_DIR, f"Score_{model_name}", target)
    model_dir = os.path.join(BASE_OUTPUT_DIR, f"Models_{model_name}", target)
    pred_dir = os.path.join(BASE_OUTPUT_DIR, f"Prediction_{model_name}", target)
    roc_dir = os.path.join(BASE_OUTPUT_DIR, f"ROC_Data_{model_name}", target)
    
    for d in [score_dir, model_dir, pred_dir, roc_dir]:
        ensure_dir(d)

    for des in des_all:
        print(f"\n==================== Running {model_name.upper()} | Descriptor: {des} ====================")
        X, y, target_data, library_data = getData(des)

        if 'SMILES' in target_data.columns:
            train_idx, test_idx = scaffold_split(target_data, smiles_col='SMILES', test_size=0.2)
            X_train_cv, X_test_holdout = X.loc[train_idx], X.loc[test_idx]
            y_train_cv, y_test_holdout = y.loc[train_idx], y.loc[test_idx]
        else:
            X_train_cv, X_test_holdout, y_train_cv, y_test_holdout = train_test_split(
                X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
            )

        model_filepath = os.path.join(model_dir, f"{target}_{des}_best_model.joblib")

        if os.path.exists(model_filepath) and not retrain:
            print(f"[LOAD] saved model: {model_filepath}")
            best_model = joblib.load(model_filepath)
        else:
            pipe, param_grid = get_pipeline_and_param_grid(model_name)
            cv_outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
            cv_fold_scores = []

            for fold_idx, (train_fold_idx, val_fold_idx) in enumerate(cv_outer.split(X_train_cv, y_train_cv)):
                X_tr, X_val = X_train_cv.iloc[train_fold_idx], X_train_cv.iloc[val_fold_idx]
                y_tr, y_val = y_train_cv.iloc[train_fold_idx], y_train_cv.iloc[val_fold_idx]

                grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=N_JOBS)
                grid.fit(X_tr, y_tr)
                
                fold_metrics, _, _ = evaluate_model(grid.best_estimator_, X_val, y_val)
                fold_metrics['Fold'] = fold_idx + 1
                cv_fold_scores.append(fold_metrics)

            cv_df = pd.DataFrame(cv_fold_scores)
            metric_cols = ['ACC', 'SPE', 'SEN', 'PRE', 'F1', 'AUC', 'Log_Loss']
            
            cv_summary_rows = []
            for col in metric_cols:
                mean_val, margin, ci_low, ci_high = get_confidence_interval_val(cv_df[col])
                cv_summary_rows.append({
                    'Metric': col, 'Mean': mean_val, 'CI_95_Margin': margin,
                    'CI_95_Lower': ci_low, 'CI_95_Upper': ci_high,
                    'Formatted': f"{mean_val:.4f} (±{margin:.4f})"
                })
            cv_summary_df = pd.DataFrame(cv_summary_rows)

            cv_df.to_csv(os.path.join(score_dir, f"{target}_{des}_10Fold_CV_Details.csv"), index=False)
            cv_summary_df.to_csv(os.path.join(score_dir, f"{target}_{des}_10Fold_CV_Summary_CI.csv"), index=False)

            final_grid = GridSearchCV(estimator=pipe, param_grid=param_grid, cv=5, scoring='roc_auc', n_jobs=N_JOBS)
            final_grid.fit(X_train_cv, y_train_cv)
            best_model = final_grid.best_estimator_

            joblib.dump(best_model, model_filepath)
            print(f"[SAVE] saved in: {model_filepath}")

        holdout_metrics, y_test_pred, y_test_proba = evaluate_model(best_model, X_test_holdout, y_test_holdout)
        holdout_df = pd.DataFrame([holdout_metrics])
        holdout_df.to_csv(os.path.join(score_dir, f"{target}_{des}_Holdout_Test_Score.csv"), index=False)

        roc_data_df = pd.DataFrame({
            'y_true': y_test_holdout.values,
            'y_pred_label': y_test_pred,
            'y_pred_proba': y_test_proba
        })
        roc_data_df.to_csv(os.path.join(roc_dir, f"{target}_{des}_ROC_Data.csv"), index=False)

        library_X = library_data.iloc[:, -640:]
        lib_preds = best_model.predict(library_X)
        lib_probas = best_model.predict_proba(library_X)[:, 1]

        res_df = library_data.iloc[:, :-640].copy()
        res_df['Pred_label'] = lib_preds
        res_df['Pred_proba'] = lib_probas
        res_df.to_csv(os.path.join(pred_dir, f"{library}_{des}_prediction.csv"), index=False)

        print(f"-> Hold-out Test AUC: {holdout_metrics['AUC']:.4f}")

def analyze_consensus_and_ranks(top_n=200):
    print("\n==================== consensus ====================")
    
    proba_dict = {}
    rank_dict = {}
    master_df = None
    id_col = None

    for model in methods:
        pred_dir = os.path.join(BASE_OUTPUT_DIR, f"Prediction_{model}", target)
        for des in des_all:
            pred_file = os.path.join(pred_dir, f"{library}_{des}_prediction.csv")
            if not os.path.exists(pred_file):
                continue
                
            df = pd.read_csv(pred_file)
            if id_col is None:
                possible_cols = ['Name', 'ID', 'Molecule_ID', 'SMILES']
                id_col = next((c for c in possible_cols if c in df.columns), df.columns[3] if len(df.columns)>=4 else df.columns[0])
                master_df = df[[c for c in df.columns if c not in ['Pred_label', 'Pred_proba']]].copy()

            key = f"{model.upper()}_{des.upper()}"
            proba_dict[key] = df['Pred_proba'].values
            rank_dict[key] = df['Pred_proba'].rank(ascending=False, method='min').values

    if not proba_dict:
        print("[Error] no predicted data")
        return

    proba_df = pd.DataFrame(proba_dict)
    rank_df = pd.DataFrame(rank_dict)

    master_df['Consensus_Score'] = proba_df.mean(axis=1)
    master_df['Mean_Rank'] = rank_df.mean(axis=1)
    master_df['Rank_Std'] = rank_df.std(axis=1)
    
    is_top_n = rank_df <= top_n
    master_df['Selection_Frequency'] = is_top_n.sum(axis=1)

    master_df = master_df.sort_values(by='Consensus_Score', ascending=False).reset_index(drop=True)
    master_df['Final_Consensus_Rank'] = master_df.index + 1

    consensus_dir = os.path.join(BASE_OUTPUT_DIR, "Consensus_Analysis")
    ensure_dir(consensus_dir)
    consensus_out_path = os.path.join(consensus_dir, f"{target}_Consensus_Rankings.csv")
    master_df.to_csv(consensus_out_path, index=False)
    print(f"[Consensus] saved: {consensus_out_path}")

    spearman_corr = proba_df.corr(method='spearman')
    spearman_corr.to_csv(os.path.join(consensus_dir, f"{target}_Spearman_Correlation.csv"))

    cols = proba_df.columns
    jaccard_mat = pd.DataFrame(index=cols, columns=cols, dtype=float)
    
    for c1 in cols:
        set1 = set(master_df.loc[rank_df[c1] <= top_n, id_col])
        for c2 in cols:
            set2 = set(master_df.loc[rank_df[c2] <= top_n, id_col])
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            jaccard_mat.loc[c1, c2] = intersection / union if union > 0 else 0.0

    jaccard_mat.to_csv(os.path.join(consensus_dir, f"{target}_Jaccard_Similarity.csv"))
    print(f"[Consensus] Jaccard Spearman")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    
    for method in methods:
        run_pipeline(model_name=method, retrain=True)
        
    analyze_consensus_and_ranks(top_n=200)