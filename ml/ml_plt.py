# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 18:47:50 2026

@author: weiru
"""

import os
import itertools
import matplotlib
import pandas as pd
from collections import Counter
from venn import venn
import seaborn as sns
from sklearn import metrics
import matplotlib.pyplot as plt  
from upsetplot import query, generate_counts, from_contents, UpSet, plot
import warnings
warnings.filterwarnings('ignore') 

des_all = ['A', 'B', 'C', 'D', 'E']
methods = ['rf', 'xg', 'lg', 'svm']
target = 'AKT3'
library = 'FDA'

BASE_OUTPUT_DIR = "..\\Results_QSAR\\"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def get_id_col(df):
    possible_cols = ['Name', 'ID', 'Molecule_ID', 'Compound_ID', 'SMILES']
    for col in possible_cols:
        if col in df.columns:
            return col
    return df.columns[3] if len(df.columns) >= 4 else df.columns[0]


def plotAllRoc():
    title_names = {'rf': 'RF', 'xg': 'XGB', 'lg': 'LGB', 'svm': 'SVM'}
    fprs = dict()
    tprs = dict()
    roc_aucs = dict()
    
    color_codes = ["#6CBAD8", "#EF8881", "#519D78", "#F9B063", "#BA7FB5"]
    
    for model in methods:
        roc_dir = os.path.join(BASE_OUTPUT_DIR, f"ROC_Data_{model}", target)
        for des in des_all:
            roc_file = os.path.join(roc_dir, f"{target}_{des}_ROC_Data.csv")
            if not os.path.exists(roc_file):
                continue
                
            data = pd.read_csv(roc_file)
            y_test = data['y_true']
            y_score = data['y_pred_proba']
            
            fpr, tpr, _ = metrics.roc_curve(y_test, y_score, pos_label=1)
            roc_auc = metrics.auc(fpr, tpr)
            
            fprs[(title_names[model], des)] = fpr
            tprs[(title_names[model], des)] = tpr
            roc_aucs[(title_names[model], des)] = roc_auc
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, model in enumerate(methods):
        ax = axes[i]
        for des, color in zip(des_all, color_codes):
            key = (title_names[model], des)
            if key in fprs:
                ax.plot(fprs[key], tprs[key], color=color, lw=2,
                        label=f'{des.upper()} AUC=%.2f' % roc_aucs[key])
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.tick_params(labelsize=18)
        ax.set_xlabel('False Positive Rate', fontsize=18)
        ax.set_ylabel('True Positive Rate', fontsize=18)
        ax.set_title(f'ROC Curves for {title_names[model]}', fontsize=22)
        ax.legend(loc="lower right", fontsize=14)
    
    plt.tight_layout()
    plt.savefig('roc_comparison_subplots.png', dpi=600)
    plt.show()

def plotUpset(model, min_intersection_size=2):
    pred_dir = os.path.join(BASE_OUTPUT_DIR, f"Prediction_{model}", target)
    
    cas_dict = {}
    last_pred_df = None
    id_col = None

    for des in des_all:
        pred_file = os.path.join(pred_dir, f"{library}_{des}_prediction.csv")
        if not os.path.exists(pred_file):
            continue
            
        pred_data = pd.read_csv(pred_file)
        last_pred_df = pred_data
        id_col = get_id_col(pred_data)
        
        hit_data = pred_data[pred_data['Pred_label'] == 1]
        cas_dict[des.upper()] = set(hit_data[id_col])
    
    if not cas_dict:
        return

    cas_tuble = from_contents(cas_dict)
    
    figsize = (10, 7)
    fig = plt.figure(figsize=figsize)
    cas_upset = UpSet(
        cas_tuble, 
        subset_size='count',
        show_counts=False,
        sort_by='degree',
        intersection_plot_elements=10
    )
    
    color_codes = ["#6CBAD8", "#EF8881", "#519D78", "#F9B063", "#BA7FB5"]
    cas_upset.style_categories(list(cas_dict.keys()), bar_facecolor="gray")
    for degree_idx in range(1, len(des_all) + 1):
        color = color_codes[(degree_idx - 1) % len(color_codes)]
        cas_upset.style_subsets(min_degree=degree_idx, facecolor=color)
    
    cas_upset.plot(fig=fig)
    
    for ax in fig.axes:
        ax.tick_params(axis='both', labelsize=16)
        if ax.get_xlabel():
            ax.set_xlabel(ax.get_xlabel(), fontsize=18)
        if ax.get_ylabel():
            ax.set_ylabel(ax.get_ylabel(), fontsize=18)
        for text in ax.texts:
            text.set_fontsize(16)

    plt.savefig(f"{model}_upset.png", dpi=600)
    plt.show()

    all_intersections = set()
    cas_lists = list(cas_dict.values())
    
    for i in range(min_intersection_size, len(cas_lists) + 1):
        for comb in itertools.combinations(cas_lists, i):
            intersection = set.intersection(*comb)
            all_intersections.update(intersection)
        
    if last_pred_df is not None and id_col in last_pred_df.columns:
        pred_res = last_pred_df[last_pred_df[id_col].isin(all_intersections)]
        out_path = os.path.join(pred_dir, f"{library}_{model}.csv")
        pred_res.to_csv(out_path, index=False)

def plotVenn(min_overlap=2):
    cas_dict = {}
    title_names = {'rf': 'RF', 'xg': 'XGB', 'lg': 'LGB', 'svm': 'SVM'}
    all_molecules_dict = {}
    id_col = None

    for model in methods:
        pred_dir = os.path.join(BASE_OUTPUT_DIR, f"Prediction_{model}", target)
        pred_file = os.path.join(pred_dir, f"{library}_{model}.csv")
        
        if not os.path.exists(pred_file):
            continue
            
        pred_data = pd.read_csv(pred_file)
        if id_col is None:
            id_col = get_id_col(pred_data)
        
        mol_ids = set(pred_data[id_col].dropna().astype(str))
        cas_dict[title_names[model]] = mol_ids

        for _, row in pred_data.iterrows():
            m_id = str(row[id_col])
            if m_id not in all_molecules_dict:
                all_molecules_dict[m_id] = row.to_dict()
    
    if len(cas_dict) < 2:
        return

    cmap = ["#6CBAD8", "#EF8881", "#519D78", "#F9B063", "#BA7FB5"]
    venn(cas_dict, cmap=cmap, fontsize=14, legend_loc="upper right")
    plt.savefig('high_res_venn.png', dpi=600, bbox_inches='tight', pad_inches=0.1)
    plt.show()
    
    all_hits = []
    for model_hits in cas_dict.values():
        all_hits.extend(list(model_hits))
    
    hit_counts = Counter(all_hits)
    final_intersection = {mol_id for mol_id, count in hit_counts.items() if count >= min_overlap}

    matched_rows = [all_molecules_dict[m_id] for m_id in final_intersection if m_id in all_molecules_dict]
    pred_res = pd.DataFrame(matched_rows)

    docking_dir = "..\\Docking\\"
    ensure_dir(docking_dir)
    
    out_filename = "ts_all_methods.csv" if min_overlap == len(cas_dict) else f"ts_overlap_gte_{min_overlap}.csv"
    docking_out_path = os.path.join(docking_dir, out_filename)
    pred_res.to_csv(docking_out_path, index=False)
    print(f"[Venn]  ({len(pred_res)} ) saved: {docking_out_path}")

def plotConsensusAnalysis(top_k_highlight=50):
    consensus_dir = os.path.join(BASE_OUTPUT_DIR, "Consensus_Analysis")
    
    rank_file = os.path.join(consensus_dir, f"{target}_Consensus_Rankings.csv")
    spearman_file = os.path.join(consensus_dir, f"{target}_Spearman_Correlation.csv")
    jaccard_file = os.path.join(consensus_dir, f"{target}_Jaccard_Similarity.csv")

    if not (os.path.exists(rank_file) and os.path.exists(spearman_file) and os.path.exists(jaccard_file)):
        print("[Error]  ml_BDes.py running analyze_consensus_and_ranks()")
        return

    df_rank = pd.read_csv(rank_file)
    id_col = get_id_col(df_rank)

    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(
        df_rank['Consensus_Score'], 
        df_rank['Rank_Std'], 
        c=df_rank['Selection_Frequency'], 
        cmap='viridis', 
        alpha=0.8, 
        edgecolors='w', 
        s=60
    )
    cbar = plt.colorbar(scatter)
    cbar.set_label('Selection Frequency (Top-N)', fontsize=14)

    top_candidates = df_rank.head(top_k_highlight)
    for _, row in top_candidates.iterrows():
        if row['Selection_Frequency'] >= df_rank['Selection_Frequency'].max() * 0.8:
            plt.annotate(
                str(row[id_col]), 
                (row['Consensus_Score'], row['Rank_Std']),
                fontsize=8, alpha=0.85, xytext=(3, 3), textcoords='offset points'
            )

    plt.xlabel('Consensus Score (Mean Probability)', fontsize=16)
    plt.ylabel('Rank Instability (Rank Std)', fontsize=16)
    plt.title('Candidate Screening: Probability Consensus vs Rank Stability', fontsize=18)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('consensus_vs_stability.png', dpi=600)
    plt.show()

    df_spearman = pd.read_csv(spearman_file, index_col=0)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df_spearman, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, cbar=True)
    plt.title('Spearman Rank Correlation Across Models & Descriptors', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('spearman_correlation_heatmap.png', dpi=600)
    plt.show()

    df_jaccard = pd.read_csv(jaccard_file, index_col=0)
    plt.figure(figsize=(11, 9))
    sns.heatmap(df_jaccard, annot=True, fmt=".2f", cmap='YlGnBu', vmin=0, vmax=1, cbar=True)
    plt.title('Top-N Candidate Intersection (Jaccard Similarity)', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('jaccard_similarity_heatmap.png', dpi=600)
    plt.show()


if __name__ == '__main__':
    plotAllRoc()
    
    for method in methods:
        plotUpset(method, min_intersection_size=2)
        
    plotVenn(min_overlap=2)
    
    plotConsensusAnalysis(top_k_highlight=30)