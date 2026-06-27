"""
data_utils.py - 数据下载、探索性分析、数据集加载
支持数据集: afqmc, lcqmc
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset
from transformers import BertTokenizer
import argparse

# ---------- 数据下载 ----------
def download_dataset(name, data_dir='wk8/data'):
    """下载指定数据集到 data_dir/name/ 下"""
    os.makedirs(f"{data_dir}/{name}", exist_ok=True)
    if name == 'afqmc':
        dataset = load_dataset('clue', 'afqmc')
        for split in ['train', 'validation']:
            df = dataset[split].to_pandas()
            pos = (df['label'] == 1).sum()
            neg = (df['label'] == 0).sum()
            print(f"{name} {split:12}: {len(df):7,} 条  正样本 {pos:6,}  负样本 {neg:6,}")
            df.to_csv(f"{data_dir}/{name}/{split}.csv", index=False)
        # test 无标签
        test_df = dataset['test'].to_pandas()
        test_df[['sentence1', 'sentence2']].to_csv(f"{data_dir}/{name}/test.csv", index=False)
    elif name == 'lcqmc':
        dataset = load_dataset('lcqmc')
        for split in ['train', 'validation', 'test']:
            df = dataset[split].to_pandas()
            if split != 'test':
                pos = (df['label'] == 1).sum()
                neg = (df['label'] == 0).sum()
                print(f"{name} {split:12}: {len(df):7,} 条  正样本 {pos:6,}  负样本 {neg:6,}")
                df.to_csv(f"{data_dir}/{name}/{split}.csv", index=False)
            else:
                df[['sentence1', 'sentence2']].to_csv(f"{data_dir}/{name}/test.csv", index=False)
    print(f"{name} 下载完成！")

# ---------- 数据探索（绘图） ----------
def explore_dataset(dataset_name, data_dir='../data', output_dir='../outputs/figures'):
    """对指定数据集绘制 4 张探索图"""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(f"{data_dir}/{dataset_name}/validation.csv")
    tokenizer = BertTokenizer.from_pretrained('wk/models/bert-base-chinese')

    # 1. 标签分布
    plt.figure()
    sns.countplot(x='label', data=df)
    plt.title(f'{dataset_name} Label Distribution')
    plt.savefig(f"{output_dir}/{dataset_name}_label_distribution.png")
    plt.close()

    # 2. 字符长度分布
    lens1 = df['sentence1'].str.len()
    lens2 = df['sentence2'].str.len()
    all_lens = pd.concat([lens1, lens2])
    plt.figure()
    sns.histplot(all_lens, bins=50)
    plt.axvline(32, color='r', linestyle='--', label='max_length=32')
    plt.title(f'{dataset_name} Character Length Distribution')
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_name}_char_length.png")
    plt.close()

    # 3. 长度差分布（按标签）
    diff = (df['sentence1'].str.len() - df['sentence2'].str.len()).abs()
    plt.figure()
    sns.histplot(diff, bins=30, hue=df['label'], multiple='stack')
    plt.title(f'{dataset_name} Length Difference by Label')
    plt.savefig(f"{output_dir}/{dataset_name}_length_diff.png")
    plt.close()

    # 4. Token 长度分布（用 BERT tokenizer）
    token_lens = []
    for text in df['sentence1'].tolist() + df['sentence2'].tolist():
        token_lens.append(len(tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)))
    plt.figure()
    sns.histplot(token_lens, bins=50)
    plt.axvline(64, color='r', linestyle='--', label='max_length=64')
    plt.title(f'{dataset_name} Token Length Distribution')
    plt.legend()
    plt.savefig(f"{output_dir}/{dataset_name}_token_length.png")
    plt.close()
    print(f"探索图表已保存至 {output_dir}/ (前缀 {dataset_name})")

# ---------- 命令行接口 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true', help='下载两个数据集')
    parser.add_argument('--explore', type=str, default=None, choices=['afqmc','lcqmc'], help='探索哪个数据集')
    args = parser.parse_args()

    if args.download:
        download_dataset('afqmc')
        download_dataset('lcqmc')
    if args.explore:
        explore_dataset(args.explore)