"""
analysis.py - 方法对比（三种方法在两个数据集上的结果）和错误分析
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from transformers import BertTokenizer
from train_eval import BiEncoder, CrossEncoder, PairDataset, CrossPairDataset
from train_eval import evaluate_biencoder, evaluate_crossencoder
from torch.utils.data import DataLoader
import json

# ---------- 方法对比 ----------
def compare_methods():
    """对比 BiEncoder(cosine/triplet) 和 CrossEncoder 在两个数据集上的表现"""
    datasets = ['afqmc', 'lcqmc']
    bert_path = 'wk8/models/bert-base-chinese'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(bert_path)

    all_results = {}
    for ds in datasets:
        print(f"\n========== Dataset: {ds} ==========")
        val_df = pd.read_csv(f"../data/{ds}/validation.csv")
        results = {}

        # BiEncoder Cosine
        ckpt_cos = f"../outputs/checkpoints/biencoder_cosine_{ds}_best.pt"
        if os.path.exists(ckpt_cos):
            ckpt = torch.load(ckpt_cos, map_location=device, weights_only=False)
            model = BiEncoder(bert_path, pool=ckpt['args']['pool'],
                              num_hidden_layers=ckpt['args']['num_hidden_layers'])
            model.load_state_dict(ckpt['state_dict'])
            model.to(device)
            val_set = PairDataset(val_df, tokenizer, max_len=64)
            loader = DataLoader(val_set, batch_size=64, shuffle=False)
            acc, f1, th = evaluate_biencoder(model, loader, device)
            results['BiEncoder_Cosine'] = {'acc': acc, 'f1': f1, 'extra': f'th={th:.2f}'}

        # BiEncoder Triplet
        ckpt_tri = f"../outputs/checkpoints/biencoder_triplet_{ds}_best.pt"
        if os.path.exists(ckpt_tri):
            ckpt = torch.load(ckpt_tri, map_location=device, weights_only=False)
            model = BiEncoder(bert_path, pool=ckpt['args']['pool'],
                              num_hidden_layers=ckpt['args']['num_hidden_layers'])
            model.load_state_dict(ckpt['state_dict'])
            model.to(device)
            val_set = PairDataset(val_df, tokenizer, max_len=64)
            loader = DataLoader(val_set, batch_size=64, shuffle=False)
            acc, f1, th = evaluate_biencoder(model, loader, device)
            results['BiEncoder_Triplet'] = {'acc': acc, 'f1': f1, 'extra': f'th={th:.2f}'}

        # CrossEncoder
        ckpt_ce = f"../outputs/checkpoints/crossencoder_{ds}_best.pt"
        if os.path.exists(ckpt_ce):
            ckpt = torch.load(ckpt_ce, map_location=device, weights_only=False)
            model = CrossEncoder(bert_path, num_hidden_layers=ckpt['args']['num_hidden_layers'])
            model.load_state_dict(ckpt['state_dict'])
            model.to(device)
            val_set = CrossPairDataset(val_df, tokenizer, max_len=128)
            loader = DataLoader(val_set, batch_size=64, shuffle=False)
            acc, f1 = evaluate_crossencoder(model, loader, device)
            results['CrossEncoder'] = {'acc': acc, 'f1': f1, 'extra': 'argmax'}

        # 打印结果
        print(f"{'Method':<20} Accuracy  F1       Extra")
        for name, res in results.items():
            print(f"{name:<20} {res['acc']:.4f}   {res['f1']:.4f}   {res['extra']}")

        all_results[ds] = results

        # 绘制柱状图
        names = list(results.keys())
        accs = [results[n]['acc'] for n in names]
        f1s = [results[n]['f1'] for n in names]
        x = np.arange(len(names))
        plt.figure(figsize=(8,5))
        plt.bar(x-0.2, accs, width=0.4, label='Accuracy')
        plt.bar(x+0.2, f1s, width=0.4, label='F1')
        plt.xticks(x, names, rotation=15)
        plt.ylabel('Score')
        plt.legend()
        plt.title(f'Comparison on {ds.upper()}')
        plt.tight_layout()
        plt.savefig(f"../outputs/figures/compare_{ds}.png")
        plt.close()
        print(f"对比图已保存至 ../outputs/figures/compare_{ds}.png")

    # 保存汇总结果
    with open('../outputs/logs/comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)

# ---------- 错误分析 ----------
def analyze_badcases(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = BertTokenizer.from_pretrained(args.bert_path)
    df = pd.read_csv(f"../data/{args.dataset}/validation.csv")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = BiEncoder(args.bert_path, pool=ckpt['args']['pool'],
                      num_hidden_layers=ckpt['args']['num_hidden_layers'])
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    threshold = ckpt['threshold']

    val_set = PairDataset(df, tokenizer, max_len=64)
    loader = DataLoader(val_set, batch_size=64, shuffle=False)
    all_labels, all_sims = [], []
    with torch.no_grad():
        for batch in loader:
            emb1 = model.encode(batch['input_ids1'].to(device),
                                batch['attn1'].to(device),
                                batch['type1'].to(device))
            emb2 = model.encode(batch['input_ids2'].to(device),
                                batch['attn2'].to(device),
                                batch['type2'].to(device))
            sim = torch.cosine_similarity(emb1, emb2).cpu().numpy()
            all_sims.extend(sim)
            all_labels.extend(batch['label'].cpu().numpy())
    all_sims = np.array(all_sims)
    all_labels = np.array(all_labels)
    preds = (all_sims >= threshold).astype(int)

    fp_idx = np.where((preds == 1) & (all_labels == 0))[0]
    fn_idx = np.where((preds == 0) & (all_labels == 1))[0]
    print(f"FP (假阳性) {len(fp_idx)} 条, FN (假阴性) {len(fn_idx)} 条")

    # 计算 Jaccard 相似度
    def jaccard(s1, s2):
        set1, set2 = set(s1), set(s2)
        inter = len(set1 & set2)
        union = len(set1 | set2)
        return inter / union if union > 0 else 0

    df['jaccard'] = df.apply(lambda r: jaccard(str(r['sentence1']), str(r['sentence2'])), axis=1)
    fp_jacc = df.iloc[fp_idx]['jaccard'].mean() if len(fp_idx) else 0
    fn_jacc = df.iloc[fn_idx]['jaccard'].mean() if len(fn_idx) else 0
    print(f"FP 平均 Jaccard = {fp_jacc:.3f}, FN 平均 Jaccard = {fn_jacc:.3f}")

    # 打印前5个案例
    print("\n--- FP 示例 ---")
    for i in fp_idx[:5]:
        row = df.iloc[i]
        print(f"{row['sentence1']} | {row['sentence2']} (sim={all_sims[i]:.3f})")
    print("\n--- FN 示例 ---")
    for i in fn_idx[:5]:
        row = df.iloc[i]
        print(f"{row['sentence1']} | {row['sentence2']} (sim={all_sims[i]:.3f})")

    # 绘制分数分布图
    plt.figure(figsize=(8,5))
    correct = (preds == all_labels)
    plt.hist(all_sims[correct], bins=30, alpha=0.6, label='Correct', color='green')
    plt.hist(all_sims[~correct], bins=30, alpha=0.6, label='Wrong', color='red')
    plt.axvline(threshold, color='blue', linestyle='--', label=f'threshold={threshold:.2f}')
    plt.xlabel('Cosine Similarity')
    plt.ylabel('Count')
    plt.legend()
    plt.title(f'Bad Case Distribution ({args.dataset})')
    plt.savefig(f"../outputs/figures/badcase_{args.dataset}.png")
    plt.close()
    print(f"分布图已保存至 ../outputs/figures/badcase_{args.dataset}.png")

# ---------- 命令行 ----------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # 对比
    parser_compare = subparsers.add_parser('compare')
    # 错误分析
    parser_bad = subparsers.add_parser('badcase')
    parser_bad.add_argument('--dataset', type=str, default='afqmc', choices=['afqmc','lcqmc'])
    parser_bad.add_argument('--ckpt', type=str, required=True)
    parser_bad.add_argument('--bert_path', type=str, default='wk8/models/bert-base-chinese')
    parser_bad.add_argument('--n_cases', type=int, default=5)

    args = parser.parse_args()
    os.makedirs('../outputs/figures', exist_ok=True)
    if args.command == 'compare':
        compare_methods()
    elif args.command == 'badcase':
        analyze_badcases(args)
    else:
        parser.print_help()