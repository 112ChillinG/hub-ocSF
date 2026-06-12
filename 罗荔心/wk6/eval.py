# eval.py

from seqeval.metrics import classification_report, f1_score

def eval_sequence_labeling(model, dataloader, tag2id, id2tag):
    """针对 BERT+Linear 和 BERT+CRF 的评估"""
    true_labels, pred_labels = [], []
    for batch in dataloader:
        true_labels.append([id2tag[l] for l in batch["labels"]])
        # 模型解码...
    return classification_report(true_labels, pred_labels)

def eval_generative_ner(model, dataloader):
    """针对 LLM API 和 LoRA SFT 的评估（宽松匹配）"""
    # 提取预测的实体字典 vs 真实实体字典，计算 micro F1
    # ...