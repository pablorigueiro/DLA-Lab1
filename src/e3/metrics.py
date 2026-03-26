import torch

def precision_recall_curve(relevant_sorted):
    relevant_sorted = relevant_sorted.float()
    total_relevant = relevant_sorted.sum()
    if total_relevant == 0:
        raise ValueError("This query has no relevant gallery items.")
    cumulative_relevant = relevant_sorted.cumsum(dim=0)
    ks = torch.arange(1, len(relevant_sorted) + 1, dtype=torch.float32)
    precisions = cumulative_relevant / ks
    recalls = cumulative_relevant / total_relevant
    return precisions, recalls

def average_precision(relevant_sorted):
    relevant_sorted = relevant_sorted.float()
    total_relevant = relevant_sorted.sum()
    if total_relevant == 0:
        return 0.0
    cumulative_relevant = relevant_sorted.cumsum(dim=0)
    ks = torch.arange(1, len(relevant_sorted) + 1, dtype=torch.float32)
    precision_at_k = cumulative_relevant / ks
    ap = (precision_at_k * relevant_sorted).sum() / total_relevant
    return ap.item()


def compute_map_per_class(similarity_matrix, query_labels, gallery_labels, num_classes=43):
    sorted_indices = similarity_matrix.argsort(dim=1, descending=True)
    all_aps = {c: [] for c in range(num_classes)}
    for i in range(len(query_labels)):
        query_class = query_labels[i].item()
        ranking = sorted_indices[i]
        relevant = (gallery_labels == query_class)
        relevant_sorted = relevant[ranking]
        ap = average_precision(relevant_sorted)
        all_aps[query_class].append(ap)
    class_ap = {}
    for c in range(num_classes):
        if len(all_aps[c]) > 0:
            class_ap[c] = sum(all_aps[c]) / len(all_aps[c])
        else:
            class_ap[c] = 0.0
    return class_ap, all_aps