import torch
import torch.nn.functional as F

def compute_similarity_matrix(query_features, gallery_features):
    query_features = F.normalize(query_features, p=2, dim=1)
    gallery_features = F.normalize(gallery_features, p=2, dim=1)
    sim = query_features @ gallery_features.T

    return sim