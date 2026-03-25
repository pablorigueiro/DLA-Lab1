import torch
import torch.nn.functional as F

def compute_class_means(features, labels, num_classes):
    class_means = []

    for c in range(num_classes):
        class_features = features[labels == c]
        mean_feature = class_features.mean(dim=0)
        mean_feature = F.normalize(mean_feature, p=2, dim=0)
        class_means.append(mean_feature)

    class_means = torch.stack(class_means, dim=0)
    return class_means


def predict_nmc(query_features, class_means):
    query_features = F.normalize(query_features, p=2, dim=1)
    class_means = F.normalize(class_means, p=2, dim=1)

    similarity_to_means = query_features @ class_means.T
    preds = similarity_to_means.argmax(dim=1)

    return preds, similarity_to_means

def compute_accuracy(preds, labels):
    return (preds == labels).float().mean().item()