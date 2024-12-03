import torch
import numpy as np


def evaluate_triples(
    model,
    test_triples,
    entity_to_idx,
    relation_to_idx,
    all_entities,
    device,
    k_values=[1, 3, 10],
):
    """
    Evaluate model performance on test triples
    Returns MRR and Hits@K scores
    """
    ranks = []
    hits = {k: [] for k in k_values}

    model.eval()
    with torch.no_grad():
        for s, p, o in test_triples:
            # Create corrupted triples by replacing object
            corrupted = [(s, p, e) for e in all_entities if e != o]
            all_triples = [(s, p, o)] + corrupted

            # Convert to tensors
            s_idx = torch.tensor(
                [entity_to_idx[t[0]] for t in all_triples], device=device
            )
            p_idx = torch.tensor(
                [relation_to_idx[t[1]] for t in all_triples], device=device
            )
            o_idx = torch.tensor(
                [entity_to_idx[t[2]] for t in all_triples], device=device
            )

            # Get scores
            scores = model(s_idx, p_idx, o_idx)

            # Get rank of correct triple
            correct_score = scores[0]
            rank = 1 + (scores > correct_score).sum().item()
            ranks.append(rank)

            # Calculate hits@k
            for k in k_values:
                hits[k].append(1 if rank <= k else 0)

    # Calculate metrics
    mrr = np.mean([1.0 / r for r in ranks])
    hits_at_k = {k: np.mean(hits[k]) for k in k_values}
    mean_rank = np.mean(ranks)

    return mean_rank, mrr, hits_at_k


def predict_links(
    model, subject, predicate, entities, entity_to_idx, relation_to_idx, device, top_n=5
):
    """Predict most likely objects for a given subject-predicate pair"""
    if subject not in entity_to_idx:
        print(f"Warning: {subject} not found in entities")
        return []

    model.eval()
    with torch.no_grad():
        # Create all possible triples with this subject-predicate pair
        all_objects = list(entities)
        s_idx = torch.tensor([entity_to_idx[subject]] * len(all_objects), device=device)
        p_idx = torch.tensor(
            [relation_to_idx[predicate]] * len(all_objects), device=device
        )
        o_idx = torch.tensor([entity_to_idx[obj] for obj in all_objects], device=device)

        # Get scores
        scores = model(s_idx, p_idx, o_idx).cpu().numpy()

        # Get top N predictions
        top_indices = np.argsort(scores)[-top_n:][::-1]
        return [(all_objects[idx], float(scores[idx])) for idx in top_indices]
