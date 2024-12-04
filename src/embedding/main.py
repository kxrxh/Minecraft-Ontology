import numpy as np
import torch
from src.ontology.minecraft_ontology import create_minecraft_ontology
from src.embedding.models.complex_model import ComplExModel
from src.embedding.trainer import ComplExTrainer
from src.embedding.utils.evaluation import evaluate_triples
from src.embedding.utils.visualization import (
    visualize_embeddings_with_colors,
    analyze_clusters,
)
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def create_minecraft_embeddings(
    embedding_dim=150, num_epochs=800, batch_size=64, lr=0.0005
):
    # Получаем граф
    g, _ = create_minecraft_ontology()

    # Конвертируем граф в тройки
    triples = []
    entities = set()
    relations = set()

    for s, p, o in g:
        subject = str(s).split("/")[-1]
        predicate = str(p).split("/")[-1]
        object_ = str(o).split("/")[-1] if "#" not in str(o) else str(o).split("#")[-1]

        triples.append([subject, predicate, object_])
        entities.add(subject)
        entities.add(object_)
        relations.add(predicate)

    # Создаем словари для маппинга
    entity_to_idx = {ent: idx for idx, ent in enumerate(entities)}
    relation_to_idx = {rel: idx for idx, rel in enumerate(relations)}

    # Конвертируем тройки в индексы
    indexed_triples = [
        [entity_to_idx[triple[0]], relation_to_idx[triple[1]], entity_to_idx[triple[2]]]
        for triple in triples
    ]

    # Конвертируем в тензор и перемещаем на GPU
    triples_tensor = torch.LongTensor(indexed_triples).to(device)

    # Создаем модель и тренер
    model = ComplExModel(
        num_entities=len(entities),
        num_relations=len(relations),
        embedding_dim=embedding_dim,
    ).to(device)

    trainer = ComplExTrainer(model, device, lr)

    # Обучение модели
    model.train()
    for epoch in range(num_epochs):
        avg_loss = trainer.train_epoch(triples_tensor, batch_size, len(entities))

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # Получаем финальные эмбеддинги
    entity_embeddings, relation_embeddings = trainer.get_embeddings(
        entity_to_idx, relation_to_idx
    )

    # Оценка модели
    test_size = min(
        0.05, 5000 / len(triples)
    )  # 5% or 5000 triples, whichever is smaller
    X_train, X_valid = stratified_split_triples(triples, test_size=test_size)

    print(f"\nSplit statistics:")
    print(f"Training triples: {len(X_train)}")
    print(f"Validation triples: {len(X_valid)}")

    # Verify entity coverage
    train_entities = set()
    valid_entities = set()
    for s, p, o in X_train:
        train_entities.add(s)
        train_entities.add(o)
    for s, p, o in X_valid:
        valid_entities.add(s)
        valid_entities.add(o)

    print(f"Entities in training: {len(train_entities)}")
    print(f"Entities in validation: {len(valid_entities)}")
    print(f"Entities in both sets: {len(train_entities & valid_entities)}")

    print("\nValidation metrics:")
    mean_rank, mrr, hits_at_k = evaluate_triples(
        model, X_valid, entity_to_idx, relation_to_idx, list(entities), device
    )

    print(f"Mean Rank: {mean_rank:.2f}")
    print(f"Mean Reciprocal Rank: {mrr:.2f}")
    print(f"Hits@1: {hits_at_k[1]:.2f}")
    print(f"Hits@3: {hits_at_k[3]:.2f}")
    print(f"Hits@10: {hits_at_k[10]:.2f}")

    # Анализ эмбеддингов
    all_embeddings = np.array(list(entity_embeddings.values()))
    print("\nEmbedding Statistics:")
    print(f"Mean: {np.mean(all_embeddings):.3f}")
    print(f"Std: {np.std(all_embeddings):.3f}")
    print(f"Min: {np.min(all_embeddings):.3f}")
    print(f"Max: {np.max(all_embeddings):.3f}")

    # Кластерный анализ
    cluster_contents = analyze_clusters(entity_embeddings)
    print("\nCluster Analysis:")
    for cluster_id, entities in cluster_contents.items():
        print(f"\nCluster {cluster_id}:")
        for entity in entities[:5]:
            print(f"  {entity}")

    # Добавляем анализ качества эмбеддингов
    print("\nAnalyzing embedding quality...")
    plot_df = analyze_embeddings_quality(model, entity_embeddings, entity_to_idx)

    return model, entity_embeddings, relation_embeddings


def predict_relations(
    model,
    entity_embeddings,
    relation_embeddings,
    entity_to_idx,
    relation_to_idx,
    head,
    relation,
    temperature=0.1,
):
    """
    Predict the most likely tail entities for a given head entity and relation.
    """
    if head not in entity_to_idx or relation not in relation_to_idx:
        print(f"Warning: '{head}' or '{relation}' not found in indices")
        return []

    head_idx = torch.tensor([entity_to_idx[head]], device=device)
    rel_idx = torch.tensor([relation_to_idx[relation]], device=device)

    # Get predictions
    with torch.no_grad():
        scores = model.predict_tails(head_idx, rel_idx)
        scores = scores.squeeze().cpu().numpy()

        # Normalize scores using min-max normalization
        scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        # Apply temperature scaling
        scores = np.exp(scores / temperature) / np.sum(np.exp(scores / temperature))

        # Filter predictions based on relation type and domain knowledge
        entity_scores = []
        for entity, idx in entity_to_idx.items():
            score = scores[idx]

            if relation == "isPartOfArmorSet":
                # For armor sets, include all armor pieces with the same material
                if not entity.endswith("_Recipe") and any(
                    piece in entity
                    for piece in ["Helmet", "Chestplate", "Leggings", "Boots"]
                ):
                    material = head.split("_")[0]
                    if material.lower() in entity.lower():
                        score *= 1.5  # Increased boost for matching material
                        entity_scores.append((entity, score))

            elif relation == "usesMaterial":
                if score > 0.01:
                    base_materials = [
                        "Diamond",
                        "Iron",
                        "Gold",
                        "Stone",
                        "Wood",
                        "Stick",
                        "Planks",
                    ]
                    if any(mat in entity for mat in base_materials):
                        entity_scores.append((entity, score))

            elif relation == "requiresPickaxe":
                if score > 0.01 and "Pickaxe" in entity:
                    entity_scores.append((entity.replace("_", " "), score))

            elif relation == "foundInLayer":
                if (
                    score > 0.01
                    and any(x in entity for x in ["to", "-", "_"])
                    and not entity.endswith("Recipe")
                ):
                    entity_scores.append((entity, score))

            elif relation == "isUsedIn":
                if score > 0.01 and "_Recipe" not in entity and entity != head:
                    material = head.lower()
                    if material in ["diamond", "iron", "gold"]:
                        if any(
                            item in entity
                            for item in [
                                "Sword",
                                "Pickaxe",
                                "Axe",
                                "Shovel",
                                "Helmet",
                                "Chestplate",
                                "Leggings",
                                "Boots",
                            ]
                        ):
                            score *= 1.3
                    entity_scores.append((entity, score))

        # Re-normalize final scores
        if entity_scores:
            scores_array = np.array([s for _, s in entity_scores])
            scores_array = (scores_array - scores_array.min()) / (
                scores_array.max() - scores_array.min() + 1e-8
            )
            entity_scores = [(e, s) for (e, _), s in zip(entity_scores, scores_array)]

        # Sort by score and filter out low scores
        entity_scores.sort(key=lambda x: x[1], reverse=True)

        # Standardize naming (remove underscores)
        entity_scores = [(e.replace("_", " "), s) for e, s in entity_scores]

    return entity_scores


def print_predictions(predictions, top_k=5, relation=None):
    """Print the top K predictions with their scores."""
    print(f"\nTop {top_k} predictions:")
    print("-" * 50)

    if relation == "isUsedIn":
        # Group by category
        tools = []
        armor = []
        other = []

        for entity, score in predictions[:top_k]:
            if any(t in entity for t in ["Sword", "Pickaxe", "Axe", "Shovel"]):
                tools.append((entity, score))
            elif any(
                a in entity for a in ["Helmet", "Chestplate", "Leggings", "Boots"]
            ):
                armor.append((entity, score))
            else:
                other.append((entity, score))

        if tools:
            print("\nTools:")
            for entity, score in tools:
                print(f"{entity:<40} {score:.4f}")

        if armor:
            print("\nArmor:")
            for entity, score in armor:
                print(f"{entity:<40} {score:.4f}")

        if other:
            print("\nOther:")
            for entity, score in other:
                print(f"{entity:<40} {score:.4f}")
    else:
        # Regular printing for other relations
        for entity, score in predictions[:top_k]:
            print(f"{entity:<40} {score:.4f}")


def stratified_split_triples(triples, test_size=0.1):
    """
    Split triples ensuring all entities appear in both train and test sets.
    """
    # Create entity occurrence dictionary
    entity_occurrences = {}
    for s, p, o in triples:
        entity_occurrences[s] = entity_occurrences.get(s, []) + [(s, p, o)]
        entity_occurrences[o] = entity_occurrences.get(o, []) + [(s, p, o)]

    train_triples = []
    test_triples = []

    # For each entity, split its triples proportionally
    for entity, entity_triples in entity_occurrences.items():
        if len(entity_triples) < 2:
            # If entity appears only once, add to training
            train_triples.extend(entity_triples)
        else:
            # Randomly select triples for test set while ensuring at least one stays in training
            num_test = max(1, int(len(entity_triples) * test_size))
            num_test = min(
                num_test, len(entity_triples) - 1
            )  # Ensure at least one for training

            # Randomly shuffle and split
            entity_triples = list(set(entity_triples))  # Remove duplicates
            np.random.shuffle(entity_triples)

            test_triples.extend(entity_triples[:num_test])
            train_triples.extend(entity_triples[num_test:])

    # Remove duplicates while preserving order
    train_triples = list(dict.fromkeys(train_triples))
    test_triples = list(dict.fromkeys(test_triples))

    return train_triples, test_triples


def analyze_embeddings_quality(model, entity_embeddings, entity_to_idx):
    """
    Analyze embeddings quality using visualization and clustering metrics
    """
    # Get relevant entities (Sets and materials)
    relevant_entities = [
        ent
        for ent in entity_to_idx.keys()
        if any(
            key in ent for key in ["Set", "Diamond", "Iron", "Gold", "Wood", "Stone"]
        )
    ]

    # Get embeddings for relevant entities
    relevant_embeddings = np.array(
        [entity_embeddings[ent] for ent in relevant_entities]
    )

    # Project to 2D using PCA
    embeddings_2d = PCA(n_components=2).fit_transform(relevant_embeddings)

    # Perform clustering
    n_clusters = 6
    clustering = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    clusters = clustering.fit_predict(relevant_embeddings)

    # Create visualization dataframe
    plot_df = pd.DataFrame(
        {
            "entity": relevant_entities,
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
            "cluster": clusters,
        }
    )

    # Plot
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=plot_df, x="x", y="y", hue="cluster", style="cluster")

    # Add labels for all points since we're only showing relevant entities
    for _, row in plot_df.iterrows():
        plt.annotate(row["entity"], (row["x"], row["y"]))

    plt.title("Entity Embeddings Visualization")
    plt.show()

    # Calculate clustering metrics
    if len(clusters) > 1:
        silhouette = metrics.silhouette_score(embeddings_2d, clusters)
        calinski = metrics.calinski_harabasz_score(embeddings_2d, clusters)
        davies = metrics.davies_bouldin_score(embeddings_2d, clusters)

        print("\nClustering Quality Metrics:")
        print(f"Silhouette Score: {silhouette:.3f}")
        print(f"Calinski-Harabasz Score: {calinski:.3f}")
        print(f"Davies-Bouldin Score: {davies:.3f}")

    return plot_df


if __name__ == "__main__":
    model, entity_embeddings, relation_embeddings = create_minecraft_embeddings(
        embedding_dim=200,  # Increased dimension
        num_epochs=400,
        batch_size=64,
        lr=0.0005,
    )

    # Create reverse mappings for prediction
    entity_to_idx = {ent: idx for idx, ent in enumerate(entity_embeddings.keys())}
    relation_to_idx = {rel: idx for idx, rel in enumerate(relation_embeddings.keys())}

    # Example predictions with better descriptions
    print("\nПредметы в алмазном сете (Diamond Set pieces):")
    predictions = predict_relations(
        model,
        entity_embeddings,
        relation_embeddings,
        entity_to_idx,
        relation_to_idx,
        "Diamond_Set",
        "isPartOfArmorSet",
    )
    print_predictions(predictions)

    print("\nНа какой высоте можно найти алмазы (Diamond ore layer range):")
    predictions = predict_relations(
        model,
        entity_embeddings,
        relation_embeddings,
        entity_to_idx,
        relation_to_idx,
        "Diamond",
        "foundInLayer",
    )
    print_predictions(predictions)

    print("\nКакие предметы можно скрафтить из алмазов (Items craftable from Diamond):")
    predictions = predict_relations(
        model,
        entity_embeddings,
        relation_embeddings,
        entity_to_idx,
        relation_to_idx,
        "Diamond",
        "isUsedIn",
    )
    print_predictions(predictions)

    visualize_embeddings_with_colors(entity_embeddings)
