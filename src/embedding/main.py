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
from sklearn.model_selection import train_test_split
from rdflib import Literal

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
        [entity_to_idx[t[0]], relation_to_idx[t[1]], entity_to_idx[t[2]]]
        for t in triples
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
    test_size = min(10000, len(triples) // 10)
    X_train, X_valid = train_test_split(triples, test_size=test_size, random_state=42)

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

    return model, entity_embeddings, relation_embeddings


def predict_relations(
    model,
    entity_embeddings,
    relation_embeddings,
    entity_to_idx,
    relation_to_idx,
    head,
    relation,
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

        # Filter predictions based on relation type and domain knowledge
        entity_scores = []
        for entity, idx in entity_to_idx.items():
            score = scores[idx]

            # Skip if score is too low
            if score < 0.1:
                continue

            # Relation-specific filtering and score adjustments
            if relation == "isPartOfArmorSet":
                # Only include actual armor pieces (not recipes)
                if not entity.endswith("_Recipe") and any(
                    piece in entity
                    for piece in ["Helmet", "Chestplate", "Leggings", "Boots"]
                ):
                    # Boost score if material matches
                    material = head.split("_")[0]
                    if material in entity:
                        score *= 1.5
                    entity_scores.append((entity, score))

            elif relation == "usesMaterial":
                # Filter out non-material items
                base_materials = [
                    "Diamond",
                    "Iron",
                    "Gold",
                    "Stone",
                    "Wood",
                    "Stick",
                    "Planks",
                ]
                entity_scores = [
                    (e, s)
                    for e, s in entity_scores
                    if any(mat in e for mat in base_materials)
                    and not any(
                        item in e
                        for item in ["Sword", "Pickaxe", "Axe", "Shovel", "Helmet"]
                    )
                ]

            elif relation == "requiresPickaxe":
                # Only show pickaxes and standardize names
                entity_scores = [
                    (e.replace("_", " "), s) for e, s in entity_scores if "Pickaxe" in e
                ]

                # Sort by material tier
                material_tiers = {
                    "Wooden": 1,
                    "Stone": 2,
                    "Iron": 3,
                    "Golden": 2,
                    "Diamond": 4,
                    "Netherite": 5,
                }

                def get_tier(item):
                    for material, tier in material_tiers.items():
                        if material in item:
                            return tier
                    return 0

                entity_scores.sort(key=lambda x: (-get_tier(x[0]), -x[1]))

            elif relation == "foundInLayer":
                # Only include valid layer ranges
                if any(x in entity for x in ["to", "-", "_"]) and not entity.endswith(
                    "Recipe"
                ):
                    entity_scores.append((entity, score))

            elif relation == "isUsedIn":
                # Include items that can be crafted from the material
                if "_Recipe" not in entity and entity != head:
                    # Boost score for items typically crafted from this material
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

        # Sort by score and filter out low scores
        entity_scores = [(e, s) for e, s in entity_scores if s > 0.1]
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


if __name__ == "__main__":
    model, entity_embeddings, relation_embeddings = create_minecraft_embeddings(
        embedding_dim=200,  # Increased dimension
        num_epochs=800,
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
