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


if __name__ == "__main__":
    model, entity_embeddings, relation_embeddings = create_minecraft_embeddings()

    print("\nРазмеры эмбеддингов:")
    print(f"Количество сущностей: {len(entity_embeddings)}")
    print(f"Количество отношений: {len(relation_embeddings)}")
    print(f"Размерность эмбеддинга: {next(iter(entity_embeddings.values())).shape}")

    visualize_embeddings_with_colors(entity_embeddings)
