import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def visualize_embeddings_with_colors(entity_embeddings):
    # Преобразование эмбеддингов в матрицу
    embeddings_matrix = np.stack(list(entity_embeddings.values()))

    # Определение категорий
    categories = {
        "Recipe": "#ff7f0e",
        "Helmet": "#1f77b4",
        "Chestplate": "#2ca02c",
        "Leggings": "#9467bd",
        "Boots": "#8c564b",
        "Sword": "#d62728",
        "Axe": "#e377c2",
        "Hoe": "#bcbd22",
        "Other": "#7f7f7f",
    }

    # Уменьшение размерности до 2D
    tsne = TSNE(n_components=2)
    embeddings_2d = tsne.fit_transform(embeddings_matrix)

    # Создаем график
    plt.figure(figsize=(15, 10))

    # Отрисовываем точки по категориям
    for category, color in categories.items():
        mask = [category in entity for entity in entity_embeddings.keys()]
        if any(mask):
            plt.scatter(
                embeddings_2d[mask, 0],
                embeddings_2d[mask, 1],
                c=color,
                label=category,
                alpha=0.6,
            )

    # Добавляем подписи для некоторых точек
    for i, entity in enumerate(entity_embeddings.keys()):
        if i % 10 == 0:  # Подписываем каждую 10-ю точку
            plt.annotate(entity, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

    plt.title("t-SNE visualization of entity embeddings")
    plt.legend()
    plt.show()


def analyze_clusters(entity_embeddings, n_clusters=6):
    # Perform clustering on the embeddings
    clustering = KMeans(n_clusters=n_clusters, random_state=42)
    entity_vectors = np.array(list(entity_embeddings.values()))
    clusters = clustering.fit_predict(entity_vectors)

    # Analyze clusters
    cluster_contents = {i: [] for i in range(n_clusters)}
    for entity, cluster_id in zip(entity_embeddings.keys(), clusters):
        cluster_contents[cluster_id].append(entity)

    return cluster_contents
