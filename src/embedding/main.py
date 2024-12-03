import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.ontology.minecraft_ontology import create_minecraft_ontology

# Проверяем доступность CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ComplExModel(nn.Module):
    def __init__(self, num_entities, num_relations, embedding_dim, dropout=0.2):
        super(ComplExModel, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # Инициализация реальной и мнимой частей эмбеддингов
        self.emb_e_real = nn.Embedding(num_entities, embedding_dim)
        self.emb_e_img = nn.Embedding(num_entities, embedding_dim)
        self.emb_rel_real = nn.Embedding(num_relations, embedding_dim)
        self.emb_rel_img = nn.Embedding(num_relations, embedding_dim)
        
        # Инициализация весов
        nn.init.xavier_uniform_(self.emb_e_real.weight)
        nn.init.xavier_uniform_(self.emb_e_img.weight)
        nn.init.xavier_uniform_(self.emb_rel_real.weight)
        nn.init.xavier_uniform_(self.emb_rel_img.weight)

    def forward(self, e1_idx, rel_idx, e2_idx):
        # Получение эмбеддингов
        e1_real = self.emb_e_real(e1_idx)
        e1_img = self.emb_e_img(e1_idx)
        rel_real = self.emb_rel_real(rel_idx)
        rel_img = self.emb_rel_img(rel_idx)
        e2_real = self.emb_e_real(e2_idx)
        e2_img = self.emb_e_img(e2_idx)

        # Добавляем dropout
        e1_real = self.dropout(e1_real)
        e1_img = self.dropout(e1_img)
        e2_real = self.dropout(e2_real)
        e2_img = self.dropout(e2_img)

        # ComplEx scoring function
        score = torch.sum(
            rel_real * e1_real * e2_real +
            rel_real * e1_img * e2_img +
            rel_img * e1_real * e2_img -
            rel_img * e1_img * e2_real,
            dim=1
        )
        return score

def create_minecraft_embeddings(embedding_dim=150, num_epochs=500, batch_size=64, lr=0.0005):
    # Получаем граф
    g, _ = create_minecraft_ontology()
    
    # Конвертируем граф в тройки
    triples = []
    entities = set()
    relations = set()
    
    for s, p, o in g:
        # Конвертируем URI/Literals в строки
        subject = str(s).split('/')[-1]
        predicate = str(p).split('/')[-1]
        object_ = str(o).split('/')[-1] if '#' not in str(o) else str(o).split('#')[-1]
        
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
    
    # Создаем модель и перемещаем на GPU
    model = ComplExModel(
        num_entities=len(entities),
        num_relations=len(relations),
        embedding_dim=embedding_dim
    ).to(device)
    
    # Оптимизатор и функция потерь
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()
    
    # Обучение модели
    model.train()
    for epoch in range(num_epochs):
        # Перемешиваем данные
        indices = torch.randperm(len(triples_tensor), device=device)
        total_loss = 0
        
        # Батчи
        for i in range(0, len(triples_tensor), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch = triples_tensor[batch_indices]
            
            # Позитивные примеры
            pos_scores = model(
                batch[:, 0],
                batch[:, 1],
                batch[:, 2]
            )
            
            # Создаем негативные примеры (corrupted triples)
            neg_batch = batch.clone()
            # Случайно заменяем либо субъект, либо объект
            corrupt_idx = torch.randint(2, (len(batch),), device=device) * 2  # 0 или 2
            random_entities = torch.randint(len(entities), (len(batch),), device=device)
            neg_batch[torch.arange(len(batch), device=device), corrupt_idx] = random_entities
            
            # Негативные скоры
            neg_scores = model(
                neg_batch[:, 0],
                neg_batch[:, 1],
                neg_batch[:, 2]
            )
            
            # Вычисляем loss
            labels = torch.cat([
                torch.ones_like(pos_scores, device=device),
                torch.zeros_like(neg_scores, device=device)
            ])
            scores = torch.cat([pos_scores, neg_scores])
            loss = criterion(scores, labels)
            
            # Обратное распространение
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            avg_loss = total_loss / (len(triples_tensor) / batch_size)
            print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
    
    # Получаем финальные эмбеддинги
    model.eval()
    with torch.no_grad():
        entity_embeddings = {}
        for entity, idx in entity_to_idx.items():
            # Комбинируем реальную и мнимую части
            idx_tensor = torch.tensor([idx], device=device)
            real = model.emb_e_real(idx_tensor).cpu().numpy()[0]
            img = model.emb_e_img(idx_tensor).cpu().numpy()[0]
            entity_embeddings[entity] = np.concatenate([real, img])
            
        relation_embeddings = {}
        for relation, idx in relation_to_idx.items():
            idx_tensor = torch.tensor([idx], device=device)
            real = model.emb_rel_real(idx_tensor).cpu().numpy()[0]
            img = model.emb_rel_img(idx_tensor).cpu().numpy()[0]
            relation_embeddings[relation] = np.concatenate([real, img])
    
    return model, entity_embeddings, relation_embeddings

def visualize_embeddings_with_colors(entity_embeddings):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    
    # Преобразование эмбеддингов в матрицу
    embeddings_matrix = np.stack(list(entity_embeddings.values()))
    
    # Определение категорий
    categories = {
        'Recipe': '#ff7f0e',  # Оранжевый для рецептов
        'Helmet': '#1f77b4',  # Синий для шлемов
        'Chestplate': '#2ca02c',  # Зеленый для нагрудников
        'Leggings': '#9467bd',  # Фиолетовый для поножей
        'Boots': '#8c564b',  # Коричневый для ботинок
        'Sword': '#d62728',  # Красный для мечей
        'Axe': '#e377c2',  # Розовый для топоров
        'Hoe': '#bcbd22',  # Оливковый для мотыг
        'Other': '#7f7f7f'  # Серый для остального
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
                alpha=0.6
            )
    
    # Добавляем подписи для некоторых точек
    for i, entity in enumerate(entity_embeddings.keys()):
        if i % 10 == 0:  # Подписываем каждую 10-ю точку
            plt.annotate(entity, (embeddings_2d[i, 0], embeddings_2d[i, 1]))
    
    plt.title("t-SNE visualization of entity embeddings")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    model, entity_embeddings, relation_embeddings = create_minecraft_embeddings()
    
    # Пример использования эмбеддингов
    print("\nРазмеры эмбеддингов:")
    print(f"Количество сущностей: {len(entity_embeddings)}")
    print(f"Количество отношений: {len(relation_embeddings)}")
    print(f"Размерность эмбеддинга: {next(iter(entity_embeddings.values())).shape}")
    
    # Визуализация
    visualize_embeddings_with_colors(entity_embeddings)
