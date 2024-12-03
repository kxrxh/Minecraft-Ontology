import torch
import torch.nn as nn


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
            rel_real * e1_real * e2_real
            + rel_real * e1_img * e2_img
            + rel_img * e1_real * e2_img
            - rel_img * e1_img * e2_real,
            dim=1,
        )
        return score
