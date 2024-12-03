import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

class ComplExTrainer:
    def __init__(self, model, device, lr=0.0005):
        self.model = model
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss()

    def train_epoch(self, triples_tensor, batch_size, num_entities):
        indices = torch.randperm(len(triples_tensor), device=self.device)
        total_loss = 0

        for i in range(0, len(triples_tensor), batch_size):
            batch_indices = indices[i:i + batch_size]
            batch = triples_tensor[batch_indices]

            # Positive examples
            pos_scores = self.model(batch[:, 0], batch[:, 1], batch[:, 2])

            # Generate multiple negative samples per positive
            num_neg = 5  # Number of negative samples per positive
            neg_scores_list = []
            
            for _ in range(num_neg):
                # Corrupt either head or tail
                neg_batch = batch.clone()
                corrupt_idx = torch.randint(2, (len(batch),), device=self.device) * 2
                random_entities = torch.randint(num_entities, (len(batch),), device=self.device)
                neg_batch[torch.arange(len(batch), device=self.device), corrupt_idx] = random_entities
                
                # Get scores for negative samples
                neg_scores = self.model(neg_batch[:, 0], neg_batch[:, 1], neg_batch[:, 2])
                neg_scores_list.append(neg_scores)

            # Combine all negative scores
            neg_scores = torch.cat(neg_scores_list)

            # Compute loss with margin ranking
            margin = 1.0
            target = torch.ones_like(pos_scores, device=self.device)
            loss = self.criterion(pos_scores, target)
            
            # Add margin ranking loss
            for neg_score in neg_scores_list:
                loss += torch.mean(torch.relu(margin - (pos_scores - neg_score)))

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / (len(triples_tensor) / batch_size)

    def get_embeddings(self, entity_to_idx, relation_to_idx):
        self.model.eval()
        with torch.no_grad():
            entity_embeddings = {}
            for entity, idx in entity_to_idx.items():
                idx_tensor = torch.tensor([idx], device=self.device)
                real = self.model.emb_e_real(idx_tensor).cpu().numpy()[0]
                img = self.model.emb_e_img(idx_tensor).cpu().numpy()[0]
                entity_embeddings[entity] = np.concatenate([real, img])

            relation_embeddings = {}
            for relation, idx in relation_to_idx.items():
                idx_tensor = torch.tensor([idx], device=self.device)
                real = self.model.emb_rel_real(idx_tensor).cpu().numpy()[0]
                img = self.model.emb_rel_img(idx_tensor).cpu().numpy()[0]
                relation_embeddings[relation] = np.concatenate([real, img])

        return entity_embeddings, relation_embeddings 