# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import tensorflow as tf
from adjustText import adjust_text
from ampligraph.discovery import find_clusters
from ampligraph.evaluation import (
    hits_at_n_score,
    mr_score,
    mrr_score,
    train_test_split_no_unseen,
)
from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer
from rdflib import RDF, Literal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from rdflib import Graph, RDF, Literal

# %%
# Load the Minecraft ontology graph
GRAPH_FILE = "../data/minecraft.owl"
g = Graph()
g.parse(GRAPH_FILE, format="xml")

# Convert graph to triples, filtering out RDF type and literals
triples = []
for s, p, o in g:
    # Include RDF type triples but skip literals to capture class relationships
    if not isinstance(o, Literal):
        # Extract URIs using proper namespaces from ontology
        if str(s).startswith("http://minecraft.example.org/"):
            subject = str(s).split("/")[-1]
        else:
            subject = str(s)
            
        if str(p).startswith("http://minecraft.example.org/"):
            predicate = str(p).split("/")[-1]
        else:
            predicate = str(p).split("#")[-1] if "#" in str(p) else str(p).split("/")[-1]
            
        if str(o).startswith("http://minecraft.example.org/"):
            object_ = str(o).split("/")[-1]
        else:
            object_ = str(o).split("#")[-1] if "#" in str(o) else str(o).split("/")[-1]
            
        # Add triple if all parts are valid
        if subject and predicate and object_:
            triples.append([subject, predicate, object_])

# Convert to numpy array and split into train/test sets
triples_array = np.array(triples)
X_train, X_valid = train_test_split_no_unseen(triples_array, test_size=0.2)

# Create DataFrame for analysis
df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])

print(f"Total number of triples: {len(triples)}")
print(f"Train set size: {X_train.shape}")
print(f"Test set size: {X_valid.shape}")

# Set device to GPU if available
device = "/GPU:0" if tf.config.list_physical_devices('GPU') else "/CPU:0"
print(f"Using device: {device}")

with tf.device(device):
    # Initialize ComplEx embedding model
    model = ScoringBasedEmbeddingModel(
        k=100,  # Embedding dimension
        eta=20,  # Number of negative samples
        scoring_type="ComplEx",
        seed=0
    )

    # Configure training parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
    loss = get_loss("multiclass_nll")
    regularizer = get_regularizer("LP", {"p": 3, "lambda": 1e-5})
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        entity_relation_regularizer=regularizer
    )

    # Train the model
    model.fit(
        x=X_train,
        batch_size=X_train.shape[0] // 50,
        epochs=400,
        verbose=True,
        validation_data=X_valid,
        shuffle=True
    )

    # Evaluate model performance
    ranks = model.evaluate(
        X_valid,
        use_filter={"train": X_train, "test": X_valid},
        corrupt_side="s,o",
        verbose=True
    )


# %%
mr = mr_score(ranks)
mrr = mrr_score(ranks)

print("MRR: %.2f" % (mrr))
print("MR: %.2f" % (mr))

hits_10 = hits_at_n_score(ranks, n=10)
print("Hits@10: %.2f" % (hits_10))
hits_3 = hits_at_n_score(ranks, n=3)
print("Hits@3: %.2f" % (hits_3))
hits_1 = hits_at_n_score(ranks, n=1)
print("Hits@1: %.2f" % (hits_1))

# Get unique items and their embeddings
items = pd.concat([df["subject"], df["object"]]).unique()
items_embeddings = dict(zip(items, model.get_embeddings(items)))

# Project embeddings to 2D using PCA
embeddings_2d = PCA(n_components=2).fit_transform(
    np.array([i for i in items_embeddings.values()])
)

# Define semantic clusters based on Minecraft item types with more granular categories
CLUSTER_MAPPING = {
    "CombatTool": ["Sword"],
    "FarmingTool": ["Hoe", "Bucket", "Shears"],
    "MiningTool": ["Pickaxe", "Axe", "Shovel"],
    "SpecialTool": ["Fishing_Rod", "Flint_and_Steel", "Brush", "Spyglass"],
    "Armor": ["Helmet", "Chestplate", "Leggings", "Boots", "Cap", "Tunic"],
    "Material": ["Diamond", "Iron", "Gold", "Stone", "Stick", "Ingot", "Nugget"],
    "Ore": ["_Ore"],
    "Recipe": ["Recipe"],
    "ArmorSet": ["Set"],
    "Layer": [" to "],
    "Other": [],
}

# Assign items to semantic clusters with improved logic
semantic_clusters = []
for item in items:
    assigned = False
    item_str = str(item)

    # First check for specific categories that should take precedence
    if "Recipe" in item_str:
        semantic_clusters.append("Recipe")
        assigned = True
    elif "_Ore" in item_str:
        semantic_clusters.append("Ore")
        assigned = True
    elif "Set" in item_str and any(
        armor in item_str for armor in ["Diamond", "Iron", "Gold", "Leather", "Chain"]
    ):
        semantic_clusters.append("ArmorSet")
        assigned = True
    elif " to " in item_str:
        semantic_clusters.append("Layer")
        assigned = True
    else:
        # Then check other categories
        for cluster_name, keywords in CLUSTER_MAPPING.items():
            if cluster_name not in ["Recipe", "Ore", "ArmorSet", "Biome", "Layer"]:
                if any(keyword.lower() in item_str.lower() for keyword in keywords):
                    semantic_clusters.append(cluster_name)
                    assigned = True
                    break

    if not assigned:
        semantic_clusters.append("Other")

# Create clustering algorithm with semantic cluster count
n_clusters = len(set(semantic_clusters))
clustering_algorithm = KMeans(
    n_clusters=n_clusters,
    n_init="auto",  # Let KMeans choose the optimal number of initializations
    max_iter=1000,  # Increase max iterations for better convergence
    random_state=42,
)

# Get embedding clusters
clusters = find_clusters(items, model, clustering_algorithm, mode="e")

# Create DataFrame with both semantic and embedding clusters
plot_df = pd.DataFrame(
    {
        "item": items,
        "embedding1": embeddings_2d[:, 0],
        "embedding2": embeddings_2d[:, 1],
        "type": semantic_clusters,
        "cluster": [f"Cluster {i}" for i in clusters],  # Changed cluster labels to be more distinct
    }
)

# %%

top_entities = [
    "Diamond_Pickaxe",
    "Iron_Pickaxe", 
    "Diamond_Sword",
    "Iron_Sword",
    "Diamond",
    "Iron",
    "Gold",
    "Redstone",
    "Coal",
    "Emerald",
    "Wooden_Axe",
    "Stone_Shovel",
]



def plot_clusters(hue):
    np.random.seed(0)
    plt.figure(figsize=(12, 12))
    plt.title("{} embeddings".format(hue).capitalize())

    ax = sns.scatterplot(data=plot_df, x="embedding1", y="embedding2", hue=hue)

    texts = []
    for i, point in plot_df.iterrows():
        if point["item"] in top_entities or np.random.random() < 0.1:
            texts.append(ax.text(point['embedding1'] + 0.02, point['embedding2'] + 0.01, str(point["item"])))
    adjust_text(texts)

# %%

plot_clusters("type")
plt.show()

plot_clusters("cluster")
plt.show()


# %%
from sklearn import metrics
metrics.adjusted_rand_score(plot_df.type, plot_df.cluster)

# %%

plot_df["results"] = plot_df["cluster"].apply(lambda x: int(x.split()[-1]))  # Example: Convert cluster label to an integer

# Calculate value counts of the results
result_counts = plot_df["results"].value_counts(normalize=True)
print(result_counts)


# %%

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OneHotEncoder

# Формирование набора данных
X = np.array([items_embeddings[item] for item in items])
y = np.array(semantic_clusters)

# Разделение на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train)

# Оценка точности модели
y_pred = xgb_model.predict(X_test)
xgb_accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Accuracy: {xgb_accuracy:.2f}")

# Базовая модель 1: всегда предсказывает наиболее частый класс
dummy_most_frequent = DummyClassifier(strategy="most_frequent")
dummy_most_frequent.fit(X_train, y_train)
y_dummy_pred = dummy_most_frequent.predict(X_test)
dummy_accuracy = accuracy_score(y_test, y_dummy_pred)
print(f"Dummy Most Frequent Accuracy: {dummy_accuracy:.2f}")

# Базовая модель 2: классификация с использованием one-hot-encoding
encoder = OneHotEncoder()
X_one_hot = encoder.fit_transform(X).toarray()
X_train_oh, X_test_oh, y_train_oh, y_test_oh = train_test_split(X_one_hot, y, test_size=0.2, random_state=42)

xgb_model_oh = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
xgb_model_oh.fit(X_train_oh, y_train_oh)
y_pred_oh = xgb_model_oh.predict(X_test_oh)
xgb_oh_accuracy = accuracy_score(y_test_oh, y_pred_oh)
print(f"XGBoost One-Hot Encoding Accuracy: {xgb_oh_accuracy:.2f}")

# Выводы
if xgb_accuracy > dummy_accuracy and xgb_accuracy > xgb_oh_accuracy:
    print("Улучшена точность классификации по сравнению с базовыми моделями.")
else:
    print("Не удалось улучшить точность классификации по сравнению с базовыми моделями.")