# -*- coding: utf-8 -*-
"""MinecraftEmbedding.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NkK5qLFSooWuiCVdZ346GmECPiVmApRy

---
# Clustering and Classification using Knowledge Graph Embeddings
---

In this tutorial we will explore how to use the knowledge embeddings generated by a graph of international football matches (since the 19th century) in clustering and classification tasks. Knowledge graph embeddings are typically used for missing link prediction and knowledge discovery, but they can also be used for entity clustering, entity disambiguation, and other downstream tasks. The embeddings are a form of representation learning that allow linear algebra and machine learning to be applied to knowledge graphs, which otherwise would be difficult to do.


We will cover in this tutorial:

1. Creating the knowledge graph (i.e. triples) from a tabular dataset of football matches
2. Training the ComplEx embedding model on those triples
3. Evaluating the quality of the embeddings on a validation set
4. Clustering the embeddings, comparing to the natural clusters formed by the geographical continents
5. Applying the embeddings as features in classification task, to predict match results
6. Evaluating the predictive model on a out-of-time test set, comparing to a simple baseline

We will show that knowledge embedding clusters manage to capture implicit geographical information from the graph and that they can be a useful feature source for a downstream machine learning classification task, significantly increasing accuracy from the baseline.

---

## Requirements

A Python environment with the AmpliGraph library installed. Please follow the [install guide](http://docs.ampligraph.org/en/latest/install.html).

Some sanity check:
"""

!pip install tensorflow==2.9.0
!pip install ampligraph

import numpy as np
import pandas as pd
import ampligraph
print(ampligraph.__version__)
import tensorflow as tf

"""## Training knowledge graph embeddings

We split our training dataset further into training and validation, where the new training set will be used to the knowledge embedding training and the validation set will be used in its evaluation. The test set will be used to evaluate the performance of the classification algorithm built on top of the embeddings.

What differs from the standard method of randomly sampling N points to make up our validation set is that our data points are two entities linked by some relationship, and we need to take care to ensure that all entities are represented in train and validation sets by at least one triple.

To accomplish this, AmpliGraph provides the [`train_test_split_no_unseen`](https://docs.ampligraph.org/en/latest/generated/ampligraph.evaluation.train_test_split_no_unseen.html#train-test-split-no-unseen) function.
"""

from ampligraph.evaluation import train_test_split_no_unseen

from rdflib import Graph, RDF, Literal
from ampligraph.evaluation import train_test_split_no_unseen

GRAPH_FILE = "minecraft.owl"

# Load the Minecraft ontology
g = Graph()
g.parse(GRAPH_FILE, format="xml")

# Extract triples from the ontology
triples = []
for s, p, o in g:
    # Convert URIs and literals to strings
    s_str = str(s)
    p_str = str(p)
    o_str = str(o)

    # Skip RDF type triples and literals to focus on relationships
    if p != RDF.type and not isinstance(o, Literal):
        triples.append([s_str, p_str, o_str])

# Convert to numpy array and split into train/validation sets
X_train, X_valid = train_test_split_no_unseen(np.array(triples), test_size=0.2)
df = pd.DataFrame(triples, columns=["subject", "predicate", "object"])
print(df.head())

print('Train set size: ', X_train.shape)
print('Test set size: ', X_valid.shape)

from ampligraph.latent_features import ScoringBasedEmbeddingModel
from ampligraph.latent_features.loss_functions import get as get_loss
from ampligraph.latent_features.regularizers import get as get_regularizer

model = ScoringBasedEmbeddingModel(k=100,
                                   eta=20,
                                   scoring_type='ComplEx',
                                   seed=0)

# Optimizer, loss and regularizer definition
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
loss = get_loss('multiclass_nll')
regularizer = get_regularizer('LP', {'p': 3, 'lambda': 1e-5})

# Compilation of the model
model.compile(optimizer=optimizer, loss=loss, entity_relation_regularizer=regularizer)

"""AmpliGraph has implemented [several Knowledge Graph Embedding models](https://docs.ampligraph.org/en/latest/ampligraph.latent_features.html#knowledge-graph-embedding-models) (TransE, ComplEx, DistMult, HolE), but to begin with we're just going to use the [ComplEx](https://docs.ampligraph.org/en/latest/generated/ampligraph.latent_features.ComplEx.html#ampligraph.latent_features.ComplEx) model, which is known to bring state-of-the-art predictive power.

The hyper-parameter choice was based on the [best results](https://docs.ampligraph.org/en/latest/experiments.html) we have found so far for the ComplEx model applied to some benchmark datasets used in the knowledge graph embeddings community. This tutorial does not cover [hyper-parameter tuning](https://docs.ampligraph.org/en/latest/examples.html#model-selection).

Lets go through the parameters to understand what's going on:

- **`k`**: the dimensionality of the embedding space.
- **`eta`** ($\\eta$) : the number of negative, or false triples that must be generated at training runtime for each positive, or true triple.
- **`scoring_type`**: type of model defined by spicific scoring function.
- **`seed`** : random seed, used for reproducibility.
- **`optimizer`** : the Adam optimizer, with a learning rate of 1e-4 set via the *optimizer_params* kwarg.
- **`loss`** : pairwise loss, with a margin of 0.5 set via the *loss_params* kwarg.
- **`regularizer`** : $L_p$ regularization with $p=3$, i.e. l3 regularization. $\\lambda$ = 1e-5, set via the *regularizer_params* kwarg.

Training should take around 10 minutes on a modern GPU:
"""

# For the fit call:
model.fit(
    x=X_train,
    batch_size=X_train.shape[0] // 50,  # Back to original batch size strategy
    epochs=400,              # Middle ground for epochs
    verbose=True,
    validation_data=X_valid,
    shuffle=True
)

"""## Evaluating knowledge embeddings

AmpliGraph aims to follow scikit-learn's ease-of-use design philosophy and simplify everything down to **`fit`**, **`evaluate`**, and **`predict`** functions.

However, there are some knowledge graph specific steps we must take to ensure our model can be trained and evaluated correctly. The first of these is defining the filter that will be used to ensure that no negative statements generated by the corruption procedure are actually positives. This is simply can be done by concatenating our train and test sets. Now when negative triples are generated by the corruption strategy, we can check that they aren't actually true statements.

For this we'll use method `evaluate` of model object:

- **`X_valid`** - the data to evaluate on. We're going to use our test set to evaluate.
- **`use_filter`** - will filter out the false negatives generated by the corruption strategy.
- **`corrupt_side`** - specifies approach for triple corruption. 's,o' option means t True, then subj and obj are corrupted separately during evaluation.
- **`verbose`** - displays a progress bar.
"""

ranks = model.evaluate(X_valid,
                      use_filter={'train': X_train,
                                  'test': X_valid},
                      corrupt_side='s,o',
                      verbose=True)

"""We're going to use the mrr_score (mean reciprocal rank) and hits_at_n_score functions.

- **mrr_score**: The function computes the mean of the reciprocal of elements of a vector of rankings ranks.
- **hits_at_n_score**: The function computes how many elements of a vector of rankings ranks make it to the top n positions.
"""

from ampligraph.evaluation import mr_score, mrr_score, hits_at_n_score

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

"""We can interpret these results by stating that the model will rank the correct entity within the top-3 possibilities 34% of the time.

By themselves, these metrics are not enough to conclude the usefulness of the embeddings in a downstream task, but they suggest that the embeddings have learned a reasonable representation enough to consider using them in more tasks.

## Clustering and embedding visualization

To evaluate the subjective quality of the embeddings, we can visualise the embeddings on 2D space and also cluster them on the original space. We can compare the clustered embeddings with natural clusters, in this case the continent where the team is from, so that we have a ground truth to evaluate the clustering quality both qualitatively and quantitatively.

Requirements:

* seaborn
* adjustText
* incf.countryutils

For seaborn and adjustText, simply install them with `pip install seaborn adjustText`.

For incf.countryutils, do the following steps:
```bash
git clone https://github.com/wyldebeast-wunderliebe/incf.countryutils.git
cd incf.countryutils
pip install .```
"""

!git clone https://github.com/wyldebeast-wunderliebe/incf.countryutils.git
!cd incf.countryutils && pip install .

!pip install adjustText

# Commented out IPython magic to ensure Python compatibility.
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text
from incf.countryutils import transformations
# %matplotlib inline

# Assuming your DataFrame is named 'df' and has columns 'Subject', 'Predicate', 'Object'
id_to_name_map = {}

# Extract entities from Subject and Object columns
entities = pd.concat([df["subject"], df["object"]]).unique()

# Create the mapping (entity URI to its name)
for entity in entities:
  # If entity starts with http, it is a URI, otherwise its already a name
  if entity.startswith("http"):
    # Assume the last part of URI after '#' is the name
    name = entity.split("#")[-1]
    id_to_name_map[entity] = name

print(id_to_name_map)

"""We now create a dictionary with the embeddings of all teams:"""

# Assuming your DataFrame is named 'df' with columns 'Subject', 'Predicate', 'Object'
# and you have 'id_to_name_map' created from the previous step

# Extract unique entities from the Subject and Object columns
teams = pd.concat([df["subject"], df["object"]]).unique()

# Filter out non-entity URIs (if needed)
# You might need to adjust this based on your data
# For example, if you only want entities starting with a specific prefix:
# teams = teams[teams.str.startswith("http://your_ontology_prefix#")]

# Create team embeddings dictionary
team_embeddings = dict(zip(teams, model.get_embeddings(teams)))

"""We use PCA to project the embeddings from the 200 space into 2D space:"""

embeddings_2d = PCA(n_components=2).fit_transform(np.array([i for i in team_embeddings.values()]))

"""We will cluster the teams embeddings on its original 200-dimensional space using the `find_clusters` in our discovery API:"""

from ampligraph.discovery import find_clusters
from sklearn.cluster import KMeans

clustering_algorithm = KMeans(n_clusters=6, n_init=50, max_iter=500, random_state=0)
clusters = find_clusters(teams, model, clustering_algorithm, mode='e')

"""This helper function uses the `incf.countryutils` library to translate country names to their corresponding continents."""

def cn_to_ctn(country):
    try:
        return transformations.cn_to_ctn(id_to_name_map[country])
    except KeyError:
        return "unk"

"""This dataframe contains for each team their projected embeddings to 2D space via PCA, their continent and the KMeans cluster. This will be used alongisde Seaborn to make the visualizations."""

plot_df = pd.DataFrame({"teams": teams,
                        "embedding1": embeddings_2d[:, 0],
                        "embedding2": embeddings_2d[:, 1],
                        "continent": pd.Series(teams).apply(cn_to_ctn),
                        "cluster": "cluster" + pd.Series(clusters).astype(str)})

"""We plot the results on a 2D scatter plot, coloring the teams by the continent or cluster and also displaying some individual team names.

We always display the names of the top 20 teams (according to [FIFA rankings](https://en.wikipedia.org/wiki/FIFA_World_Rankings)) and a random subset of the rest.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

# Define the top entities (replace with relevant entities from your ontology)
top_entities = ["Tool", "Ore", "Material", "Biome", "Layer", "Recipe", "Armor"]

def plot_clusters(hue):
    np.random.seed(0)
    plt.figure(figsize=(12, 12))
    plt.title("{} embeddings".format(hue).capitalize())

    # Assuming your DataFrame is named 'plot_df' with columns 'entity', 'embedding1', 'embedding2', and 'cluster'
    ax = sns.scatterplot(data=plot_df, x="embedding1", y="embedding2", hue=hue)

    texts = []
    for i, point in plot_df.iterrows():
        if point["entity"] in top_entities or np.random.random() < 0.1:
            texts.append(plt.text(point['embedding1'] + 0.02, point['embedding2'] + 0.01, str(point["entity"])))
    adjust_text(texts)

"""The first visualisation of the 2D embeddings shows the natural geographical clusters (continents), which can be seen as a form of the ground truth:"""

plot_clusters("continent")

"""We can see above that the embeddings learned geographical similarities even though this information was not explicit on the original dataset.

Now we plot the same 2D embeddings but with the clusters found by K-Means:
"""

plot_clusters("cluster")

"""We can see that K-Means found very similar cluster to the natural geographical clusters by the continents. This shows that on the 200-dimensional embedding space, similar teams appear close together, which can be captured by a clustering algorithm.

Our evaluation of the clusters can be more objective by using a metric such as the [adjusted Rand score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html), which varies from -1 to 1, where 0 is random labelling and 1 is a perfect match:
"""

from sklearn import metrics
metrics.adjusted_rand_score(plot_df.continent, plot_df.cluster)

"""## Classification

We will use the knowledge embeddings to predict future matches as a classification problem.

We can model it as a multiclass problem with three classes: home team wins, home team loses, draw.

The embeddings are used directly as features to a XGBoost classifier.

First we need to determine the target:
"""

df["results"] = (df.home_score > df.away_score).astype(int) + \
                (df.home_score == df.away_score).astype(int)*2 + \
                (df.home_score < df.away_score).astype(int)*3 - 1

df.results.value_counts(normalize=True)

"""### Original dataset

First install xgboost with `pip install xgboost`.
"""

!pip install xgboost

new_df = df[["results", "neutral", "train"]].copy()
new_df

new_df.neutral = new_df.neutral.astype(bool)

"""Split date:"""

new_df["year"] = df["date"].apply(lambda x: int(x[:4]))
new_df["month"] = df["date"].apply(lambda x: int(x[5:7]))
new_df["day"] = df["date"].apply(lambda x: int(x[8:]))

"""Apply one hot encoding:"""

encoded_cols = pd.get_dummies(df[["home_team", "away_team", "tournament", "city", "country"]])
encoded_cols

new_df = new_df.join(encoded_cols)
new_df

"""Create a multiclass model:"""

from xgboost import XGBClassifier

clf_model = XGBClassifier(n_estimators=500, max_depth=5, objective="multi:softmax")

X_train = new_df[df["train"]].drop(["results"], axis=1)
y_train = new_df[df["train"]].results
X_val = new_df[~df["train"]].drop(["results"], axis=1)
y_val = new_df[~df["train"]].results

clf_model.fit(X_train, y_train, verbose=1)

"""Result"""

from sklearn import metrics
metrics.accuracy_score(y_val, clf_model.predict(X_val))

"""### Graph embedings

Now we create a function that extracts the features (knowledge embeddings for home and away teams) and the target for a particular subset of the dataset:
"""

def get_features_target(mask):

    def get_embeddings(team):
        return team_embeddings.get(team, np.full(200, np.nan))

    X = np.hstack((np.vstack(df[mask].home_team_id.apply(get_embeddings).values),
                   np.vstack(df[mask].away_team_id.apply(get_embeddings).values)))
    y = df[mask].results.values
    return X, y

clf_X_train, y_train = get_features_target((df["train"]))
clf_X_test, y_test = get_features_target((~df["train"]))

clf_X_train.shape, clf_X_test.shape

"""Note that we have 200 features by team because the ComplEx model uses imaginary and real number for its embeddings, so we have twice as many parameters as defined by `k=100` in its model definition.

We also have some missing information from the embeddings of the entities (i.e. teams) that only appear in the test set, which are unlikely to be correctly classified:
"""

np.isnan(clf_X_test).sum()/clf_X_test.shape[1]

"""Create a multiclass model with 500 estimators:"""

clf_model = XGBClassifier(n_estimators=500, max_depth=5, objective="multi:softmax")

"""Fit the model using all of the training samples:"""

clf_model.fit(clf_X_train, y_train)

"""The baseline accuracy for this problem is 47%, as that is the frequency of the most frequent class (home team wins):"""

df[~df["train"]].results.value_counts(normalize=True)

metrics.accuracy_score(y_test, clf_model.predict(clf_X_test))

"""In conclusion, while the baseline for this classification problem was 47%, with just the knowledge embeddings alone we were able to build a classifier that achieves **53%** accuracy.

As future work, we could add more features to the model (not embeddings related) and tune the model hyper-parameters.

## Link prediction

Link prediction allows us to infer missing links in a graph.

In our case, we're going to predict match result.
Choose match that exist in train dataset.
"""

X_train, X_valid = train_test_split_no_unseen(np.array(triples), test_size=10000)

df = pd.DataFrame(X_train,columns = ['subject','predicate','object'])
matchSubject = "Match1324"
print(df[df.subject==matchSubject])

"""Remove result for this match from train dataframe."""

dfFiltered = np.array(df[(df.subject!=matchSubject) | ((df.subject==matchSubject) & ~df.predicate.isin(["homeScores","awayScores"]))])

"""Fit model on triples without results for current match."""

model.fit(dfFiltered)

"""We can create a few statements for this match result."""

statements = np.array([
    [f'{matchSubject}', 'homeScores', '0'],
    [f'{matchSubject}', 'homeScores', '1'],
    [f'{matchSubject}', 'homeScores', '2'],
    [f'{matchSubject}', 'homeScores', '3'],
    [f'{matchSubject}', 'homeScores', '4'],
    [f'{matchSubject}', 'awayScores', '0'],
    [f'{matchSubject}', 'awayScores', '1'],
    [f'{matchSubject}', 'awayScores', '2'],
    [f'{matchSubject}', 'awayScores', '3'],
    [f'{matchSubject}', 'awayScores', '4']
])

"""Unite the triplets of the graph and the proposed statements."""

statements_filter = np.array(list({tuple(i) for i in np.vstack((dfFiltered, statements))}))
statements_filter

ranks = model.evaluate(statements,
                      use_filter={'train': dfFiltered,
                                  'test': statements},
                      corrupt_side='s,o',
                      verbose=True)

scores = model.predict(statements)
scores

"""Present the result of predictions."""

from scipy.special import expit
probs = expit(scores)

pd.DataFrame(list(zip([' '.join(x) for x in statements],
                      ranks,
                      np.squeeze(scores),
                      np.squeeze(probs))),
             columns=['statement', 'rank', 'score', 'prob']).sort_values("prob")