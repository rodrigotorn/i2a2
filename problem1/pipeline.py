# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Problem 1
#
# Transform a text input into a matrix. The chosen method is word embedding using the word2vec algorithm and continuous bag of words (CBOW) model.

# %% [markdown]
# ### Import models for word embedding, principal component analysis and plotting

# %%
import pandas as pd
import plotly.express as px
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# %% [markdown] tags=[]
# ### Read data from text file

# %%
f = open("input.txt", "r")
input_text = f.read()

# %% [markdown]
# ### Preprocess raw text
#
# This stage transform the paragraphs into sentences. The sentences are splitted into words, all words in lowercase.

# %%
preproc_text = input_text.replace('\n',"")
preproc_text = preproc_text.replace(',',"")
preproc_text = preproc_text.lower()
preproc_text = preproc_text.split('.')

sentences = []
for i in range(len(preproc_text)):
    sentences.append(
        list(filter(None, preproc_text[i].split(' ')))
    )

# %% [markdown] tags=[]
# ### Analysis
#
# The word2vec algorithm is applied to the sentences. A sample from the resulting matrix is shown.

# %%
model = Word2Vec(sentences, min_count=1)
X = model[model.wv.vocab]
print(pd.DataFrame(X).head())

# %% [markdown]
# ### Present the PCA from the resulting matrix

# %%
pca = PCA(n_components=2)
result = pca.fit_transform(X)
words = list(model.wv.vocab)

fig = px.scatter(result, result[:, 0], result[:, 1], text=words)
fig.update_traces(textposition='top center')
fig.show()
