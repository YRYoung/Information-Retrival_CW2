"""
Logistic Regression (LR)(25 marks)

Represent passages and query based on a word embedding method,
(such as Word2Vec, GloVe, FastText, or ELMo).

Compute query (/passage) embeddings by averaging embeddings of all the words in that query (/passage).

With these query and passage embeddings as input, implement a logistic regression model
to assess relevance of a passage to a given query.

Describe how you perform input processing & representation or features used.

Using the metrics you have implemented in the previous part,
report the performance of your model based on the validation data.

Analyze the effect of the learning rate on the model training loss.

(All implementations for logistic regression algorithm must be your own for this part.)
"""

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence

# init embedding
embedding = TransformerWordEmbeddings('bert-base-uncased')

# create a sentence
sentence = Sentence('The grass is green.')

# embed words in sentence
embedding.embed(sentence)