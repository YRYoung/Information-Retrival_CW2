"""
Task 1 – Text statistics (30 marks).

Use the data in passage-collection.txt for this task.
Extract terms (1-grams) from the raw text.
perform basic text preprocessing steps. (you can also choose not to)



Do not remove stop words in the first instance.

Describe and justify any other text preprocessing choice(s), if any, and report the size of the identified index of
terms (vocabulary).

Then, implement a function that counts the number of occurrences of terms in the provided data set,  plot their
probability of occurrence (normalised frequency) against their frequency ranking,  and qualitatively  justify  that
these  terms  follow  Zipf ’s  law
"""
import re
from collections import Counter

import numpy as np
from nltk import WordNetLemmatizer, download
from nltk.corpus import stopwords
from tqdm.autonotebook import tqdm

try:
    remove_words = stopwords.words('english')
except:
    download('wordnet')
    download('stopwords')
    download('omw-1.4')
    remove_words = stopwords.words('english')

__all__ = ['preprocessing', 'read_txt', 'remove_stop_words']


def load_tqdm(iterate, verbose, **kwargs):
    return tqdm(iterate, **kwargs) if verbose else iterate


def get_tokens(text) -> list:
    token_pattern = r'\w[.]{1}\w|\w+'
    tokens = re.findall(token_pattern, string=text)

    return tokens


def read_txt(filename='passage-collection.txt'):
    with open(filename, "r", encoding='utf-8') as f:  # 打开文件
        data = f.read()  # 读取文件
    return data


def preprocessing(data, remove_stop_words=True, verbose=False):
    data = get_tokens(data.lower())
    data = remove_nums(data)

    # 把一个任何形式的语言词汇还原为一般形式（能表达完整语义）

    lemmatizer = WordNetLemmatizer()
    data = [lemmatizer.lemmatize(d) for d in (tqdm(data, unit='word', desc='lemmatizing') if verbose else data)]

    if remove_stop_words:
        data = list(filter(lambda x: x not in remove_words, data))

    return Counter(data).most_common()


def remove_nums(word_list):
    pattern = re.compile('\d')
    word_list = list(filter(lambda word: not bool(pattern.search(word)), word_list))
    return word_list


def remove_stop_words(text):
    indexes = [i for i in range(text.shape[0]) if text[i, 0] not in remove_words]
    return text[indexes]
