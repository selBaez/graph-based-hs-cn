from collections import Counter
from itertools import chain

import pandas as pd
import plotly.express as px
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from spacy.lang.en import stop_words


def get_tfidf(corpus_columns):
    # Get tfidf features
    tfidf_df = pd.DataFrame(columns=['HS', 'CN', 'K', 'KCN'])
    k_features = 10

    for corpus, column in corpus_columns:
        vectorizer = TfidfVectorizer(token_pattern=r'(?u)\b[A-Za-z]+\b', max_features=k_features, stop_words='english')
        _ = vectorizer.fit_transform(corpus)
        tfidf_df[column] = vectorizer.get_feature_names_out()[:k_features]

    return tfidf_df


def get_type_token_ratio(corpus_columns):
    type_token_df = pd.DataFrame(columns=['HS', 'CN', 'K', 'KCN'])
    nlp = spacy.load("en_core_web_sm")

    for corpus, column in corpus_columns:
        tokens = []
        for doc in nlp.pipe(corpus):
            this_tokens = [token.text for sent in doc.sents for token in sent]
            tokens.extend(this_tokens)
        type_token_df._set_value(0, column, (len(set(tokens)) / len(tokens)) * 100)

    return type_token_df


def find_ngrams(input_list, n, remove_stopwords=True):
    if remove_stopwords:
        input_list = [t for t in input_list if t not in stop_words.STOP_WORDS]
    return list(zip(*[input_list[i:] for i in range(n)]))


def get_ngram_count(df, column, ngram_size=2, topk=20):
    # create ngrams
    df['ngrams'] = df[column].map(lambda x: find_ngrams(x.split(" "), ngram_size) if type(x) == str else x)
    ngrams = [x for x in df['ngrams'].tolist() if type(x) == list]
    ngrams = list(chain(*ngrams))

    # get ngram frequency
    ngram_counts = Counter(ngrams)
    ngram_counts = pd.DataFrame.from_dict(ngram_counts, orient='index', columns=['count'])

    # format output
    ngram_counts.sort_values(by="count", inplace=True, ascending=False)
    ngram_counts = ngram_counts[:topk]
    ngram_counts.reset_index(inplace=True)

    return ngram_counts


def sunburst_ngrams(corpus_columns):
    for corpus, column in corpus_columns:
        trigram_df = pd.DataFrame({"text": corpus})
        trigrams = get_ngram_count(trigram_df, 'text', ngram_size=3)

        trigrams["A"] = trigrams["index"].apply(lambda x: x[0])
        trigrams["B"] = trigrams["index"].apply(lambda x: x[1])
        trigrams["C"] = trigrams["index"].apply(lambda x: x[2])
        trigrams["G"] = trigrams["count"]
        trigrams.drop(columns=["index", "count"], inplace=True)

        fig = px.sunburst(trigrams, path=['A', 'B', 'C'], values='G')
        plot_filename = f"./../plots/sunburst_distribution_{column}.html"
        fig.write_html(plot_filename)

        print(f"Wrote plot to file: {plot_filename}")
