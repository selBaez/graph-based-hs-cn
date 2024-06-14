'''
Adapted from https://github.com/amazon-science/mm-cot
'''

import re
import warnings

import pandas as pd
import syllables
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

from eval_utils.heavy_metrics import Bleurt, CounterspeechScore, SentenceSimilarityScore
from eval_utils.toxic_metric import ToxicHateXplain

warnings.filterwarnings('ignore')


def get_scores_conan(results, reference, training_corpus):
    df = pd.DataFrame()
    df["pred"] = results
    df["ref"] = reference

    # format for metrics
    training_corpus = {idx: t["counter_narrative"] for idx, t in enumerate(training_corpus)}
    results = {idx: item for idx, item in enumerate(results)}
    reference = {idx: item for idx, item in enumerate(reference)}
    results_tokens = {idx: word_tokenize(item) for idx, item in results.items()}
    reference_tokens = {idx: word_tokenize(item) for idx, item in reference.items()}

    ## BLEU
    df["bleu1"] = calculate_bleu(results, reference, gram=1)
    df["bleu4"] = calculate_bleu(results, reference, gram=4)

    ## Rouge-L
    df["rouge"] = calculate_rouge(results, reference)

    ## Meteor
    df["meteor"] = calculate_meteor(results_tokens, reference_tokens)

    ## Gleu
    df["gleu"] = calculate_gleu(results_tokens, reference_tokens)

    ## RR
    df["repetition_rate"] = calculate_repetition_rate(results)

    ## Fre
    df["fre"] = fre_readability(results)

    ## Similarity
    similarity_scorer = SentenceSimilarityScore()
    df["similarity"] = similarity_scorer.score(results, reference)

    ## Bleurt
    bleurt_score = Bleurt(model_path="Elron/bleurt-base-512", max_length=400, batch_size=16)
    df["bleurt"] = bleurt_score.score(results, reference)

    ## Counter-speech
    counterspeech_score = CounterspeechScore(model_path='Hate-speech-CNERG/counterspeech-quality-bert',
                                             max_length=100, batch_size=16)
    df["counterspeech"] = counterspeech_score.scoring(results)

    ## toxicity
    toxicity_score = ToxicHateXplain(model_path="Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
                                     max_length=100, batch_size=16)
    df["toxicity"] = toxicity_score.scoring(results)

    ## Diversity
    df["diversity"] = get_diversity(results)

    ## Novelty
    df["novelty"] = get_novelty(results, training_corpus)

    scores = {'bleu1': df["bleu1"].mean() * 100,
              'bleu4': df["bleu4"].mean() * 100,
              'rouge': df["rouge"].mean() * 100,
              'meteor': df["meteor"].mean() * 100,
              'gleu': df["gleu"].mean() * 100,
              'repetition_rate': df["repetition_rate"].mean() * 100,
              'fre': df["fre"].mean(),
              'sentence_similariry': df["similarity"].mean() * 100,
              'bleurt': df["bleurt"].mean() * 100,
              'counterspeech': df["counterspeech"].mean() * 100,
              'toxicity': df["toxicity"].mean() * 100,
              'diversity': df["diversity"].mean() * 100,
              'novelty': df["novelty"].mean() * 100
              }

    scores = {k: float(val) for k, val in scores.items()}

    return scores, df


########################
## BLEU
########################
def tokenize(text):
    tokens = re.split(r'\s|\.', text)
    tokens = [t for t in tokens if len(t) > 0]
    return tokens


def bleu_score(reference, hypothesis, gram):
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)

    if gram == 1:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1.,))  # BLEU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BLEU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BLEU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BLEU-4

    return bleu


def calculate_bleu(results, data, gram):
    bleus = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if target == "":
            bleu = 0
        else:
            bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)

    return bleus


########################
## Rouge-L
########################
def score_rouge(str1, str2):
    rouge = Rouge(metrics=["rouge-l"])
    scores = rouge.get_scores(str1, str2, avg=True)
    rouge_l = scores['rouge-l']['f']
    return rouge_l


def calculate_rouge(results, data):
    rouges = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if prediction == "" or target == "":
            rouge = 0
        else:
            rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    return rouges


########################
## Meteor
########################
def calculate_meteor(results, data):
    meteors = []
    for qid, output in results.items():
        score = meteor_score([output], data[qid])
        meteors.append(score)

    return meteors


########################
## Gleu
########################
def calculate_gleu(results, data):
    gleus = []
    for qid, output in results.items():
        score = sentence_gleu([output], data[qid])
        gleus.append(score)

    return gleus


########################
## Readability
########################
def fre_readability(results):
    fres = []
    for qid, output in results.items():
        score = fre(output)
        fres.append(score)

    return fres


def fre(para):
    '''Flesch Reading Ease
    Arguments:
        nsyll:  syllable count
        nwords:  word count
        nsentences:  sentence count
    Returns:
        float:  Flesch reading ease score
    '''
    nsentences = len(para.split("\n"))
    words = para.split()
    nwords = len(words)
    nsyll = 0
    for word in words:
        nsyll += syllables.estimate(word)
    try:
        return 206.835 - (84.6 * (nsyll / nwords)) - (1.015 * (nwords / nsentences))
    except ZeroDivisionError:
        return 0


########################
## Repetition rate
########################
def calculate_repetition_rate(results):
    rates = []
    for qid, output in results.items():
        score = (len(set(output)) / len(output))

        rates.append(score)

    return rates


########################
## Diversity and Novelty
########################
def get_jaccard_sim(str1, str2):
    try:
        a = set(str1.split())
        b = set(str2.split())
        c = a.intersection(b)
        return float(len(c)) / (len(a) + len(b) - len(c))
    except:
        print((str1))
        print(type(str2))
        return 0


def get_diversity(results):
    diversities = []
    for qid1, output1 in results.items():
        max_overlap = 0
        for qid2, output2 in results.items():
            if qid1 != qid2:
                max_overlap = max(max_overlap, get_jaccard_sim(output1, output2))
        diversities.append(1 - max_overlap)

    return diversities


def get_novelty(results, training_corpus):
    novelties = []
    for qid1, output1 in results.items():
        max_overlap = 0
        for qid2, instance in training_corpus.items():
            max_overlap = max(max_overlap, get_jaccard_sim(instance, output1))

        novelties.append(1 - max_overlap)

    return novelties
