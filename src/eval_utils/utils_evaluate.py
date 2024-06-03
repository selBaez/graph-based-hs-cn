'''
Adapted from https://github.com/lupantech/ScienceQA and https://github.com/amazon-science/mm-cot
'''

import re
import warnings

import numpy as np
import syllables
from eval_utils.heavy_metrics import Bleurt, CounterspeechScore, SentenceSimilarityScore
from eval_utils.toxic_metric import ToxicHateXplain
from nltk import word_tokenize
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

warnings.filterwarnings('ignore')


def get_scores_conan(results, results_reference):
    results = {idx: item for idx, item in enumerate(results)}
    results_reference = {idx: item for idx, item in enumerate(results_reference)}

    ## BLEU
    bleu1 = calculate_bleu(results, results_reference, gram=1)
    bleu4 = calculate_bleu(results, results_reference, gram=4)

    ## Rouge-L
    rouge = calculate_rouge(results, results_reference)

    ## Meteor
    meteor = calculate_meteor(results, results_reference)

    ## Gleu
    gleu = calculate_gleu(results, results_reference)

    ## RR

    ## Fre
    fre = fre_readability(results)

    ## Similarity
    similarity_scorer = SentenceSimilarityScore()
    similarity = similarity_scorer.score(results, results_reference)

    ## Bleurt
    bleurt_score = Bleurt(model_path="Elron/bleurt-base-512", max_length=400, batch_size=16)
    bleurt = bleurt_score.score(results, results_reference)

    ## Counter-speech
    counterspeech_score = CounterspeechScore(model_path='Hate-speech-CNERG/counterspeech-quality-bert',
                                             max_length=100, batch_size=16)
    counterspeech = counterspeech_score.scoring(results)

    ## toxicity
    toxicity_score = ToxicHateXplain(model_path="Hate-speech-CNERG/bert-base-uncased-hatexplain-rationale-two",
                                     max_length=100, batch_size=16)
    toxicity = toxicity_score.scoring(results)

    ## Diversity

    ## Novelty

    scores = {'bleu1': bleu1 * 100,
              'bleu4': bleu4 * 100,
              'rouge': rouge * 100,
              'meteor': meteor * 100,
              'gleu': gleu * 100,
              'fre': fre,
              'sentence_similariry': similarity * 100,
              'bleurt': bleurt,
              'counterspeech': counterspeech,
              'toxicity': toxicity,
              }

    scores = {k: float(val) for k, val in scores.items()}

    return scores


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
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1.,))  # BELU-1
    elif gram == 2:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 2., 1. / 2.))  # BELU-2
    elif gram == 3:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 3., 1. / 3., 1. / 3.))  # BELU-3
    elif gram == 4:
        bleu = sentence_bleu([reference_tokens], hypothesis_tokens, (1. / 4., 1. / 4., 1. / 4., 1. / 4.))  # BELU-4

    return bleu


def calculate_bleu(results, data, gram):
    bleus = []
    for qid, output in results.items():
        prediction = output
        target = data[qid]
        target = target.strip()
        if target == "":
            continue
        bleu = bleu_score(target, prediction, gram)
        bleus.append(bleu)

    avg_bleu = sum(bleus) / len(bleus)

    return avg_bleu


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
            continue
        rouge = score_rouge(target, prediction)
        rouges.append(rouge)

    avg_rouge = sum(rouges) / len(rouges)
    return avg_rouge


########################
## Meteor
########################
def calculate_meteor(results, data):
    meteors = []
    for qid, output in results.items():
        hypothesis_tokens = word_tokenize(output)
        reference_tokens = word_tokenize(data[qid])
        score = meteor_score([hypothesis_tokens], reference_tokens)
        meteors.append(score)

    return np.mean(meteors)


########################
## Gleu
########################
def calculate_gleu(results, data):
    gleus = []
    for qid, output in results.items():
        hypothesis_tokens = word_tokenize(output)
        reference_tokens = word_tokenize(data[qid])
        score = sentence_gleu([hypothesis_tokens], reference_tokens)
        gleus.append(score)

    return np.mean(gleus)


########################
## Readability
########################
def fre_readability(results):
    fres = []
    for qid, output in results.items():
        score = fre(output)
        fres.append(score)

    return np.mean(fres)


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
