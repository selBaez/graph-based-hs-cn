import argparse
import pickle
import string
from pathlib import Path

import neuralcoref
import numpy as np
import spacy
import stanza
from stanza.server import CoreNLPClient
from tqdm import tqdm

from train_utils.utils_data import load_data

stanza.install_corenlp()
nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp)  # Add neural coref to SpaCy's pipe
punc = string.punctuation
alphabets = "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr|Prof|Capt|Cpt|Lt|Mt)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov|me|edu)"
max_nodes = 100


def coreference(s):
    doc = nlp(s)
    return doc._.coref_clusters


def extract_triples(document):
    triples = []
    for sent in document.sentence:
        for triple in sent.openieTriple:
            subject = getattr(triple, 'subject')
            relation = getattr(triple, 'relation')
            object = getattr(triple, 'object')

            triples.append({'subject': subject, 'relation': relation, 'object': object})
    return triples


def compress_triple(annotate_result, coref):
    triples = extract_triples(annotate_result)

    temp_set = []
    for i in range(0, len(triples)):
        cur = triples[i]
        cur_subject = cur['subject'].lower()
        cur_relation = cur['relation'].lower()
        cur_object = cur['object'].lower()

        for cluster in coref:
            span = [w.text.lower() for w in cluster.mentions]
            if cur_subject in span:
                cur_subject = cluster.main.text.lower()
            if cur_object in span:
                cur_object = cluster.main.text.lower()

        if len(temp_set) == 0:
            temp_set.append([cur_subject, cur_relation, cur_object])
        else:
            flag = 0
            # print(temp_set)
            for j in range(0, len(temp_set)):
                ###save the longest when have two same entities
                if temp_set[j][0] == cur_subject and temp_set[j][1] == cur_relation:

                    if len(cur_object) > len(temp_set[j][2]):
                        temp_set[j][2] = cur_object
                    flag = 1

                elif temp_set[j][0] == cur_subject and temp_set[j][2] == cur_object:
                    if len(cur_relation) > len(temp_set[j][1]):
                        temp_set[j][1] = cur_relation
                    flag = 1

                elif temp_set[j][2] == cur_object and temp_set[j][1] == cur_relation:
                    if len(cur_subject) > len(temp_set[j][0]):
                        temp_set[j][0] = cur_subject
                    flag = 1

            if flag == 0:
                ##if no editing, then it is a new triplet, add to temp
                temp_set.append([cur_subject, cur_relation, cur_object])

    return temp_set


def get_mind_chart(mc_context, max_nodes, client):
    """get mind chart

    Args:
        mc_context (string): the context to construct mind chart (question+" "+context+" "+lecture+" "+solution+" "+choice)

    Returns:
        triples(list of triplets list): [[I, love, NLP],[NLP,is,fun]]
        action_input(list):["I</s><s>love</s><s>NLP</s><s>NLP</s><s>is</s><s>fun"]
        action_adj(list): [adjecent matrix] 
    """
    mc_context = mc_context.replace("\n", " ")
    coref = coreference(mc_context)

    mc_context = mc_context.replace("\n", " ")
    annotate_result = client.annotate(mc_context)
    triples = compress_triple(annotate_result, coref)

    action_input = []

    id2node = {}
    node2id = {}
    adj_temp = np.zeros([max_nodes, max_nodes])
    index = 0
    if len(triples) == 0:
        action_input.append('<pad>')
    else:
        temp_text = ' <s> '
        for u in triples:
            if u[0] not in node2id:
                node2id[u[0]] = index
                id2node[index] = u[0]
                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[0]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[0]

                    index = index + 1
                else:
                    break
            if u[1] not in node2id:
                node2id[u[1]] = index
                id2node[index] = u[1]

                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[1]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[1]
                    index = index + 1
                else:
                    break

            if u[2] not in node2id:
                node2id[u[2]] = index
                id2node[index] = u[2]

                if index < max_nodes:
                    if temp_text == ' <s> ':
                        temp_text = temp_text + u[2]
                    else:
                        temp_text = temp_text + ' </s> <s> ' + u[2]
                    index = index + 1
                else:
                    break

            adj_temp[node2id[u[0]]][node2id[u[0]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[2]]] = 1

            adj_temp[node2id[u[0]]][node2id[u[1]]] = 1
            adj_temp[node2id[u[1]]][node2id[u[0]]] = 1

            adj_temp[node2id[u[1]]][node2id[u[2]]] = 1
            adj_temp[node2id[u[2]]][node2id[u[1]]] = 1

        action_input.append(temp_text)
    # action_adj.append(adj_temp)

    return action_input, adj_temp


def make_output_directory(args, split):
    # Make path for output
    outpath = Path(args.output_dir) / f"{split}/"
    outpath.mkdir(parents=True, exist_ok=True)

    # Check if the files exist already?
    # if os.path.isfile(args.input_text_path) or os.path.isfile(args.adj_matrix_path):
    #     assert False

    return outpath


def save_data(mc_input_text_list, mc_adj_matrix_list, outpath, args):
    mc_input_text_path = outpath / args.input_text_file
    with open(mc_input_text_path, 'wb') as f:
        pickle.dump(mc_input_text_list, f)

    mc_adj_matrix_path = outpath / args.adj_matrix_file
    with open(mc_adj_matrix_path, 'wb') as f:
        pickle.dump(mc_adj_matrix_list, f)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--splits', nargs="+", default=["train", "dev", "test"])  # "mini-test"])
    parser.add_argument('--output_dir', type=str, default='./../data/DIALOCONAN/got/')
    parser.add_argument('--input_text_file', type=str, default='mc_input_text.pkl')
    parser.add_argument('--adj_matrix_file', type=str, default='mc_adj_matrix.pkl')
    parser.add_argument('--exclude_context', action='store_true', help='remove dialogue history from the prompt')

    args = parser.parse_args()
    return args


def main(args):
    for split in args.splits:
        # Create directories
        outpath = make_output_directory(args, split)

        # Read data
        dialogues = load_data(args, split)

        # Analyze
        mc_input_text_list, mc_adj_matrix_list = [], []
        with CoreNLPClient(annotators=["ner", "openie", "coref"], memory='4G',
                           endpoint='http://localhost:2727', be_quiet=True) as client:

            for dialogue in tqdm(dialogues):
                dialogue_history = dialogue["dialogue_history"]
                hs = dialogue["hate_speech"]

                if args.exclude_context:
                    mc_context_text = f"{hs}"
                else:
                    mc_context_text = f"{dialogue_history}\n{hs}"

                mc_input_text, mc_adj_matrix = get_mind_chart(mc_context_text, max_nodes, client)
                mc_input_text_list.append(mc_input_text)
                mc_adj_matrix_list.append(mc_adj_matrix)

            client.stop()

        # Save data
        save_data(mc_input_text_list, mc_adj_matrix_list, outpath, args)


if __name__ == '__main__':
    args = parse_args()
    main(args)
