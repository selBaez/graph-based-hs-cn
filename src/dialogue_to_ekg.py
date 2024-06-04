# Imports
import argparse
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from random import getrandbits

import networkx as nx
import numpy as np
import requests
from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.entity_linking.label_linker import LabelBasedLinker
from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
from cltl.triple_extraction.conversational_analyzer import ConversationalAnalyzer
from cltl.triple_extraction.oie_analyzer import OIEAnalyzer
from cltl.triple_extraction.spacy_analyzer import spacyAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules
from rdflib import ConjunctiveGraph, URIRef, RDF, RDFS, OWL
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from tqdm import tqdm

from dialogue_to_got import save_data

# Set logging levels
chat_logger.setLevel(logging.ERROR)
brain_logger.setLevel(logging.ERROR)
max_nodes = 100

ONTOLOGY_DETAILS = {"filepath": "./../data/ontology.ttl",
                    "namespace": "http://cltl-hs.org/",
                    "prefix": "hs"}
GAF_DENOTEDIN = URIRef("http://groundedannotationframework.org/gaf#denotedIn")
GAF_DENOTEDBY = URIRef("http://groundedannotationframework.org/gaf#denotedBy")
GAF_CONTAINSDEN = URIRef("http://groundedannotationframework.org/gaf#containsDenotation")
GRASP_ATTFOR = URIRef("http://groundedannotationframework.org/grasp#isAttributionFor")
GRASP_ATTTO = URIRef("http://groundedannotationframework.org/grasp#wasAttributedTo")
SEM_TST = URIRef("http://semanticweb.cs.vu.nl/2009/11/sem/hasBeginTimeStamp")
HS_ID = URIRef("http://cltl-hs.org/id")

INSTANCE_GRAPH = URIRef("http://cltl.nl/leolani/world/Instances")
PERSPECTIVE_GRAPH = URIRef("http://cltl.nl/leolani/talk/Perspectives")
ONTOLOGY_GRAPH = URIRef("http://cltl.nl/leolani/world/Ontology")

"""
[' <s> i </s>
<s>  take </s>
<s>  second here </s>
<s>  second </s>
<s>  say </s>
<s>  something positive about gay community </s>
<s>  epidemic </s>
<s>  is in </s>
<s>  80s </s>
<s>  you </s>
<s>  so ’re saying </s>
<s>  homos </s>
<s>  they </s>
<s>  are </s>
<s>  so positive']

[['i', 'take', 'second here'],
[ 'second', 'say', 'something positive about gay community'],
[ 'epidemic', 'is in', '80s'],
[ 'you', 'so ’re saying', 'homos'],
[ 'they', 'are', 'so positive']]
"""


def format_for_string_encoding(graph_data):
    temp_text = " <s> "

    contexts = sorted([c for c in graph_data.contexts()])
    for context in contexts:
        triples = sorted([t for t in context],
                         key=lambda s: (not s[0].startswith('http://cltl.nl/leolani/talk/chat'), s))

        for triple in triples:
            for el in triple:
                el = el.split("/")[-1]

                tmp_el = el.split("#")
                if tmp_el[0] in ["factuality", "sentiment", "emotion"]:
                    el = el.replace("#", "_")
                else:
                    el = tmp_el[-1]

                if temp_text == ' <s> ':
                    temp_text += f"{el} "
                else:
                    temp_text += f"</s> <s> {el} "

    return temp_text


def format_adj_matrix(graph_data):
    g = rdflib_to_networkx_graph(graph_data)
    g.remove_nodes_from(list(nx.isolates(g)))
    adj_temp = nx.adjacency_matrix(g)
    adj_temp = adj_temp.todense()

    # Fix dimensions
    num_nodes = adj_temp.shape[0]
    if num_nodes > max_nodes:
        # We need to remove nodes with less connections. add per row and go from there?
        degrees = np.sum(adj_temp, axis=0)
        degrees_idx = sorted(range(len(degrees)), key=lambda k: degrees[k])[:max_nodes]
        degrees_idx = sorted(degrees_idx)
        adj_temp = adj_temp[np.ix_(degrees_idx, degrees_idx)]

    elif num_nodes < max_nodes:
        # we need to add zeros to get the right dimensions
        missing_nodes = max_nodes - num_nodes
        adj_temp = np.pad(adj_temp, ((0, missing_nodes), (0, missing_nodes)))

    return adj_temp


def remove_empty_contexts(graph_data):
    new_cg = ConjunctiveGraph()
    for context in graph_data.contexts():
        if len(context) != 0 and str(context.identifier).startswith("http"):
            new_context = new_cg.get_context(context.identifier)
            for triple in context.triples((None, None, None)):
                new_context.add(triple)
    return new_cg


def clear_context_nodes(graph_data):
    """
    Query graph for claims
    """
    q_claims = """SELECT distinct ?node  WHERE {{ ?node ?p ?o . 
                FILTER(STRSTARTS(STR(?node), STR(leolaniContext:))) . }}"""
    all_ctx_nodes = graph_data.query(q_claims)
    all_ctx_nodes = [c for c in all_ctx_nodes]

    for ctx in all_ctx_nodes:
        graph_data.remove((ctx[0], None, None))

    return graph_data


def clear_claims_as_context(graph_data):
    """
    Query graph for claims
    """
    q_claims = """SELECT distinct ?claim  WHERE {{ ?claim rdf:type gaf:Assertion . }}"""
    all_claims = graph_data.query(q_claims)
    all_claims = [c for c in all_claims]
    print(f"CLAIMS IN DATASET: {len(all_claims)}")

    for claim in all_claims:
        graph_data = clear_context(graph_data, claim[0])

    return graph_data


def clear_context(graph_data, context_id):
    context = graph_data.get_context(context_id)

    for triple in context.triples((None, None, None)):
        graph_data.remove(triple)

    return graph_data


def clean_graph(graph_data):
    # Remove redundant
    graph_data.remove((None, GAF_DENOTEDIN, None))
    graph_data.remove((None, GAF_CONTAINSDEN, None))
    graph_data.remove((None, GAF_DENOTEDBY, None))
    graph_data.remove((None, GRASP_ATTFOR, None))
    graph_data.remove((None, GRASP_ATTTO, None))

    # Remove details of instances
    graph_data = clear_context(graph_data, ONTOLOGY_GRAPH)
    graph_data = clear_context(graph_data, INSTANCE_GRAPH)
    # graph_data = clear_context(graph_data, PERSPECTIVE_GRAPH)
    graph_data = clear_claims_as_context(graph_data)
    graph_data = clear_context_nodes(graph_data)

    # Remove not significant
    graph_data.remove((None, SEM_TST, None))
    graph_data.remove((None, OWL.sameAs, None))
    graph_data.remove((None, RDFS.label, None))
    graph_data.remove((None, RDF.type, None))
    graph_data.remove((None, HS_ID, None))

    # New graph
    graph_data = remove_empty_contexts(graph_data)

    return graph_data


def download_from_triplestore(scenario_filepath, brain):
    # aggregated data
    response = brain._connection.export_repository()
    graph_data = ConjunctiveGraph()
    graph_data.parse(data=response, format="trig")

    # save
    raw_file = scenario_filepath / f"full_graph.trig"
    graph_data.serialize(destination=raw_file, format="trig")

    return graph_data


def save_scenario_data(scenario_filepath, all_capsules, all_responses):
    # Time step data
    f = open(scenario_filepath / "capsules.json", "w")
    json.dump(all_capsules, f)
    f = open(scenario_filepath / "responses.json", "w")
    json.dump(all_responses, f)


def analyze_utterance(analyzer, chat, args=None):
    if args.analyzer in ["CFG", "spacy", "OIE"]:
        analyzer.analyze(chat.last_utterance)
    elif args.analyzer == "albert":
        analyzer.analyze_in_context(chat)
    else:
        print(f"ANALYZER option: {args.analyzer} not implemented")


def create_and_submit_context_capsule(brain):
    # Define contextual features
    context_id = getrandbits(8)
    place_id = getrandbits(8)
    location = requests.get("https://ipinfo.io").json()
    start_date = date(2024, 4, 1)

    context_capsule = {"context_id": context_id,
                       "date": start_date,
                       "place": "Online forum",
                       "place_id": place_id,
                       "country": location['country'],
                       "region": location['region'],
                       "city": location['city']}

    brain.capsule_context(context_capsule)


def make_output_directory(args, dialogue, split):
    scenario_filepath = Path(f"{args.output_dir}/{split}/{dialogue['dialogue_id']}/")
    graph_filepath = scenario_filepath / Path("graph/")
    graph_filepath.mkdir(parents=True, exist_ok=True)

    return graph_filepath, scenario_filepath


def read_data(args, split):
    filename = f"{args.data_root}/{args.dataset}/{split}.json"
    with open(filename, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"Read dataset: {filename}")

    return logs


def process_dialogues(split, analyzer, linker, timers):
    # read dataset
    logs = read_data(args, split)

    # loop through dialogues
    mc_input_text_list, mc_adj_matrix_list = [], []
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")
        dialogue["utterances"] = [{"text": utt[4:], "speaker": utt[:2]} for utt in
                                  dialogue["dialogue_history"].split("\n")]
        dialogue["utterances"].append({"text": dialogue["hate_speech"], "speaker": "HS"})

        # Create folders
        graph_filepath, scenario_filepath = make_output_directory(args, dialogue, split)

        # Initialize brain, Chat,
        brain = LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                               # Location to save accumulated graph
                               log_dir=graph_filepath,  # Location to save step-wise graphs
                               ontology_details=ONTOLOGY_DETAILS,
                               clear_all=True)  # To start from an empty brain
        chat = Chat("CN", "HS")

        # Create context
        print(f"\tCreating context")
        create_and_submit_context_capsule(brain)

        # convert to eKG
        print(f"\n\tProcessing {len(dialogue['utterances'])} utterances")
        all_responses, all_capsules = [], []
        capsules_skipped = 0
        for utterance in dialogue["utterances"]:
            # add utterance to chat and use CFG analyzer to analyze
            print(f"\t\tAnalyze utterance, (accumulated time: {timers['time_analyzer']})")
            chat.add_utterance(utterance["text"], utterance["speaker"])
            this_time_analyzer = time.time()
            analyze_utterance(analyzer, chat, args=args)
            timers['time_analyzer'] += time.time() - this_time_analyzer
            capsules = utterance_to_capsules(chat.last_utterance)

            # add statement capsules to brain
            if len(capsules) > 0:
                print(f"\t\t\tProcessing {len(capsules)} capsules")
            for capsule in capsules:
                try:
                    # Ugly fix of capsule
                    capsule['author'] = {"label": utterance["speaker"], "type": ["person"]}
                    capsule['timestamp'] = datetime.now()

                    # Link to specific instances
                    print(f"\t\t\t\tLink capsule (accumulated time: {timers['time_linker']})")
                    this_time_linker = time.time()
                    linker.link(capsule)
                    timers['time_linker'] += time.time() - this_time_linker

                    # Add capsule to brain
                    print(f"\t\t\t\tAdding capsule to brain (accumulated time: {timers['time_brain']})")
                    this_time_brain = time.time()
                    response = brain.capsule_statement(capsule, reason_types=True,
                                                       create_label=True, return_thoughts=False)
                    timers['time_brain'] += time.time() - this_time_brain

                    # Keep track of responses
                    capsule['rdf_file'] = str(response['rdf_log_path'].stem) + '.trig'
                    capsule_json = brain_response_to_json(capsule)
                    all_capsules.append(capsule_json)
                    response_json = brain_response_to_json(response)
                    all_responses.append(response_json)
                except Exception as e:
                    capsules_skipped += 1
                    print(f"\t\t\tCapsule skipped. Total skipped: {capsules_skipped}")
                    print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

        # Save responses
        save_scenario_data(scenario_filepath, all_capsules, all_responses)

        # Postprocessing
        graph_data = download_from_triplestore(scenario_filepath, brain)
        graph_data = clean_graph(graph_data)

        mc_adj_matrix = format_adj_matrix(graph_data)
        mc_input_text = format_for_string_encoding(graph_data)
        mc_input_text_list.append(mc_input_text)
        mc_adj_matrix_list.append(mc_adj_matrix)

    # Save data
    save_data(mc_input_text_list, mc_adj_matrix_list, Path(f"{args.output_dir}/{split}/"), args)


def create_analyzer(args):
    if args.analyzer == "CFG":
        analyzer = CFGAnalyzer()
    elif args.analyzer == "albert":
        path = '/Users/sbaez/Documents/PhD/leolani/cltl-knowledgeextraction/resources/conversational_triples/albert-base-v2'
        base_model = 'albert-base-v2'
        lang = 'en'
        analyzer = ConversationalAnalyzer(model_path=path, base_model=base_model, lang=lang)
    elif args.analyzer == "spacy":
        analyzer = spacyAnalyzer()
    elif args.analyzer == "OIE":
        analyzer = OIEAnalyzer()
    else:
        print(f"ANALYZER option: {args.analyzer} not implemented")
        return None
    return analyzer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--splits', nargs="+", default=["train", "dev", "test"])  # "mini-test"])
    parser.add_argument('--output_dir', type=str, default='./../data/DIALOCONAN/ekg/')
    parser.add_argument('--input_text_file', type=str, default='mc_input_text.pkl')
    parser.add_argument('--adj_matrix_file', type=str, default='mc_adj_matrix.pkl')
    parser.add_argument('--analyzer', type=str, default='OIE', choices=["CFG", "albert", "spacy", "OIE"])

    args = parser.parse_args()
    return args


def main(args):
    # Create analyzers
    analyzer = create_analyzer(args)
    linker = LabelBasedLinker()

    # time each function separately to see what is taking so long
    timers = {"time_analyzer": 0, "time_linker": 0, "time_brain": 0}

    # Process dialogues
    for split in args.splits:
        process_dialogues(split, analyzer, linker, timers)


if __name__ == "__main__":
    args = parse_args()
    main(args)

    print("Done")
