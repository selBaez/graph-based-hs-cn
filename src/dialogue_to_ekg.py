# Imports
import argparse
import gc
import json
import logging
import pickle
from datetime import date, datetime
from pathlib import Path
from random import getrandbits

import networkx as nx
import numpy as np
import requests
from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.emotion_extraction.utterance_go_emotion_extractor import GoEmotionDetector
from cltl.entity_linking.label_linker import LabelBasedLinker
from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.oie_analyzer import OIEAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules
from rdflib import ConjunctiveGraph, RDF, RDFS, OWL
from rdflib.extras.external_graph_libs import rdflib_to_networkx_graph
from tqdm import tqdm

from utils.constants import ONTOLOGY_DETAILS, GAF_DENOTEDIN, GAF_DENOTEDBY, GAF_CONTAINSDEN, GRASP_ATTFOR, \
    GRASP_ATTTO, SEM_TST, HS_ID, INSTANCE_GRAPH, ONTOLOGY_GRAPH

# Set logging levels
chat_logger.setLevel(logging.ERROR)
brain_logger.setLevel(logging.ERROR)
max_nodes = 100


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

    if len(g) > 0:
        adj_temp = nx.adjacency_matrix(g)
        adj_temp = adj_temp.todense()
    else:
        adj_temp = np.zeros([max_nodes, max_nodes])

    # Fix dimensions
    num_nodes = adj_temp.shape[0]
    print(f"\tNodes in simple graph: {num_nodes}")
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
    print(f"\tClaims in dataset: {len(all_claims)}")

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


def merge_all_graphs(scenario_filepath):
    # use folder with latest timestamp
    graph_path = scenario_filepath / Path("graph/")
    runs = sorted([f for f in graph_path.iterdir() if f.is_dir()], reverse=True)
    rdf_files_path = runs[-1]
    trig_files = list(rdf_files_path.glob(f'*.trig'))

    # aggregated data
    graph_data = ConjunctiveGraph()
    for trig_file in trig_files:
        graph_data.parse(trig_file, format="trig")

    # save
    raw_file = scenario_filepath / f"full_graph.trig"
    graph_data.serialize(destination=raw_file, format="trig")

    return graph_data


def analyze_utterance(analyzer, chat, args=None):
    if args.analyzer in ["CFG", "spacy", "OIE"]:
        analyzer.analyze(chat.last_utterance)
    elif args.analyzer == "albert":
        analyzer.analyze_in_context(chat)
    else:
        print(f"ANALYZER option: {args.analyzer} not implemented")


def create_analyzer(args):
    # if args.analyzer == "CFG":
    #     analyzer = CFGAnalyzer()
    # elif args.analyzer == "albert":
    #     path = '/Users/sbaez/Documents/PhD/leolani/cltl-knowledgeextraction/resources/conversational_triples/albert-base-v2'
    #     base_model = 'albert-base-v2'
    #     lang = 'en'
    #     analyzer = ConversationalAnalyzer(model_path=path, base_model=base_model, lang=lang)
    # elif args.analyzer == "spacy":
    #     analyzer = spacyAnalyzer()
    if args.analyzer == "OIE":
        analyzer = OIEAnalyzer()
    else:
        print(f"ANALYZER option: {args.analyzer} not implemented")
        return None
    return analyzer


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
                       "country": "MX",  # location['country'],
                       "region": "Mexico City",  # location['region'],
                       "city": "Mexico City"}  # location['city']}

    brain.capsule_context(context_capsule)


def make_output_directory(args, dialogue, split):
    scenario_filepath = Path(f"{args.output_dir}/{split}/{dialogue['dialogue_id']}/")
    graph_filepath = scenario_filepath / Path("graph/")
    graph_filepath.mkdir(parents=True, exist_ok=True)

    return graph_filepath, scenario_filepath


def save_scenario_data(scenario_filepath, all_capsules, all_responses):
    # Time step data
    if all_capsules:
        f = open(scenario_filepath / "capsules.json", "w")
        json.dump(all_capsules, f)

    if all_responses:
        f = open(scenario_filepath / "responses.json", "w")
        json.dump(all_responses, f)


def save_data(mc_input_text_list, mc_adj_matrix_list, outpath, args):
    mc_input_text_path = outpath / args.input_text_file
    with open(mc_input_text_path, 'wb') as f:
        pickle.dump(mc_input_text_list, f)

    mc_adj_matrix_path = outpath / args.adj_matrix_file
    with open(mc_adj_matrix_path, 'wb') as f:
        pickle.dump(mc_adj_matrix_list, f)


def read_data(args, split):
    filename = f"{args.data_root}/{args.dataset}/{split}.json"
    with open(filename, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"Read dataset: {filename}")

    return logs


def dialogue_as_capsules(split, logs, analyzer, emotion_classifier):
    # loop through dialogues
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")
        dialogue["utterances"] = [{"text": u[4:], "speaker": u[:2]} for u in dialogue["dialogue_history"].split("\n")]
        dialogue["utterances"].append({"text": dialogue["hate_speech"], "speaker": "HS"})

        # Create folders
        _, scenario_filepath = make_output_directory(args, dialogue, split)

        # Create chat
        print(f"\t\tCreating chat")
        chat = Chat("CN", "HS")
        chat.id = dialogue['dialogue_id']

        # convert to capsules
        print(f"\n\tProcessing {len(dialogue['utterances'])} utterances")
        all_capsules, utterances_skipped, emotions_skipped = [], 0, 0
        for utterance in dialogue["utterances"]:
            # add utterance to chat and analyze
            try:
                print(f"\t\tAnalyze utterance")
                chat.add_utterance(utterance["text"], utterance["speaker"])
                analyze_utterance(analyzer, chat, args=args)
                capsules = utterance_to_capsules(chat.last_utterance)
            except Exception as e:
                capsules = []
                utterances_skipped += 1
                print(f"\t\t\tUtterance skipped. Total skipped: {utterances_skipped}")
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

            # Get emotion
            try:
                print(f"\t\tExtract emotion")
                emotion = emotion_classifier.extract_text_emotions(utterance["text"])
                emotion = f"EmotionType.GO:{emotion[0].value.upper()}"

                for capsule in capsules:
                    # Ugly fix of capsule
                    capsule['author'] = {"label": utterance["speaker"], "type": ["person"]}
                    capsule['timestamp'] = datetime.now()
                    capsule["perspective"]["emotion"] = emotion

                    # append to list
                    capsule_json = brain_response_to_json(capsule)
                    all_capsules.append(capsule_json)
            except Exception as e:
                emotions_skipped += 1
                print(f"\t\t\tUtterance skipped. Total skipped: {emotions_skipped}")
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

        # Save responses
        save_scenario_data(scenario_filepath, all_capsules, None)


def dialogue_as_rdf(split, logs, linker):
    # loop through dialogues
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")

        # Create folders
        graph_filepath, scenario_filepath = make_output_directory(args, dialogue, split)

        # Get capsules
        try:
            with open(scenario_filepath / "capsules.json") as f:
                capsules = json.load(f)
        except:
            capsules = []

        # Initialize brain, Chat,
        brain = LongTermMemory(address="http://localhost:7200/repositories/sandbox",  # Accumulated graph
                               log_dir=graph_filepath,  # Location to save step-wise graphs
                               ontology_details=ONTOLOGY_DETAILS,
                               clear_all=True)  # To start from an empty brain

        # Create context
        print(f"\tCreating context")
        create_and_submit_context_capsule(brain)

        # convert to eKG
        print(f"\t\t\tProcessing {len(capsules)} capsules")
        all_responses, capsules_skipped = [], 0
        for capsule in capsules:
            try:
                # Link to specific instances
                print(f"\t\t\t\tLink capsule")
                linker.link(capsule)

                # Add capsule to brain
                print(f"\t\t\t\tAdding capsule to brain")
                response = brain.capsule_statement(capsule, reason_types=False,  # reason types makes it super slow
                                                   create_label=True, return_thoughts=False)

                # Keep track of responses
                capsule['rdf_file'] = str(response['rdf_log_path'].stem) + '.trig'
                response_json = brain_response_to_json(response)
                all_responses.append(response_json)
            except Exception as e:
                capsules_skipped += 1
                print(f"\t\t\tCapsule skipped. Total skipped: {capsules_skipped}")
                print(f"{type(e).__name__} at line {e.__traceback__.tb_lineno} of {__file__}: {e}")

        # Save responses
        save_scenario_data(scenario_filepath, None, all_responses)


def dialogues_as_matrix(split, logs):
    # loop through dialogues
    mc_input_text_list, mc_adj_matrix_list = [], []
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")

        # Create folders
        _, scenario_filepath = make_output_directory(args, dialogue, split)

        # Postprocessing
        graph_data = merge_all_graphs(scenario_filepath)
        graph_data = clean_graph(graph_data)

        mc_adj_matrix = format_adj_matrix(graph_data)
        mc_input_text = format_for_string_encoding(graph_data)
        mc_input_text_list.append(mc_input_text)
        mc_adj_matrix_list.append(mc_adj_matrix)

    # Save data
    save_data(mc_input_text_list, mc_adj_matrix_list, Path(f"{args.output_dir}/{split}/"), args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--splits', nargs="+", default=["test", "dev", "train"])  # "mini-test"])
    parser.add_argument('--output_dir', type=str, default='./../data/DIALOCONAN/ekg/')
    parser.add_argument('--input_text_file', type=str, default='mc_input_text.pkl')
    parser.add_argument('--adj_matrix_file', type=str, default='mc_adj_matrix.pkl')
    parser.add_argument('--analyzer', type=str, default='OIE', choices=["CFG", "albert", "spacy", "OIE"])
    parser.add_argument('--start_step', type=str, default='merge', choices=["analyze", "link", "merge"])
    parser.add_argument('--stop_step', type=str, default='format', choices=["emotion", "brain", "format"])

    args = parser.parse_args()
    return args


def main(args):
    # Create analyzers
    analyzer = create_analyzer(args)
    linker = LabelBasedLinker()
    emotion_classifier = GoEmotionDetector()

    # Process dialogues
    for split in args.splits:
        # read dataset
        logs = read_data(args, split)

        if args.start_step == "analyze":
            # create capsules
            dialogue_as_capsules(split, logs, analyzer, emotion_classifier)
            gc.collect()

        if args.stop_step == "emotion":
            continue

        if args.start_step in ["link", "analyze"]:
            # Create RDF
            dialogue_as_rdf(split, logs, linker)
            gc.collect()

        if args.stop_step in ["emotion", "brain"]:
            continue

        if args.start_step in ["merge", "link", "analyze"]:
            # create adj matrix
            dialogues_as_matrix(split, logs)
            gc.collect()


if __name__ == "__main__":
    args = parse_args()
    main(args)

    print("Done")
