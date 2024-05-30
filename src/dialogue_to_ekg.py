# Imports
import json
import logging
import time
from datetime import date, datetime
from pathlib import Path
from random import getrandbits

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
from tqdm import tqdm

# Set logging levels
chat_logger.setLevel(logging.ERROR)
brain_logger.setLevel(logging.ERROR)

ANALYZER = "OIE"  # one of: CFG, albert, spacy, OIE


def create_and_submit_context_capsule(brain):
    # Define contextual features
    context_id = getrandbits(8)
    place_id = getrandbits(8)
    location = requests.get("https://ipinfo.io").json()
    start_date = date(2021, 3, 12)

    context_capsule = {"context_id": context_id,
                       "date": start_date,
                       "place": "Unknown",
                       "place_id": place_id,
                       "country": location['country'],
                       "region": location['region'],
                       "city": location['city']}

    brain.capsule_context(context_capsule)


def create_analyzer():
    if ANALYZER == "CFG":
        analyzer = CFGAnalyzer()
    elif ANALYZER == "albert":
        path = '/Users/sbaez/Documents/PhD/leolani/cltl-knowledgeextraction/resources/conversational_triples/albert-base-v2'
        base_model = 'albert-base-v2'
        lang = 'en'
        analyzer = ConversationalAnalyzer(model_path=path, base_model=base_model, lang=lang)
    elif ANALYZER == "spacy":
        analyzer = spacyAnalyzer()
    elif ANALYZER == "OIE":
        analyzer = OIEAnalyzer()
    else:
        print(f"ANALYZER option: {ANALYZER} not implemented")
        return None
    return analyzer


def analyze_utterance(analyzer, chat):
    if ANALYZER in ["CFG", "spacy", "OIE"]:
        analyzer.analyze(chat.last_utterance)
    elif ANALYZER == "albert":
        analyzer.analyze_in_context(chat)
    else:
        print(f"ANALYZER option: {ANALYZER} not implemented")


def time_process(function, timer):
    now = time.time()
    tmp = function()
    timer += time.time() - now

    return tmp, timer


def main():
    # read dataset
    filename = "./../data/KDIALOCONAN_grounded_gold.json"
    with open(filename, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"Read dataset: {filename}")

    # Create analyzers
    analyzer = create_analyzer()
    linker = LabelBasedLinker()

    # time each function separately to see what is taking so long
    time_analyzer, time_linker, time_brain = 0, 0, 0

    # loop through dialogues
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")

        # Create folders
        scenario_filepath = Path(f"./../data/KDIALOCONAN_grounded_gold({ANALYZER})/{dialogue['dialogue_id']}/")
        graph_filepath = scenario_filepath / Path("graph/")
        graph_filepath.mkdir(parents=True, exist_ok=True)

        # Initialize brain, Chat,
        brain = LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                               # Location to save accumulated graph
                               log_dir=graph_filepath,  # Location to save step-wise graphs
                               clear_all=True)  # To start from an empty brain
        chat = Chat("CN", "HS")

        # Create context
        print(f"\tCreating context")
        create_and_submit_context_capsule(brain)

        # convert to eKG
        print(f"\n\tProcessing {len(dialogue['utterances'])} utterances")
        all_responses = []
        all_capsules = []
        capsules_skipped = 0
        for utterance in dialogue["utterances"]:
            # add utterance to chat and use CFG analyzer to analyze
            print(f"\t\tAnalyze utterance: {utterance['turn_id']} (accumulated time: {time_analyzer})")
            chat.add_utterance(utterance["text"], utterance["speaker"])
            # _, time_analyzer = time_process(lambda: analyze_utterance(analyzer, chat), time_analyzer)
            this_time_analyzer = time.time()
            analyze_utterance(analyzer, chat)
            time_analyzer += time.time() - this_time_analyzer
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
                    print(f"\t\t\t\tLink capsule (accumulated time: {time_linker})")
                    # _, time_linker = time_process(lambda: linker.link(capsule), time_linker)
                    this_time_linker = time.time()
                    linker.link(capsule)
                    time_linker += time.time() - this_time_linker

                    # Add capsule to brain
                    print(f"\t\t\t\tAdding capsule to brain (accumulated time: {time_brain})")
                    # print(f"Utterance: {utterance}, Capsule: {json.dumps(capsule_json, indent=2)}")
                    # response, time_brain = time_process(lambda: brain.capsule_statement(capsule, reason_types=True,
                    #                                                                     create_label=True,
                    #                                                                     return_thoughts=False),
                    #                                     time_brain)
                    this_time_brain = time.time()
                    response = brain.capsule_statement(capsule, reason_types=True,
                                                       create_label=True, return_thoughts=False)
                    time_brain += time.time() - this_time_brain

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
        f = open(scenario_filepath / "capsules.json", "w")
        json.dump(all_capsules, f)
        f = open(scenario_filepath / "responses.json", "w")
        json.dump(all_responses, f)


if __name__ == "__main__":
    main()
