# Imports
import json
import logging
import requests
import time
from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.entity_linking.label_linker import LabelBasedLinker
from datetime import date, datetime
from pathlib import Path
from random import getrandbits
from tqdm import tqdm

from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
# from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
# from cltl.triple_extraction.spacy_analyzer import spacyAnalyzer
from cltl.triple_extraction.conversational_analyzer import ConversationalAnalyzer
from cltl.triple_extraction.utils.helper_functions import utterance_to_capsules

# Set logging levels
chat_logger.setLevel(logging.ERROR)
brain_logger.setLevel(logging.ERROR)


def create_context_capsule():
    # Define contextual features
    context_id = getrandbits(8)
    place_id = getrandbits(8)
    location = requests.get("https://ipinfo.io").json()
    start_date = date(2021, 3, 12)

    return {"context_id": context_id,
            "date": start_date,
            "place": "Unknown",
            "place_id": place_id,
            "country": location['country'],
            "region": location['region'],
            "city": location['city']}


def main():
    # read dataset
    filename = "./../data/KDIALOCONAN_grounded_gold.json"
    with open(filename, 'r', encoding='utf-8') as f:
        logs = json.load(f)
    print(f"Read dataset: {filename}")

    # Create analyzers
    # analyzer = CFGAnalyzer()
    path = '/Users/sbaez/Documents/PhD/leolani/cltl-knowledgeextraction/resources/conversational_triples/albert-base-v2'
    base_model = 'albert-base-v2'
    lang = 'en'
    analyzer = ConversationalAnalyzer(model_path=path, base_model=base_model, lang=lang)
    linker = LabelBasedLinker()

    # time each function separately to see what is taking so long
    time_analyzer = 0
    time_linker = 0
    time_brain = 0

    # loop through dialogues
    for dialogue in tqdm(logs):
        print(f"\n\nProcessing dialogue: {dialogue['dialogue_id']}")

        # Create folders
        scenario_filepath = Path(f"./../data/KDIALOCONAN_grounded_gold/{dialogue['dialogue_id']}/")
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
        context_capsule = create_context_capsule()
        brain.capsule_context(context_capsule)

        # convert to eKG
        print(f"\n\tProcessing {len(dialogue['utterances'])} utterances")
        all_responses = []
        all_capsules = []
        capsules_skipped = 0
        for utterance in dialogue["utterances"]:
            # add utterance to chat and use CFG analyzer to analyze
            print(f"\t\tAnalyze utterance: {utterance['turn_id']} (accumulated time: {time_analyzer})")
            chat.add_utterance(utterance["text"], utterance["speaker"])
            this_time_analyzer = time.time()
            analyzer.analyze_in_context(chat)
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
                    this_time_linker = time.time()
                    linker.link(capsule)
                    time_linker += time.time() - this_time_linker

                    # Add capsule to brain
                    print(f"\t\t\t\tAdding capsule to brain (accumulated time: {time_brain})")
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

                    # print(f"Utterance: {utterance}, Capsule: {json.dumps(capsule_json, indent=2)}")
                except:
                    capsules_skipped += 1
                    print(f"\t\t\tCapsule skipped. Total skipped: {capsules_skipped}"
                          f"\n{json.dumps(brain_response_to_json(capsule), indent=2)}")

            # Save responses
        f = open(scenario_filepath / "capsules.json", "w")
        json.dump(all_capsules, f)
        f = open(scenario_filepath / "responses.json", "w")
        json.dump(all_responses, f)


if __name__ == "__main__":
    main()
