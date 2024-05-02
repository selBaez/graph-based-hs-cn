# Imports
import json
import logging
import requests
from cltl.brain import logger as brain_logger
from cltl.brain.long_term_memory import LongTermMemory
from cltl.brain.utils.helper_functions import brain_response_to_json
from cltl.entity_linking.label_linker import LabelBasedLinker
from datetime import date, datetime
from pathlib import Path
from random import getrandbits

from cltl.triple_extraction import logger as chat_logger
from cltl.triple_extraction.api import Chat
from cltl.triple_extraction.cfg_analyzer import CFGAnalyzer
# from cltl.triple_extraction.spacy_analyzer import spacyAnalyzer
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
    analyzer = CFGAnalyzer()
    linker = LabelBasedLinker()

    # loop through dialogues
    for dialogue in logs:
        # Create folders
        scenario_filepath = Path(f'./../data/KDIALOCONAN_grounded_gold_{dialogue["dialogue_id"]}/')
        graph_filepath = scenario_filepath / Path('graph/')
        graph_filepath.mkdir(parents=True, exist_ok=True)

        # Initialize brain, Chat,
        brain = LongTermMemory(address="http://localhost:7200/repositories/sandbox",
                               # Location to save accumulated graph
                               log_dir=graph_filepath,  # Location to save step-wise graphs
                               clear_all=True)  # To start from an empty brain
        chat = Chat("CN", "HS")

        # Create context
        context_capsule = create_context_capsule()
        brain.capsule_context(context_capsule)

        # convert to eKG
        all_responses = []
        all_capsules = []
        capsules_skipped = 0
        for utterance in dialogue["utterances"]:
            # add utterance to chat and use CFG analyzer to analyze
            chat.add_utterance(utterance["text"], utterance["speaker"])
            analyzer.analyze(chat.last_utterance)
            capsules = utterance_to_capsules(chat.last_utterance)

            # add statement capsules to brain
            for capsule in capsules:
                try:
                    # Ugly fix of capsule
                    capsule['author'] = {"label": utterance["speaker"], "type": ["person"]}
                    capsule['timestamp'] = datetime.now()

                    # Link to specific instances
                    linker.link(capsule)

                    # Add capsule to brain
                    print("\tAdding capsule to brain")
                    response = brain.capsule_statement(capsule, reason_types=True,
                                                       create_label=True)  # Fix problem with overlaps?

                    # Keep track of responses
                    capsule['rdf_file'] = str(response['rdf_log_path'].stem) + '.trig'
                    capsule_json = brain_response_to_json(capsule)
                    all_capsules.append(capsule_json)
                    response_json = brain_response_to_json(response)
                    all_responses.append(response_json)

                    # print(f"Utterance: {utterance}, Capsule: {json.dumps(capsule_json, indent=2)}")
                except:
                    capsules_skipped += 1
                    print(
                        f"\tCapsule skipped. Total skipped: {capsules_skipped}\n{json.dumps(brain_response_to_json(capsule), indent=2)}")

            # Save responses
        f = open(scenario_filepath / "capsules.json", "w")
        json.dump(all_capsules, f)
        f = open(scenario_filepath / "responses.json", "w")
        json.dump(all_responses, f)


if __name__ == "__main__":
    main()
