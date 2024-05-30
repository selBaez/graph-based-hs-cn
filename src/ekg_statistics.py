import json
import logging
import pandas as pd
from pathlib import Path
from rdflib import ConjunctiveGraph
from rdflib.extras.external_graph_libs import rdflib_to_networkx_multidigraph
from tqdm import tqdm

from cltl.dialogue_evaluation import logger as evaluation_logger
from cltl.dialogue_evaluation.graph_evaluation import GraphEvaluator

evaluation_logger.setLevel(logging.ERROR)

ANALYZERS = ["albert", "OIE", "spacy"]


def main():
    # retrieve dialogue ids
    with open("./../data/KDIALOCONAN_grounded_gold.json", 'r', encoding='utf-8') as f:
        logs = json.load(f)
        dialogue_ids = [dialogue['dialogue_id'] for dialogue in logs]
    print(f"There are: {len(dialogue_ids)} on this dataset")

    # create pandas df # multiindex: dialogue analyzer columns: metrics
    dfs = {}

    # create evaluator
    graph_evaluator = GraphEvaluator()

    # for each analyzer
    for analyzer in ANALYZERS:
        print(f"Process graphs from analyzer: {analyzer}")

        # create temp dataframe
        tmp_df = pd.DataFrame(index=dialogue_ids)

        # for each dialogue
        for dialogue_id in tqdm(dialogue_ids):
            print(f"\tDialogue id: {dialogue_id}")

            # create paths
            scenario_filepath = Path(f"./../data/KDIALOCONAN_grounded_gold({analyzer})/{dialogue_id}/graph/")
            rdf_folder = sorted([path for path in scenario_filepath.iterdir() if path.is_dir()])[-1]
            rdf_files = sorted([path for path in rdf_folder.glob('*.trig')])

            # create RDFlib to aggregate dialogue
            print(f"\tClear brain")
            brain_as_graph = ConjunctiveGraph()

            # for each graph file
            for file in rdf_files:
                # read file into RDFlib
                brain_as_graph.parse(file, format='trig')

            # calculate metric
            brain_as_netx = rdflib_to_networkx_multidigraph(brain_as_graph)
            tmp_df = graph_evaluator._calculate_metrics(brain_as_graph, brain_as_netx, tmp_df, dialogue_id)

            dfs[analyzer] = tmp_df

    # concat df from diff analyzers
    full_df = pd.concat(dfs, axis=1)
    full_df.sort_index(axis=1, level=[1], ascending=[True], inplace=True)
    full_df.to_csv("./../data/KDIALOCONAN_grounded_gold(graph_statistics).csv")
    print("done")


if __name__ == "__main__":
    main()
