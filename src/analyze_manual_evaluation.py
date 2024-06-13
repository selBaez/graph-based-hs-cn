import argparse
from pathlib import Path
from random import randint

import pandas as pd

from train_utils.utils_data import load_data

SAMPLE_SIZE = 0.25

def read_and_sample(experiments_path):
    prediction_files = sorted(f / "checkpoint-3500" / "predictions_ans_test.csv" for f in experiments_path.iterdir())

    experiments_data = []
    for i, p_file in enumerate(prediction_files):
        # Get predictions from each model
        predictions = pd.read_csv(p_file, index_col=0)
        predictions[f"{p_file.parents[1].name}"] = predictions[["pred"]]
        if i == 0:
            predictions = predictions[["ref", f"{p_file.parents[1].name}"]]
        else:
            predictions = predictions[[f"{p_file.parents[1].name}"]]
        experiments_data.append(predictions)

    # Create dataframe
    data = pd.concat(experiments_data, axis=1)

    # Sample
    to_annotate = data.sample(frac=SAMPLE_SIZE, random_state=42)

    return to_annotate


def rearrange_dialogues(to_annotate, problems_test):
    all_annotations = []
    all_keys = []
    for idx, row in to_annotate.iterrows():
        # Shuffle order
        row = row.sample(frac=1, random_state=42)

        # Reformat and add idx, id, and annotation columns
        new_format = pd.DataFrame(row).reset_index().rename(columns={'index': 'source', idx: "text"})
        new_format["og_idx"] = idx
        new_format["id"] = [randint(1, 10000) for k in new_format.index]
        new_format[["Relatedness", "Specificity", "Richness", "Coherence", "Grammatically"]] = "", "", "", "", ""
        new_format = new_format[["source", "og_idx", "id", "text",
                                 "Relatedness", "Specificity", "Richness", "Coherence", "Grammatically"]]

        # Separate secret parts from public parts
        key_for_id = new_format[["source", "og_idx", "id"]]
        annotators_df = new_format[["og_idx", "id", "text",
                                    "Relatedness", "Specificity", "Richness", "Coherence", "Grammatically"]]

        # Add HS and dialogue history
        context_df = pd.DataFrame(columns=annotators_df.columns)
        context_df.loc[0] = ["Dialogue history:", "", "", "", "", "", "", ""]
        context_df.loc[1] = [problems_test.iloc[[idx]]["dialogue_history"].values[0], "", "", "", "", "", "", ""]
        context_df.loc[2] = ["HS to address:", "", "", "", "", "", "", ""]
        context_df.loc[3] = [f'HS: {problems_test.iloc[[idx]]["hate_speech"].values[0]}', "", "", "", "", "", "", ""]

        # Add empty rows
        empty_between = pd.DataFrame([{}] * 1, columns=annotators_df.columns)
        empty_end = pd.DataFrame([{}] * 3, columns=annotators_df.columns)

        # Concat
        dialogue = pd.concat([context_df, empty_between, annotators_df, empty_end], ignore_index=True)

        all_annotations.append(dialogue)
        all_keys.append(key_for_id)

    return all_annotations, all_keys


def main(args):
    experiments_path = Path(f"./../experiments/{args.dataset}").resolve()
    annotations_path = Path(f"./../annotations/{args.dataset}").resolve()

    # read data
    to_annotate = read_and_sample(experiments_path)
    problems_test = pd.DataFrame(load_data(args, 'test'))

    # format data
    all_annotations, all_keys = rearrange_dialogues(to_annotate, problems_test)

    # Create final files
    print(f"Selected {len(all_annotations)} dialogues for annotation")
    public_data = pd.concat(all_annotations, axis=0)
    secret_data = pd.concat(all_keys, axis=0)

    public_data.to_csv(annotations_path / "annotation_template.tsv", index=False, sep='\t')
    secret_data.to_csv(annotations_path / "annotation_keys.tsv", index=False, sep='\t')

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')

    args = parser.parse_args()

    main(args)
