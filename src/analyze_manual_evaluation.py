import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from prepare_manual_evaluation import DIMENSIONS

ANNOTATORS = [
    # 779698106,  # i
    1235414710,  # i2 (30)
    # 742083769, # h
    410231477,  # f (30)
    1106223098,  # s (30)
    # 182740748,  # l (5)
    73519088,  # w (30)
    # 1664249700,  # j (11)
    # 1718239219,  # b (20)
    # 1975017435,  # m
    # 1014977765  # a
]

N_ANNOTATORS = len(ANNOTATORS)

MODELS = [
    'declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_2024-06-01-04-17',  # got
    'declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_2024-06-06-18-34',  # ekg
    'declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_2024-06-08-23-44',  # txt
    'declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_2024-06-09-21-36',  # tnc
    'declare-lab-flan-alpaca-base_lr5e-05_bs32_op256_ep50_2024-06-11-00-09',  # got-nc
    'ref'
]


def read_data(key):
    per_dim = {"Relatedness": defaultdict(list), "Specificity": defaultdict(list), "Richness": defaultdict(list),
               "Coherence": defaultdict(list), "Grammatically": defaultdict(list), "Effectiveness": defaultdict(list)}

    min_common_annotations = 27 * 6  # dialogues x dimensions
    for sheet_id in ANNOTATORS:
        annotations = pd.read_csv(f'https://docs.google.com/spreadsheets/d/'
                                  f'1_VviKQ__XZ1mwtVq6XrSOMuZV0ut449xFekUjo0fnSY/'
                                  f'export?gid={sheet_id}&format=csv')

        # format data
        annotations = annotations[annotations['Relatedness'].notna()]
        annotations = annotations[annotations['id'].notna()]
        annotations = annotations[["id", "og_idx", "text",
                                   "Relatedness", "Specificity", "Richness", "Coherence", "Grammatically",
                                   "Effectiveness"]]

        # map source model
        annotations['og_idx'] = annotations['og_idx'].astype('int64')
        key['og_idx'] = key['og_idx'].astype('int64')
        merged_df = pd.merge(annotations, key, left_on=['id', 'og_idx'], right_on=['id', 'og_idx'], how='left')

        if len(merged_df) > 0:
            min_common_annotations = min(min_common_annotations, len(merged_df))

            # put in dictionary
            for dim in DIMENSIONS:
                per_dim[dim]["all"].append(merged_df[dim].tolist())

                # group by model
                for model, model_ann in merged_df.groupby(by=["source"]):
                    per_dim[dim][model[0]].append(model_ann[dim].tolist())

    return per_dim, min_common_annotations


def main(args):
    annotations_path = Path(f"./../annotations/{args.dataset}").resolve()

    # read data
    key = pd.read_csv(f"{annotations_path}/annotation_keys.tsv", sep='\t')
    all_annotations, min_common_annotations = read_data(key)
    min_common_dialogues = min_common_annotations // 6

    per_dim = {"Relatedness": defaultdict(dict), "Specificity": defaultdict(dict), "Richness": defaultdict(dict),
               "Coherence": defaultdict(dict), "Grammatically": defaultdict(dict), "Effectiveness": defaultdict(dict),
               "All": defaultdict(dict)}
    iaa = []
    for dim, info in all_annotations.items():
        dim_annotations = []
        for ann_id, ann in enumerate(info["all"]):
            if np.nan not in ann[:min_common_annotations]:
                dim_annotations.append(ann[:min_common_annotations])
            else:
                print(f"\tCorrupt annotation for dimension:{dim} from annotator {ann_id}\n"
                      f"\t{ann[:min_common_annotations]}\n\n")

        # Export as csv
        dimension_file = pd.DataFrame(dim_annotations).transpose()
        dimension_file.to_csv(annotations_path / f'annotations_{dim.lower()}.csv', header=False, index=False)

        # Calculate Cohen's kappa for all pairs of annotators
        kappas = []
        for i in range(len(dim_annotations)):
            for j in range(i + 1, len(dim_annotations)):
                kappa = cohen_kappa_score(dim_annotations[i], dim_annotations[j])
                kappas.append(kappa)
        per_dim[dim]["IAA"] = np.mean(kappas)
        iaa.append(np.mean(kappas))

        for model in MODELS:
            model_dim_annotations = []
            for annotator in info[model]:
                if np.nan not in annotator[:min_common_dialogues]:
                    fixed_annotations = [int(ann) for ann in annotator[:min_common_dialogues]]
                    model_dim_annotations.append(fixed_annotations)

            model_dim_avg = [np.mean(annotator) for annotator in model_dim_annotations]
            per_dim[dim][model]["per_annotator"] = model_dim_avg
            per_dim[dim][model]["total_average"] = np.mean(model_dim_avg)

            print(f"Dialogues: {min_common_dialogues}, Num annotators: {len(dim_annotations)}, "
                  f"Dimension: {dim}, "
                  f"Average value: {per_dim[dim][model]['total_average']:.2f}, "
                  f"Average Cohen's kappa: {per_dim[dim]['IAA']:.2f}\n\n")

    per_dim["All"]["IAA"] = np.mean(iaa)

    with open(annotations_path / 'interannotator_agreement.json', 'w', encoding='utf-8') as f:
        json.dump(per_dim, f, ensure_ascii=False, indent=4)

    print("DONE")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')

    args = parser.parse_args()

    main(args)
