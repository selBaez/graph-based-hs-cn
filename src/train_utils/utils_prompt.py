'''
Adapted from https://github.com/lupantech/ScienceQA and https://github.com/amazon-science/mm-cot
'''

from dataclasses import dataclass
from typing import List, Optional

import nltk


def build_train_pair_dialoconan(problems, exclude_context=False):
    examples = []
    # create the prompt input
    if exclude_context:
        text_example = f"Hate Speech:\n{problems['hate_speech']}\n\n" \
                       f"Counter-narrative:\n"
    else:
        text_example = f"Hate Speech:\n{problems['hate_speech']}\n\n" \
                       f"Dialogue History:\n{problems['dialogue_history']}\n\n" \
                       f"Counter-narrative:\n"

    target = problems['counter_narrative']

    examples.append(text_example)
    prompt_input = '\n\n'.join(examples)

    return prompt_input, target


def postprocess_text(preds, labels):
    processed_preds = []
    for pred in preds:
        pred = pred.strip()
        try:
            # use nltk to split the text into sentences
            processed_pred = "\n".join(nltk.sent_tokenize(pred))
        except IndexError:
            # if the text is too long, it may cause an IndexError
            print(f"IndexError occurred with text: {pred}")
            processed_pred = pred
        processed_preds.append(processed_pred)

    processed_labels = []
    for label in labels:
        label = label.strip()
        try:
            # use nltk to split the text into sentences
            processed_label = "\n".join(nltk.sent_tokenize(label))
        except IndexError:
            print(f"IndexError occurred with text: {label}")
            processed_label = label
        processed_labels.append(processed_label)

    return processed_preds, processed_labels


@dataclass(frozen=True)
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[List[int]]
    attention_mask: Optional[List[List[int]]]
    token_type_ids: Optional[List[List[int]]]
    le_input_ids: List[List[int]]
    le_attention_mask: Optional[List[List[int]]]
    le_token_type_ids: Optional[List[List[int]]]
    label: Optional[int]
