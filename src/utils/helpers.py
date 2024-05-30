import json


def save_json(dataset, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)
    print(f"Wrote dataset to file: {filename}")


def merge_data(dialogue_df, knowledge_df, format='DSTC', save_as_json=True):
    dataset = []

    if format == "DSTC":
        dataset = merge_data_as_dstc(dialogue_df, knowledge_df)
        if save_as_json:
            save_json(dataset, filename="./../data/KDIALOCONAN_gold.json")

    elif format == "DSTC_filtered":
        dataset = merge_data_as_dstc(dialogue_df, knowledge_df)
        dataset = filter_dstc_data(dataset)
        if save_as_json:
            save_json(dataset, filename="./../data/KDIALOCONAN_grounded_gold.json")

    if format == "AQuA":
        dataset = merge_data_as_aqua(dialogue_df, knowledge_df)
        if save_as_json:
            save_json(dataset, filename="./../data/KDIALOCONAN_aqua_gold.json")

    else:
        print(f"Format not supported")

    return dataset


def merge_data_as_dstc(dialogue_df, knowledge_df):
    # Create dataset as in DSTC: {dialogue_id}
    logs = []

    # Looping helpers
    prev_row = None
    dialogue_id = 0
    utterances = []
    grounded_dialogue = False

    for i, row in dialogue_df.iterrows():
        # determine if we are in a new dialogue. if so, add the last one to the logs
        if row["dialogue_id"] != dialogue_id:
            dialogue = {"dialogue_id": prev_row["dialogue_id"],
                        "hs_target": prev_row["TARGET"],
                        "dialogue_source": prev_row["source"],
                        "knowledge_grounded": grounded_dialogue,
                        "utterances": utterances}
            logs.append(dialogue)
            utterances = []
            grounded_dialogue = False

        # gather labels
        match = knowledge_df[["knowledge_sentence", "counter_narrative"]][knowledge_df["hate_speech"] == row["text"]]
        grounded_dialogue = len(match) > 0 or grounded_dialogue

        knowledge_items = {"knowledge_sentence": match["knowledge_sentence"].drop_duplicates().to_list(),
                           "counter_narrative": match["counter_narrative"].drop_duplicates().to_list()}

        # Create utterance object and add it to the utterances list
        utterance = {"turn_id": row["turn_id"], "speaker": row["type"], "text": row["text"],
                     "knowledge_grounded": len(match) > 0, "knowledge": knowledge_items}
        utterances.append(utterance)

        # move variables along the iteration
        dialogue_id = row["dialogue_id"]
        prev_row = row

        # add last dialogue
        if i == len(dialogue_df) - 1:
            dialogue = {"dialogue_id": prev_row["dialogue_id"],
                        "hs_target": prev_row["TARGET"],
                        "dialogue_source": prev_row["source"],
                        "knowledge_grounded": grounded_dialogue,
                        "utterances": utterances}
            logs.append(dialogue)

    return logs


def filter_dstc_data(logs):
    logs_grounded = [dialogue for dialogue in logs if dialogue["knowledge_grounded"]]

    return logs_grounded


def merge_data_as_aqua(dialogue_df, knowledge_df):
    pass
