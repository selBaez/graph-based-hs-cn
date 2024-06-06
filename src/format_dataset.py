import pandas as pd

from sklearn.model_selection import train_test_split


def format_dialogue_history(dialogue):
    # Remove last HS and CN as they will be treated separatedly
    dialogue = dialogue[:-2]

    # Format each utterance and join them
    utt = [f"{u['type']}: {u['text']}" for idx, u in dialogue.iterrows()]
    dialogue = "\n".join(utt)

    return dialogue


def structure_dataset(per_dialogue_df):
    logs = []
    for dialogue_id, dialogue in per_dialogue_df:
        item = {"dialogue_id": dialogue_id[0],
                "target": dialogue["TARGET"].unique()[0],
                "hate_speech": dialogue["text"].values[-2],
                "counter_narrative": dialogue["text"].values[-1],
                "dialogue_history": format_dialogue_history(dialogue)}

        logs.append(item)

    return logs


def main():
    # Read data
    dialogue_df = pd.read_csv("./../data/original_datasets/DIALOCONAN.csv").sort_values(by=["dialogue_id", "turn_id"])

    # Group by dialogue id
    per_dialogue_df = dialogue_df.groupby(["dialogue_id"])

    # Create dataset as in AQuA: {dialogue_id}
    logs = structure_dataset(per_dialogue_df)

    # Split into train/dev/test
    [train, test] = train_test_split(logs, test_size=0.10, random_state=42)
    [train, dev] = train_test_split(train, test_size=0.1111, random_state=42)
    print(f'Train size: {len(train)}, Dev size: {len(dev)}, Test size: {len(test)}')

    # Save to file
    # dataset_directory = Path("./../data/DIALOCONAN")
    # dataset_directory.mkdir(parents=True, exist_ok=True)
    #
    # save_json(logs, filename= dataset_directory / "full.json")
    # save_json(train, filename= dataset_directory / "train.json")
    # save_json(dev, filename= dataset_directory / "dev.json")
    # save_json(test, filename= dataset_directory / "test.json")

    # ######################## Split train into 5 ########################
    # with open("./../data/DIALOCONAN/train.json") as f:
    #     full = json.load(f)
    #
    # parts = []
    # bgn = 0
    # end = 500
    # for i in range(5):
    #     part = full[bgn:end]
    #     parts.append(part)
    #
    #     bgn += 500
    #     end += 500
    #
    #     with open(f"./../data/DIALOCONAN/train_{i}.json", "w") as f:
    #         json.dump(part, f)

    print("DONE")


if __name__ == '__main__':
    main()
