import json
import os
import pickle
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import Dataset

from src.train_utils.utils_prompt import build_train_pair_dialoconan


def make_save_directory(args):
    model_name = args.model.replace("/", "-")
    gpu_count = torch.cuda.device_count()
    save_dir = f"{args.output_dir}/{model_name}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}_useG{args.use_generate}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
    mk_dir(save_dir)
    print(save_dir)

    return save_dir


def mk_dir(dir):
    dir = Path(dir)
    dir.mkdir(parents=True, exist_ok=True)


def load_data(args, split):
    datapath = Path(args.data_root) / Path(args.dataset) / f"{split}.json"
    with open(datapath, 'r', encoding='utf-8') as file:
        dialogues = json.load(file)

    return dialogues


def load_data_std_aqua(args, console):
    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    problems_train = load_data(args, 'train')
    problems_dev = load_data(args, 'dev')
    problems_test = load_data(args, 'test')

    console.log(f"number of train problems: {len(problems_train)}\n")
    console.log(f"number of val problems: {len(problems_dev)}\n")
    console.log(f"number of test problems: {len(problems_test)}\n")

    return problems_train, problems_dev, problems_test


class DialoconanDatasetGoT(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(self, problems, split, tokenizer, source_len, target_len, args):
        self.tokenizer = tokenizer
        self.data = problems  # {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []

        with open(os.path.join(args.data_root, args.dataset, args.got_root, split, 'mc_input_text.pkl'), 'rb') as f:
            self.got_input_text_list = pickle.load(f)
        with open(os.path.join(args.data_root, args.dataset, args.got_root, split, 'mc_adj_matrix.pkl'), 'rb') as f:
            self.got_adj_matrix_list = pickle.load(f)

        for qid, prob in enumerate(self.data):
            prompt, target = build_train_pair_dialoconan(prob)
            self.target_text.append(target)
            self.source_text.append(prompt)

    def __len__(self):
        return len(self.target_text)

    def __getitem__(self, index):
        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])

        # cleaning data to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        got_input_text = self.got_input_text_list[index]
        got_adj_matrix = self.got_adj_matrix_list[index]
        got_adj_matrix = torch.tensor(got_adj_matrix)

        source = self.tokenizer.batch_encode_plus([source_text],
                                                  max_length=self.source_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors="pt",
                                                  )
        target = self.tokenizer.batch_encode_plus([target_text],
                                                  max_length=self.summ_len,
                                                  pad_to_max_length=True,
                                                  truncation=True,
                                                  padding="max_length",
                                                  return_tensors="pt",
                                                  )

        encoded_got_input_text = self.tokenizer.batch_encode_plus(got_input_text,
                                                                  max_length=self.source_len,
                                                                  pad_to_max_length=True,
                                                                  truncation=True,
                                                                  padding="max_length",
                                                                  return_tensors="pt",
                                                                  )

        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        encoded_got_input_text_ids = encoded_got_input_text["input_ids"].squeeze()
        encoded_got_input_text_mask = encoded_got_input_text["attention_mask"].squeeze()

        return {"input_ids": source_ids,
                "attention_mask": source_mask,
                "labels": target_ids,
                "got_adj_matrix": got_adj_matrix,
                "got_input_ids": encoded_got_input_text_ids,
                "got_mask": encoded_got_input_text_mask,
                }
