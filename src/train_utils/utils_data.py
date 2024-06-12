import json
from datetime import datetime
from pathlib import Path

import torch


def make_save_directory(args):
    model_name = args.model.replace("/", "-")
    gpu_count = torch.cuda.device_count()
    save_dir = f"{args.output_dir}/{model_name}_lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}_{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
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


def load_data_std_dialoconan(args, console):
    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")

    problems_train = load_data(args, 'train')
    problems_dev = load_data(args, 'dev')
    problems_test = load_data(args, 'test')  # TODO change for real

    console.log(f"number of train problems: {len(problems_train)}\n")
    console.log(f"number of val problems: {len(problems_dev)}\n")
    console.log(f"number of test problems: {len(problems_test)}\n")

    return problems_train, problems_dev, problems_test
