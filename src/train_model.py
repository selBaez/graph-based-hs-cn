import argparse
import json
import os
import random

import evaluate
import nltk
import numpy as np
import torch
from rich import box
from rich.console import Console
from rich.table import Column, Table
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer

from src.train_utils.model import T5ForGoTGeneration
from src.train_utils.utils_data import load_data_std_aqua, DialoconanDatasetGoT, mk_dir, make_save_directory

console = Console(record=True)
os.environ["WANDB_PROJECT"] = "GoT_reproduction"


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


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak.
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    pred_ids = torch.argmax(logits[0], dim=-1)
    return pred_ids, labels


def T5Trainer(args):
    set_random_seeds(args)

    # make directories for output
    print('====Make directories====')
    save_dir = make_save_directory(args)
    mk_dir(args.output_dir)

    # Create tokenizer
    print(f'====Create tokenizer====')
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.add_special_tokens({'additional_special_tokens': ['<s>']})
    vocab = tokenizer.get_vocab()
    s_token_id = vocab["<s>"]
    datacollator = DataCollatorForSeq2Seq(tokenizer)

    # Load data as dataset
    print('====Load dataset====')
    train_problems, dev_problems, test_problems = load_data_std_aqua(args, console=console)
    train_set = DialoconanDatasetGoT(train_problems, "train", tokenizer, args.input_len, args.output_len, args)
    eval_set = DialoconanDatasetGoT(dev_problems, "dev", tokenizer, args.input_len, args.output_len, args)
    test_set = DialoconanDatasetGoT(test_problems, "test", tokenizer, args.input_len, args.output_len, args)

    # Load model
    print(f'====Load model: {args.model} ====')
    model = T5ForGoTGeneration.from_pretrained(args.model, s_token_id=s_token_id)
    model.resize_token_embeddings(len(tokenizer))
    print("model parameters: ", model.num_parameters())

    # rougel for cn generation
    metric = evaluate.load("rouge")

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
        pred_result = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(pred_result, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # evaluate at each epoch
    print('====Load training arguments====')
    training_args = Seq2SeqTrainingArguments(save_dir,
                                             do_train=True,
                                             do_eval=True,
                                             evaluation_strategy="steps",
                                             logging_strategy="steps",
                                             logging_steps=10,
                                             save_strategy="steps",
                                             eval_steps=1000,
                                             save_steps=1000,
                                             save_total_limit=2,
                                             learning_rate=args.lr,
                                             eval_accumulation_steps=args.eval_acc,
                                             per_device_train_batch_size=args.bs,
                                             per_device_eval_batch_size=args.eval_bs,
                                             weight_decay=args.weight_decay,
                                             num_train_epochs=args.epoch,
                                             metric_for_best_model="rougeL",
                                             predict_with_generate=args.use_generate,
                                             generation_max_length=args.output_len,
                                             load_best_model_at_end=True,
                                             report_to="wandb",
                                             bf16=args.bf16
                                             )

    print('====Load trainer====')
    trainer = Seq2SeqTrainer(model=model,
                             args=training_args,
                             train_dataset=train_set,
                             eval_dataset=eval_set,
                             data_collator=datacollator,
                             tokenizer=tokenizer,
                             compute_metrics=compute_metrics_rougel,
                             preprocess_logits_for_metrics=preprocess_logits_for_metrics if not args.use_generate else None
                             )

    # Train
    print('====Train====')
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(save_dir)

    # Evaluate
    print('====Evaluate====')
    metrics = trainer.evaluate(eval_dataset=test_set, max_length=args.output_len)
    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len)

    # generate the rationale for the eval set
    torch.cuda.empty_cache()
    predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len)
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            # preds = preds.argmax(axis=2)

        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        preds = [pred.strip() for pred in preds]
        output_data = {"preds": preds,
                       "labels": targets}
        if args.use_generate:
            output_prediction_file = os.path.join(save_dir, "predictions_ans_eval.json")
        else:  # with teacher forcing
            # make dir
            mk_dir(os.path.join(save_dir, "tf_pred"))
            output_prediction_file = os.path.join(save_dir, "tf_pred", "predictions_ans_eval.json")

        with open(output_prediction_file, "w") as writer:
            writer.write(json.dumps(output_data, indent=4))


def set_random_seeds(args):
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./../data')
    parser.add_argument('--dataset', type=str, default='DIALOCONAN')
    parser.add_argument('--got_root', type=str, default='got/')
    parser.add_argument('--output_dir', type=str, default='./../experiments')
    parser.add_argument('--model', type=str, default='declare-lab/flan-alpaca-base')  # TODO or large?
    parser.add_argument('--epoch', type=int, default=2)  # TODO change
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=4)  # TODO change
    parser.add_argument('--input_len', type=int, default=512) # TODO check
    parser.add_argument('--output_len', type=int, default=64) # TODO check
    parser.add_argument('--eval_bs', type=int, default=4) # TODO change
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--use_generate', action='store_true', help='only for baseline to improve inference speed')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--eval_strategy', type=str, default="steps", help='evaluation strategy',
                        choices=['steps', 'epoch'])
    parser.add_argument('--weight_decay', type=float, default=0.05, help='weight decay')
    parser.add_argument('--bf16', action='store_true', help='use bf16 dtype')
    args = parser.parse_args()

    print("args", args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))

    random.seed(args.seed)

    return args


def main():
    print(f"\n\n\nCUDA AVAILABLE? {torch.cuda.is_available()}\n\n\n")

    # training logger to log training progress
    training_logger = Table(Column("Epoch", justify="center"),
                            Column("Steps", justify="center"),
                            Column("Loss", justify="center"),
                            title="Training Status",
                            pad_edge=False,
                            box=box.ASCII)

    args = parse_args()

    T5Trainer(args)


if __name__ == '__main__':
    main()