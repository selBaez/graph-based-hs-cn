import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer


class SentenceSimilarityScore():
    def __init__(self, use_gpu=torch.cuda.is_available()):
        self.use_gpu = use_gpu
        self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)

    def similariry_score(self, str1, str2):
        # compute embedding for both lists
        embedding_1 = self.model.encode(str1, convert_to_tensor=True)
        embedding_2 = self.model.encode(str2, convert_to_tensor=True)
        score = util.pytorch_cos_sim(embedding_1, embedding_2).item()
        return score

    def score(self, results, data):
        scores = []
        for qid, output in results.items():
            prediction = output
            target = data[qid]
            target = target.strip()

            score = self.similariry_score(target, prediction)
            scores.append(score)

        return scores


class Bleurt():
    def __init__(self, model_path, max_length, batch_size, use_gpu=torch.cuda.is_available()):
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()

    def score(self, hypo, refs):
        device = self.device
        hypo_all = []
        refs_all = []
        for qid, output in hypo.items():
            hypo_all.append(output)
            refs_all.append(refs[qid])

        scores_all = []
        for i in tqdm(range(0, len(hypo_all), self.batch_size)):
            with torch.no_grad():
                if (len(refs_all[i:i + self.batch_size]) == 1):
                    continue

                inputs = self.tokenizer(refs_all[i:i + self.batch_size], hypo_all[i:i + self.batch_size],
                                        return_tensors='pt', truncation=True, padding=True, max_length=self.max_length)

                if (self.use_gpu):
                    scores = list(self.model(input_ids=inputs['input_ids'].to(device),
                                             attention_mask=inputs['attention_mask'].to(device),
                                             token_type_ids=inputs['token_type_ids'].to(device))[0].
                                  squeeze().cpu().numpy())
                else:
                    scores = list(self.model(input_ids=inputs['input_ids'],
                                             attention_mask=inputs['attention_mask'],
                                             token_type_ids=inputs['token_type_ids'])[0].squeeze().cpu().numpy())

                scores_all += scores

        return scores_all


class CounterspeechScore():
    def __init__(self, model_path, max_length, batch_size, use_gpu=torch.cuda.is_available()):
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.device = torch.device("cpu")
        if self.use_gpu:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
        self.model.eval()

    def scoring(self, hypo):
        device = self.device
        hypo = [el for qid, el in hypo.items()]

        scores_all = []
        for i in tqdm(range(0, len(hypo), self.batch_size)):
            with torch.no_grad():

                inputs = self.tokenizer(hypo[i:i + self.batch_size], return_tensors='pt', truncation=True, padding=True,
                                        max_length=self.max_length)

                if (self.use_gpu):
                    scores = self.model(input_ids=inputs['input_ids'].to(device),
                                        attention_mask=inputs['attention_mask'].to(device))[0].squeeze()
                else:
                    scores = self.model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])[
                        0].squeeze()
                scores = torch.softmax(scores.T, dim=0).T.cpu().numpy()
                scores_all += list(scores[:, 1])

        return scores_all
