import os
import sys
import argparse
import json

import numpy as np
import torch

from torch.utils.data import DataLoader

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.spice.spice import Spice

from word2number import w2n

import re

def remove_there(text):
    text = re.sub(r'\b(there are|there is)\b', '', text).strip()
    text = re.sub(r'  ', ' ', text)
    return text

def remove_periods(text):
    return re.sub(r'\.', '', text)

def remove_articles(text):
    text = re.sub(r'\b(a|an|the)\b', '', text, flags=re.IGNORECASE).strip()
    text = re.sub(r'  ', ' ', text)
    return text

def replace_number(text):
    def replace(match):
        word = match.group(0)
        try:
            return str(w2n.word_to_num(word))
        except ValueError:
            return word
    pattern = r'\b[a-zA-z-]+\b'
    return re.sub(pattern, replace, text)

def process_text(text, question_type):
    text = text.lower()
    text = replace_number(text)
    text = remove_articles(text)
    text = remove_there(text)
    text = remove_periods(text)

    return text

def evals_json(data):

    score = []

    for ins in data:
        question_id=ins['question_id']
        question=ins['question']
        answer = ins['pred_answer']
        ref_answers=ins['ref_answers']
        scene_id=ins['scene_id']
        qt = qclass1(question)

        if process_text(answer, qt) in ref_answers:
            score.append(1)
        else:
            score.append(0)
            
    em = np.mean(score)*100
    return em

class EM:
    def __init__(self):
        pass
    
    def method(self):
        return "EM@1"    
        
    def compute_score(self, gts, res):
    
        scores = []
        
        for qid in gts.keys():
            if res[qid][0] in gts[qid]:
                scores.append(1)
            else:
                scores.append(0)
        score = np.mean(scores)
        
        return score, scores

def eval_pycoco(data, use_spice=False):
    score_list = ['EM@1', 'BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    score = {s:[] for s in score_list}

    scorers = [
        (EM(), "EM@1"),
        (Bleu(4), ["Bleu_1", "Bleu_2", "Blue_3", "Blue_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
    ]
    if use_spice:
        scorers.append((Spice(), "SPICE"))

    tokenizer = PTBTokenizer()

    # pycocoeval
    gts = {ins['question_id']:[{'caption':ans} for ans in ins['ref_answers']] for ins in data}
    res = {ins['question_id']:[{'caption':process_text(ins['pred_answer'], qclass1(ins['question']))}] for ins in data}
    gts = tokenizer.tokenize(gts)
    res = tokenizer.tokenize(res)

    # =================================================
    # Compute scores
    # =================================================
    rlt={}
    for scorer, method in scorers:
        print('computing %s score...'%(scorer.method()))
        score, scores = scorer.compute_score(gts, res)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                print("%s: %0.3f"%(m, sc*100))
                rlt[m]=sc*100
        else:
            print("%s: %0.3f"%(method, score*100))
            rlt[method]=score*100
    return rlt

QT=['Place', 'Number', 'Color', 'Object nature', 'Object', 'Other']
def qclass1(question):
    lques = question
    if 'Where' in lques:
        return 'Place'
    if 'How many' in lques:
        return 'Number'
    if 'What color' in lques or 'What is the color' in lques:
        return 'Color'
    if 'What shape' in lques or 'What type' in lques or 'What kind' in lques:
        return 'Object nature'
    if 'What is' in lques:
        return 'Object'
    return 'Other'

def main(args):

    # load prediction
    with open(args.pred, "r") as f:
        data = [json.loads(line) for line in f]
        
    # ALL
    score = evals_json(data)
    print(score)
    eval_pycoco(data, use_spice=args.use_spice)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="base path")
    parser.add_argument("--pred", type=str, help="path to prediction")
    parser.add_argument("--use_spice", action="store_true")
    args = parser.parse_args()
    
    main(args)
