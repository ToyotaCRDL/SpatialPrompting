import os
import sys
import argparse
import json
import collections
import re
import numpy as np

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
    
def extract_enhanced(text):
    match = re.search(r'\*\*(.+?)\*\*', text)
    text = match.group(1) if match else text
    return text
    
def process_text(text):
    text = text.lower()
    text = remove_articles(text)
    text = remove_periods(text)
    text = extract_enhanced(text)
    
    return text

def evals_json(preds, ref_answers):

    score = []
    score_qt = {qt:[] for qt in QT}
    
    for p in preds:
        scene_id = p['scene_id']
        question_id = p['question_id']
        situation = p['situation']
        question = p['question']
        qt = question_type(question)
        answer = p['pred_answer']
        ref_answer = ref_answers[question_id]
        
        if process_text(answer) == ref_answer:
            score.append(1)
            score_qt[qt].append(1)
        else:
            score.append(0)
            score_qt[qt].append(0)
            
    em = np.mean(score)*100
    em_qt = {qt:np.mean(score_qt[qt]) for qt in QT}
    
    return em, em_qt
    
QT=['What', 'Is', 'How', 'Can', 'Which', 'Others']
          
def question_type(question):

    if question.lower().startswith('what'):
        return 'What'
    if question.lower().startswith('is'):
        return 'Is'
    if question.lower().startswith('how'):
        return 'How'
    if question.lower().startswith('can'):
        return 'Can'
    if question.lower().startswith('which'):
        return 'Which'
    return 'Others'
    
def main(args):
    
    sqa3d_path = os.path.join(args.base_path, "data", "SQA3D")
    
    # load prediction
    with open(args.pred, "r") as f:
        preds = [json.loads(line) for line in f]
    
    SPLITS=['test']
    annotation = {split:json.load(open(os.path.join(sqa3d_path, "sqa_task", "balanced", f"v1_balanced_sqa_annotations_{split}_scannetv2.json"))) for split in SPLITS}

    ref_answers = {anno['question_id']: anno['answers'][0]['answer'] for anno in annotation['test']['annotations']}

    score, score_qt = evals_json(preds, ref_answers)
    print(score)
    print(score_qt)
    
    print()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", type=str, help="base path")
    parser.add_argument("--pred", type=str, help="path to prediction")
    args = parser.parse_args()
    
    main(args)
    
