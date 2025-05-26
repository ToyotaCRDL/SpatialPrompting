import os
import sys
import argparse
import json
import collections

import numpy as np
import torch

from spatial_feature import SpatialFeature
from spatialqa import SpatialQA

from easydict import EasyDict

def predict(gold_data, sefile, outputfile, common_answer, args):

    with open(outputfile, "a+") as f:
        f.seek(0)
        line_count = sum(1 for _ in f)
        print("line:", line_count)
        
        print(f"#data = {len(gold_data)}")
        current_scene_id = ""

        for i, ins in enumerate(gold_data):
        
            if i < line_count:
                continue
        
            question_id=ins['question_id']
            question=ins['question']
            ref_answers=ins['answers']
            scene_id=ins['scene_id']

            # make prediction
            if scene_id != current_scene_id:
                print(f"load spatial features of {scene_id}")
                
                se = SpatialFeature.load(
                    os.path.join(
                        CONF.PATH.SCANNET_SCANS, 
                        scene_id,
                        sefile
                        ),
                    device="cuda:0"
                    )
                current_scene_id = scene_id

                sqa = SpatialQA(args.llm, se, args)
            
            # Few-shot Prompt
            if not args.zeroshot:
                text = "Note that the answer for the question is as short as possible such as:\n"
            
                for qt in QT:
                    text += f"If question is start with {', '.join(map(str, QPhrase[qt]))}\n"
                    text += "Example of answers: "
                    for ans in common_answer[qt][:20]:
                        text += f"{ans[0]}"
                        text += ", "
                    text += "\n\n"
            
                sqa.messages.append({
                    "role": "user",
                    "content": [ text ]
                })
            
            if not args.zeroshot:
                answer = sqa(question)
            else:
                answer = sqa(question + " The answer shourld be a phrase or a single word.")
            
            print(f"({i}/{len(gold_data)}):{question_id}\nQ:{question}\nA:{answer}")
            
            sqa_result = {
                "question_id": question_id,
                "question": question,
                "pred_answer": answer,
                "ref_answers": ref_answers,
                "scene_id": scene_id
            }
            
            json.dump(sqa_result, f)
            f.write("\n")

QT=['Place', 'Number', 'Color', 'Object nature', 'Object', 'Other']
QPhrase = {
    "Place": ["Where"],
    "Number": ["How many"],
    "Color": ["What color", "What is the color"],
    "Object nature": ["What shape", "What type", "What kind"],
    "Object": ["What is"],
    "Other": ["others"],
}
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

    SPLITS=['train', 'val']

    ds={split:json.load(open(os.path.join(CONF.PATH.SCANQA, f'ScanQA_v1.0_{split}.json'))) for split in SPLITS}
    
    answer_example = {}
    for qt in QT:
        answer_example[qt] = []
    
    for ins in ds['train']:
        qt = qclass1(ins["question"])
        answer_example[qt].append(ins["answers"])

    num_answer = {}    
    common_answer = {}
    for qt in QT:
        answer_example[qt] = sum(answer_example[qt], [])
        answer_example[qt] = collections.Counter(sorted(answer_example[qt]))
        num_answer[qt] = len(answer_example[qt])
        common_answer[qt] = answer_example[qt].most_common()

    # prediction
    sefile = f"spatial_features_{args.model}_no_merge.npz"
    
    if not args.zeroshot:
        outputfile = f"results/predict_scanqa_fewshot_{args.llm}_{args.model}_img{args.image_num}"
    else:
        outputfile = f"results/predict_scanqa_zeroshot_{args.llm}_{args.model}_img{args.image_num}" 

    if args.nomerge:
        outputfile += "_nomerge"
    else:
        outputfile += f"_dist{args.merge_dist}_sim{args.merge_sim}"
    
    if args.nopose:
        outputfile += "_nopose"

    outputfile += ".jsonl"

    preds = predict(ds['val'], sefile, outputfile, common_answer, args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", help="base path")
    parser.add_argument("--llm", default="gpt-4o-2024-11-20")
    parser.add_argument("--model", default="vitl336")
    parser.add_argument("--image_num", type=int, default=30)
    parser.add_argument("--merge_dist", type=float, default=1.0)
    parser.add_argument("--merge_sim", type=float, default=0.8)
    parser.add_argument("--nomerge", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nopose", action="store_true")
    parser.add_argument("--zeroshot", action="store_true")
    args = parser.parse_args()

    # reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    CONF = EasyDict()

    # path
    CONF.PATH = EasyDict()
    CONF.PATH.BASE = args.base_path

    CONF.PATH.DATA = os.path.join(CONF.PATH.BASE, "data")
    CONF.PATH.SCANNET = os.path.join(CONF.PATH.DATA, "ScanNet")

    CONF.PATH.SCANNET_SCANS = os.path.join(CONF.PATH.SCANNET, "scans")
    CONF.PATH.SCANNET_META = os.path.join(CONF.PATH.SCANNET, "meta_data")
    CONF.PATH.SCANNET_DATA = os.path.join(CONF.PATH.SCANNET, "scannet_data")

    CONF.PATH.SCANQA = os.path.join(CONF.PATH.DATA, "ScanQA", "qa")
    
    main(args)
