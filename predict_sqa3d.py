import os
import sys
import argparse
import json
import collections

import numpy as np
import torch

from spatial_feature import SpatialFeature
from spatialqa import SpatialQA

def predict(questions, sefile, outputfile, common_answer, args):

    scene_data = {}
    for q in questions:
        scene_id = q["scene_id"]
        if scene_id not in scene_data.keys():
            scene_data[scene_id] = []
        scene_data[scene_id].append(q)

    with open(outputfile, "a+") as f:
        f.seek(0)
        line_count = sum(1 for _ in f)
        print("line:", line_count)
        
        print(f"#data = {len(questions)}")
        i = 0

        for scene_id in scene_data.keys():
            
            print(f"load spatial features of {scene_id}")
            scannet_path = os.path.join(args.base_path, "data", "ScanNet", "scans")
            se = SpatialFeature.load(os.path.join(
                scannet_path, 
                scene_id,
                sefile), device="cuda:0")

            obj = None

            sqa = SpatialQA(args.llm, se, args, obj=obj)
                
            
            # Few-shot Prompt
            if not args.zeroshot:
                text = "Note that the answer for the question based on the situation is as short as possible such as:\n\n"
            
                for qt in QT:
                    text += f"If question is start with {qt}\n"
                    text += "Example of answers: "
                    for ans in common_answer[qt][:20]:
                        text += f"{ans[0]}"
                        text += ", "
                    text += "\n\n"
                
                sqa.messages.append({
                    "role": "user",
                    "content": [text]
                })
                
            for q in scene_data[scene_id]:
            
                if i < line_count:
                    i+= 1
                    continue
            
                question_id=q['question_id']
                situation=q['situation']
                question=q['question']
                scene_id=q['scene_id']

                if not args.zeroshot:
                    answer = sqa(situation + question)
                else:
                    answer = sqa(situation + question + "The answer should be a phrase or a single word.")
                
                print(f"({i}/{len(questions)}):{question_id}\nS:{situation}\nQ:{question}\nA:{answer}")
                
                sqa_result = {
                    "question_id": question_id,
                    "situation": situation,
                    "question": question,
                    "pred_answer": answer,
                    "scene_id": scene_id
                }
                i += 1
                
                json.dump(sqa_result, f)
                f.write("\n")
                
      
QT=['What', 'Is', 'How', 'Can', 'Which', 'Others']
          
def question_type(question):

    if question.startswith('What'):
        return 'What'
    if question.startswith('Is'):
        return 'Is'
    if question.startswith('How'):
        return 'How'
    if question.startswith('Can'):
        return 'Can'
    if question.startswith('Which'):
        return 'Which'
    return 'Others'

def main(args):

    SPLITS=['train', 'test']
    sqa3d_path = os.path.join(args.base_path, "data", "SQA3D")
    ds = {split:json.load(open(os.path.join(sqa3d_path, "sqa_task", "balanced", f"v1_balanced_questions_{split}_scannetv2.json"))) for split in SPLITS}
    annotation = {split:json.load(open(os.path.join(sqa3d_path, "sqa_task", "balanced", f"v1_balanced_sqa_annotations_{split}_scannetv2.json"))) for split in SPLITS}
    
    # answer
    answer_example = {}
    for qt in QT:
        answer_example[qt] = []
    
    for i, question in enumerate(ds['train']['questions']):
        qt = question_type(question['question'])
        answer = annotation['train']['annotations'][i]['answers'][0]['answer']
        answer_example[qt].append(answer)
    
    num_answer = {}
    common_answer = {}
    for qt in QT:
        answer_example[qt] = collections.Counter(sorted(answer_example[qt]))
        num_answer[qt] = len(answer_example[qt])
        common_answer[qt] = answer_example[qt].most_common()
    
    # prediction
    sefile = f"spatial_features_{args.model}_no_merge.npz"

    # fewshot or zeroshot
    if not args.zeroshot:
        outputfile = "results/predict_sqa3d_fewshot"
    else:
        outputfile = "results/predict_sqa3d_zeroshot"

    outputfile = outputfile + f"_{args.llm}_{args.model}_img{args.image_num}"

    if args.nomerge:
        outputfile += "_nomerge"
    else:
        outputfile += f"_dist{args.merge_dist}_sim{args.merge_sim}"

    if args.nopose:
        outputfile += "_nopose"

    outputfile = outputfile + ".jsonl"

    preds = predict(ds['test']["questions"], sefile, outputfile, common_answer, args)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_path", help="base path")
    parser.add_argument("--llm", default="gpt-4o-2024-11-20")
    parser.add_argument("--model", default="vitl336")
    parser.add_argument("--image_num", type=int, default=30)
    parser.add_argument("--merge_dist", type=float, default=1.0)
    parser.add_argument("--merge_sim", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nomerge", action="store_true")
    parser.add_argument("--nopose", action="store_true")
    parser.add_argument("--zeroshot", action='store_true')
    args = parser.parse_args()

    # reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    
    main(args)
