#!/usr/bin/env python3

import os
import argparse
import csv
from spatial_feature import SpatialFeature
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import io
import base64
import numpy as np
from PIL import Image
import time

def rotation_matrix_to_euler_zyx(R):

    pitch = np.arcsin(-R[2, 0]) * 180.0 / np.pi

    if np.abs(R[2, 0]) != 1:
        roll = np.arctan2(R[2, 1], R[2, 2]) * 180.0 / np.pi
        yaw = np.arctan2(R[1, 0], R[0, 0]) * 180.0 / np.pi
    else:
        yaw = 0
        if R[2, 0] == -1:
            pitch = 90.0
            roll = np.arctan2(-R[0, 1], R[0, 2]) * 180.0 / np.pi
        else:
            pitch = -90.0
            roll = np.arctan2(-R[0, 1], R[0, 2]) * 180.0 / np.pi

    return roll, pitch, yaw
    
class LLM:
    def __init__(self, model="gpt-4o"):

        self.model = model
        print("model:", self.model)

        if self.model.startswith("gpt") or self.model.startswith("o1") or self.model.startswith("o3"):
            
            import openai
            openai.api_key = os.environ["OPENAI_API_KEY"]

        elif self.model.startswith("gemini"):
            
            import google.generativeai as genai
            genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
    
    def __call__(self, messages):

        if self.model.startswith("gpt") or self.model.startswith("o1") or self.model.startswith("o3"):

            import openai
            
            # parse messages
            msgs = []
            for message in messages:
                msg = {}
                msg["role"] = message["role"]
                msg["content"] = []
                for content in message["content"]:
                    if isinstance(content, dict):
                        msg["content"].append(content)
                    elif isinstance(content, str):
                        c = {
                            "type": "text",
                            "text": content
                        }
                        msg["content"].append(c)
                    elif isinstance(content, Image.Image):
                        image = content
                        with io.BytesIO() as buffer:
                            image.save(buffer, format="JPEG")
                            image_data = buffer.getvalue()
                        base64_image = base64.b64encode(image_data).decode("utf-8")
                        c = {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                        msg["content"].append(c)
                msgs.append(msg)
           
            for i in range(5): # try 5 times
                try:
                    response = openai.chat.completions.create(
                        model=self.model,
                        messages=msgs,
                        )
                    break
                except Exception as e:
                    print("Error is occured:", e)
                    print("cool down and retry...")
                    time.sleep(20)
            res = response.choices[0].message

            answer = res.content

        elif self.model.startswith("gemini"):

            import google.generativeai as genai
            
            # parse messages
            msgs = []
            for message in messages:
                msg = {}
                msg["role"] = message["role"]
                msg["parts"] = message["content"]
                msgs.append(msg)

            model = genai.GenerativeModel(self.model)
            for i in range(5): # try 5 times
                try:
                    response = model.generate_content(msgs)
                    break
                except Exception as e:
                    print("Error is occured:", e)
                    print("cool down and retry...")
                    time.sleep(20)
            answer = response.text

        return answer


class SpatialQA:

    def __init__(self, model, spatial_feature, args, obj=None):
        
        self.llm = LLM(model)

        if args.nopose:
            self.messages = [{
                "role": "user",
                "content": []
            }]    
        else:
            self.messages = [{
                    "role": "user",
                    "content": [
                        "You will be provided images captured from specific camera positions and orientations as follows (z-axis represents up direction):\n"
                        #"You will be provided with images captured from specific camera positions and orientations as follows:\n"
                    ]
                }
            ]
        
        self.spatial_feature = spatial_feature
        
        if args.nomerge:
            indices = np.linspace(0, len(self.spatial_feature.image_paths) - 1, args.image_num, dtype=int)
            self.keyfeatures = {
                "camera_poses": self.spatial_feature.camera_poses[indices],
                "image_paths": [self.spatial_feature.image_paths[i] for i in indices],
            }
        else:
            self.keyfeatures = self.spatial_feature.extract_keyframes(alpha=args.alpha, beta=args.beta, max_frames=args.image_num)
        
        self.images = []
        self.depths = []
                
        for i in range(len(self.keyfeatures["image_paths"])):
            T = self.keyfeatures["camera_poses"][i].cpu().numpy()
            P_T = np.array([[0, 1, 0, 0],
                                [0, 0, -1, 0],
                                [1, 0, 0, 0],
                                [0, 0, 0, 1]], dtype=np.float32)
            T = T @ P_T
            R = T[:3, :3]
            position = T[:3, 3]
            x = position[0]
            y = position[1]
            z = position[2]
            
            roll, pitch, yaw = rotation_matrix_to_euler_zyx(R)

            if args.nopose:
                pass
                
            else:
            
                camera_prompt = (f"\n**Camera position**: [x={x:.2f}m, y={y:.2f}m, z={z:.2f}m]\n"
                    + f"**Camera rotation**: [x={roll:.1f}°, y={pitch:.1f}°, z={yaw:.1f}°]\n"
                    + "**Image data**: ")

                self.messages[0]["content"].append(camera_prompt)
                print(camera_prompt)


            rgb_path = self.keyfeatures["image_paths"][i]
            print(rgb_path)
            image = Image.open(rgb_path)
            self.images.append(transforms.ToTensor()(image).unsqueeze(0).cuda())
            width, height = image.size
            short_edge = 336
            if width < height:
                scale = short_edge / width
            else:
                scale = short_edge / height
            new_size = (int(width * scale), int(height * scale))
            resized_image = image.resize(new_size, resample=Image.Resampling.LANCZOS)
            self.messages[0]["content"].append(resized_image)

    def __call__(self, question, detection=False):

        messages = list(self.messages)
        
        if isinstance(question, str): 
            messages.append({
                "role": "user",
                "content": [question]
            })
        elif isinstance(question, dict):
            messages.append(question)
        elif isinstance(question, list):
            for q in question:
                messages.append(q)
        
        res = self.llm(messages)
        
        return res
            
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--llm", default="gpt-4o-2024-11-20")
    parser.add_argument("-feat", "--feature", help="path to spatial embedding")
    parser.add_argument("--image_num", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--nomerge", action="store_true")
    parser.add_argument("--nopose", action="store_true")
    parser.add_argument("--noannotation", action="store_true")
    args = parser.parse_args()
    
    se = SpatialFeature.load(args.feature)
    sqa = SpatialQA(args.llm, se, args)

    if not args.noannotation:
        sqa.messages[-1]["content"].append('Note that the user does not know what the images you have. Therefore, you should answer the question as concisely as possible without directly referring to the image with words like “image" or “photo."')
    

    print("Please feel free to ask any question about this environment.")
    
    prompt = input()
    
    while len(prompt) > 0:
        res = sqa(prompt)
        print(res)
        prompt = input()
    
      
        
