# something-something-v2でプリトレイン済みのモデルではなく、ベースのモデルを使う。
import numpy as np
from decord import VideoReader, cpu
import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification_my_model
from huggingface_hub import hf_hub_download
import os.path as osp
import os
import pandas as pd
import tqdm as tqdm
from glob import glob
import pickle
from transformers import logging
logging.set_verbosity_error()
import json
import asyncio
import pdb
import argparse
parser = argparse.ArgumentParser(description='for sandbox use')
parser.add_argument('--video_root')
parser.add_argument('--fp_annotation')
parser.add_argument('--dir_out')
args = parser.parse_args()

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-ssv2")
model = VideoMAEForVideoClassification_my_model.from_pretrained("MCG-NJU/videomae-base-ssv2")
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("Device : ", device)
device = "cpu"
#pdb.set_trace()
#model = model.to(device)
# assuming that the video is first resized and then cropped
assert feature_extractor.do_resize == True
#feature_extractor.do_center_crop = False

#def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
#    indices = np.linspace(0, seg_len, num=clip_len)
#    indices = np.clip(indices, 0, seg_len - 1).astype(np.int64)
#    return indices
def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len < seg_len:
        #import pdb; pdb.set_trace()
        #end_idx = np.random.randint(converted_len, seg_len)
        #start_idx = end_idx - converted_len
        start_idx = int((seg_len - converted_len)/2)
        end_idx = start_idx + converted_len
    else:
        end_idx = seg_len
        start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def get_feat(fp_video):
    vr = VideoReader(fp_video, num_threads=1, ctx=cpu(0))
    # sample 16 frames
    vr.seek(0)
    # NOTE: experimentally found that frame_sample_rate=4 is the best
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))

    buffer = vr.get_batch(indices).asnumpy()
    # create a list of NumPy arrays
    video = [buffer[i] for i in range(buffer.shape[0])]
    inputs = feature_extractor(video, return_tensors="pt")
    #inputs = inputs.to(device)
    with torch.no_grad():
        feat = model(**inputs)
    return feat

# obtain original file paths
#video_root = '/home/nawake/ssv2/dataset/videos'
#fp_annotation = '/home/nawake/ssv2/dataset/annotations/something-something-v2-train.json'
#dir_out = '/home/nawake/ssv2/dataset/features/videoMAE'
video_root = args.video_root
fp_annotation = args.fp_annotation
dir_out = args.dir_out

fp_cohesion = fp_annotation.replace('something-something-v2-train.json', 'cohesion_label.csv')
df = pd.read_csv(fp_cohesion)
phrases = df['ssv2label']
flag = df['is_manipulation']
skip_phrases = phrases[flag == 'None']
import re
skip_phrases = [re.findall('[A-Z][^A-Z]*', i) for i in skip_phrases]
skip_phrases = [' '.join(item) for item in skip_phrases] 
skip_phrases = [item.lower().capitalize()
  for item in skip_phrases]
skip_phrases = [item.replace('_',' ')
  for item in skip_phrases]
skip_phrases = [item.replace('  ',' ')
  for item in skip_phrases]
#phrases = list(set(phrases))

#skip_phrases = ['tilting''tipping','touching','trying','Tturning', 'uncovering','unfolding']
# filelist
with open(fp_annotation, "r") as f:
    annotation = json.load(f)
fp_videos = [osp.join(video_root, y['id']+'.webm') for y in annotation]
str_annotations = [ y['template'].replace('[','').replace(']','').replace(',','').replace('\'',' ').replace('(','').replace(')','') for y in annotation]

# Debugging label parsing
#for i in str_annotations:
#    if i not in skip_phrases:
#        print('label error')
#        included = False
#        count = 0
#        for j in skip_phrases:
#            if j in i:
#                included = True
#                count +=1
#        if included == False:
#            print('content error')
#        if count !=1:
#            print('count error')

calculated_list = os.listdir(dir_out)
fp_videos_multithread = []
for video, item_annotation in zip(fp_videos,str_annotations):
    if item_annotation not in skip_phrases:
        out_name = osp.join(dir_out, osp.basename(video).split('.')[0]+'.npy')
        #pdb.set_trace()
        if osp.basename(out_name) not in calculated_list:
            fp_videos_multithread.append(video)
# pdb.set_trace()        
# extract feature and save
import random
random.shuffle(fp_videos_multithread)
for video in tqdm.tqdm(fp_videos_multithread):
    out_name = osp.join(dir_out, osp.basename(video).split('.')[0]+'.npy')
    #calculated_list = os.listdir(dir_out)
    if not os.path.exists(out_name):
        feat = get_feat(video)
        np.save(out_name, feat.numpy())
        print('saved:')
    else:
        print('file skipped:')
    print(out_name)
    #print('number of files wrote: '+str(len(os.listdir(dir_out))))
