import numpy as np
from decord import VideoReader, cpu
import torch
from transformers import VideoMAEFeatureExtractor, VideoMAEForVideoClassification_my_model
from huggingface_hub import hf_hub_download
import os.path as osp
import os
import pandas as pd
import tqdm as tqdm

feature_extractor = VideoMAEFeatureExtractor.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
model = VideoMAEForVideoClassification_my_model.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    converted_len = int(clip_len * frame_sample_rate)
    if converted_len < seg_len:
        end_idx = np.random.randint(converted_len, seg_len)
        start_idx = end_idx - converted_len
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
    indices = sample_frame_indices(clip_len=16, frame_sample_rate=4, seg_len=len(vr))
    buffer = vr.get_batch(indices).asnumpy()

    # create a list of NumPy arrays
    video = [buffer[i] for i in range(buffer.shape[0])]
    inputs = feature_extractor(video, return_tensors="pt")
    with torch.no_grad():
        feat = model(**inputs)
    return feat

dataroot='/home/nawake/sthv2/'
# pseudoaありなしに対応
#annotation_root='/home/nawake/sthv2/annotations/with_pseudo_largedatanum'
#out_dir = osp.join(dataroot, 'videomae/hand_crop_right')
annotation_root='/home/nawake/sthv2/annotations/wo_pseudo'
out_dir = osp.join(dataroot, 'videomae/wo_pseudo_hand_crop_right')
fp_annotation_train = osp.join(annotation_root, 'breakfast_train_list_videos.txt')
fp_annotation_test = osp.join(annotation_root, 'breakfast_test_list_videos.txt')
fp_annotation_val = osp.join(annotation_root, 'breakfast_val_list_videos.txt')

if not osp.exists(out_dir):
    os.makedirs(out_dir)

file_list_test = pd.read_csv(fp_annotation_test, header=None, sep=' ')
feat = []
label = []
fp_video_list = []
print('start extracting features')
for i in tqdm.tqdm(range(len(file_list_test))):
    fp_video = osp.join(dataroot, file_list_test[0][i])
    annotation = file_list_test[1][i]
    #import pdb; pdb.set_trace()
    feat.append(get_feat(fp_video).numpy())
    label.append(annotation)
    fp_video_list.append(fp_video)
    # print(fp_video)
    # print(annotation)
    # print(osp.exists(fp_video))
feat = np.array(feat)
label = np.array(label)

# save
np.save(osp.join(out_dir, 'feat_test.npy'), feat)
np.save(osp.join(out_dir, 'label_test.npy'), label)
# save as csv
with open(osp.join(out_dir, 'fp_video_list_test.csv'), 'w') as f:
    for fp in fp_video_list:
        f.write(fp + '\n')

file_list_val = pd.read_csv(fp_annotation_val, header=None, sep=' ')
feat = []
label = []
fp_video_list = []
print('start extracting features')
for i in tqdm.tqdm(range(len(file_list_val))):
    fp_video = osp.join(dataroot, file_list_val[0][i])
    annotation = file_list_val[1][i]
    feat.append(get_feat(fp_video).numpy())
    label.append(annotation)
    fp_video_list.append(fp_video)
feat = np.array(feat)
label = np.array(label)

# save
np.save(osp.join(out_dir, 'feat_val.npy'), feat)
np.save(osp.join(out_dir, 'label_val.npy'), label)
# save as csv
with open(osp.join(out_dir, 'fp_video_list_val.csv'), 'w') as f:
    for fp in fp_video_list:
        f.write(fp + '\n')


file_list_train = pd.read_csv(fp_annotation_train, header=None, sep=' ')
feat = []
label = []
fp_video_list = []
print('start extracting features')
for i in tqdm.tqdm(range(len(file_list_train))):
    fp_video = osp.join(dataroot, file_list_train[0][i])
    annotation = file_list_train[1][i]
    # print(fp_video)
    # print(annotation)
    # print(osp.exists(fp_video))
    feat.append(get_feat(fp_video).numpy())
    label.append(annotation)
    fp_video_list.append(fp_video)
feat = np.array(feat)
label = np.array(label)

# save
np.save(osp.join(out_dir, 'feat_train.npy'), feat)
np.save(osp.join(out_dir, 'label_train.npy'), label)
# save as csv
with open(osp.join(out_dir, 'fp_video_list_train.csv'), 'w') as f:
    for fp in fp_video_list:
        f.write(fp + '\n')
