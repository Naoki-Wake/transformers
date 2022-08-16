from __future__ import print_function, absolute_import
from typing import List
import os
import os.path as osp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
import json
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> List[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(topk)  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = y_pred.t()  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(y_pred)  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (y_pred == target_reshaped)  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = ind_which_topk_matched_truth.reshape(-1).float()  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(dim=0, keepdim=True)  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]

def train_loop(dataloader, model, loss_fn, optimizer, scheduler, logP):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        # add class bias
        pred = pred + logP
        #import pdb; pdb.set_trace()
        if len(y)>1:
            try:
                loss = loss_fn(pred, y)
                #import pdb; pdb.set_trace()
            except:
                import pdb; pdb.set_trace()
            # Bacqkpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            scheduler.step()

def test_loop(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    accuracies = accuracy(pred, y, topk=(1,3,5))
    accuracies = [a.item() for a in accuracies]
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    # import pdb; pdb.set_trace()
    return accuracies

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.classifier = nn.Linear(768, 11)

    def forward(self, x):
        # remobe the dimension
        x = x.squeeze()
        logits = self.classifier(x)
        return logits

class NeuralNetwork_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_2, self).__init__()
        self.classifier = nn.Linear(768, 100)
        self.classifier2 = nn.Linear(100, 11)

    def forward(self, x):
        # remobe the dimension
        x = x.squeeze()
        logits = self.classifier(x)
        logits2 = self.classifier2(logits)
        return logits2

if __name__ == '__main__':
    results = {}
    learning_rate = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    batch_size = 100
    epochs = 50

    model = NeuralNetwork().to(device)
    #model = NeuralNetwork_2().to(device)
    print(model)
    dataroot='/home/nawake/sthv2/'
    out_dir = osp.join(dataroot, 'videomae/hand_crop_right')
    fp_data = osp.join(out_dir, 'feat_train.npy')
    fp_label = osp.join(out_dir, 'label_train.npy')
    training_data = dataset.prepare_dataset(fp_data, fp_label)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)

    annotation_root='/home/nawake/sthv2/annotations/with_pseudo_largedatanum'
    fp_annotation_train = osp.join(annotation_root, 'breakfast_train_list_videos.txt')
    with open(fp_annotation_train, 'r') as f:
        lines = f.readlines()
    labels = [int(item.split(' ')[1].strip()) for item in lines]
    class_num = len(list(set(labels)))
    class_bias = [labels.count(i) for i in range(class_num)]
    class_bias = np.array(class_bias)
    class_bias = class_bias/np.sum(class_bias)
    logP = torch.from_numpy(np.log(class_bias))
    logP = logP.to(device)
    #import pdb; pdb.set_trace()
    fp_data = osp.join(out_dir, 'feat_val.npy')
    fp_label = osp.join(out_dir, 'label_val.npy')
    validation_data = dataset.prepare_dataset(fp_data, fp_label)
    validation_dataloader = DataLoader(validation_data, batch_size=batch_size)

    fp_data = osp.join(out_dir, 'feat_test.npy')
    fp_label = osp.join(out_dir, 'label_test.npy')
    test_data = dataset.prepare_dataset(fp_data, fp_label)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data))

    loss_fn = nn.CrossEntropyLoss()

    
    for lr in learning_rate:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=0)
        print('Done preparation')
        #train_features, train_labels = next(iter(dataloader))
        #print(f"Feature batch shape: {train_features.size()}")
        #print(f"Labels batch shape: {train_labels.size()}")
        #logits = model(train_features)
        #pred_probab = nn.Softmax(dim=1)(logits)
        ##loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
        #y_pred = pred_probab.argmax(1)
        #print(logits)
        #print(f"Predicted class: {y_pred}")
        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            train_loop(train_dataloader, model, loss_fn, optimizer, scheduler, logP)
            test_loop(test_dataloader, model, loss_fn)
        print("Done!")
        accuracies = test_loop(test_dataloader, model, loss_fn)
        results[lr] = accuracies
        #torch.save(model, PATH)
    # save results to a file
    print(results)
    with open(osp.join(out_dir, 'results_twolayers.json'), 'w') as f:
        json.dump(results, f, indent=4)