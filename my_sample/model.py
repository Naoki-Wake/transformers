import os
import os.path as osp
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import dataset
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        #import pdb; pdb.set_trace()
        loss = loss_fn(pred, y)

        # Bacqkpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss, current = loss.item(), batch * len(X)
        print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

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
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.classifier = nn.Linear(768, 11)

    def forward(self, x):
        # remobe the dimension
        x = x.squeeze()
        logits = self.classifier(x)
        return logits
        loss = None
        self.config.problem_type = "single_label_classification"
        if self.config.problem_type == "single_label_classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

if __name__ == '__main__':

    learning_rate = 1e-2
    batch_size = 5
    epochs = 50

    model = NeuralNetwork().to(device)
    print(model)
    dataroot='/home/nawake/sthv2/'
    out_dir = osp.join(dataroot, 'videomae/hand_crop_right')
    fp_data = osp.join(out_dir, 'feat_test.npy')
    fp_label = osp.join(out_dir, 'label_test.npy')
    training_data = dataset.prepare_dataset(fp_data, fp_label)
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
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
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(train_dataloader, model, loss_fn)
    print("Done!")
