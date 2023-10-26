from torch.utils.data import DataLoader, ConcatDataset
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets
from torchvision import datasets, transforms
import torch.nn.functional as F
from sklearn.model_selection import KFold
import itertools
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn

if __name__ == '__main__':

    class CNN(nn.Module):
        def __init__(self):
            super(CNN, self).__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(8, 8)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16, 100)
            self.fc2 = nn.Linear(100, 50)
            self.fc3 = nn.Linear(50, 4)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = x.view(-1, 16)
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    def reset_weights(m):
        '''
          Try resetting model weights to avoid
          weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
    
    classes = ('Cloth', 'N95', 'None' , 'Surgical')
    y_pred_total = []
    y_true_total = []
    num_epochs = 10
    k_folds = 10
    results = {}
    kfold =KFold(n_splits=k_folds, shuffle=True)
    learning_rate = 0.001
    loss_function = nn.CrossEntropyLoss()

    transform = transforms.Compose([transforms.Resize((128,128)),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomRotation(degrees=(30, 70)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(
                                        mean=[0.5383, 0.4914, 0.4727],
                                        std=[0.2934, 0.2795, 0.2810])
        ])

    data_dir = './Dataset_pt2'
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    running_confusion = torch.zeros(4, 4)

    for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
        # Print
        print(f'FOLD {fold + 1}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = DataLoader(dataset, batch_size=2, sampler=train_subsampler)
        test_loader = DataLoader(dataset, batch_size=2, sampler=test_subsampler)

        network = CNN()
        #network.load_state_dict(torch.load('./modelpt1.pth'))
        #network.eval()
        

        network.apply(reset_weights)
        optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)

        for epoch in range(0, num_epochs):
            # Print epoch
            print(f'Starting epoch {epoch + 1}')

            # Set current loss value
            current_loss = 0.0

            # Iterate over the DataLoader for training data
            for i, data in enumerate(train_loader, 0):

                # Get inputs
                inputs, targets = data
                
                # Zero the gradients
                optimizer.zero_grad()

                # Perform forward pass
                outputs = network(inputs)

                # Compute loss
                loss = loss_function(outputs, targets)

                # Perform backward pass
                loss.backward()

                # Perform optimization
                optimizer.step()

                # Print statistics
                current_loss += loss.item()
                if i % 100 == 99:
                    print('Loss after mini-batch %5d: %.3f' %
                          (i + 1, current_loss / 100))
                    current_loss = 0.0

            # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')

        # Saving the model
        save_path = f'./model-fold-{fold+1}.pth'
        torch.save(network.state_dict(), save_path)

        # Evaluation for this fold
        #correct, total = 0, 0
        with torch.no_grad():
            y_pred = []
            y_true = []

# iterate over test data
            for inputs, labels in test_loader:
                output = network(inputs) # Feed Network

                output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
                y_pred.extend(output) # Save Prediction
                y_pred_total.extend(output)
                
                
                labels = labels.data.cpu().numpy()
                y_true.extend(labels)
                y_true_total.extend(labels)

            cf_matrix = confusion_matrix(y_true, y_pred)
            
            
            
            print("Confusion Matrix for fold " + str(fold + 1))
            print(cf_matrix)
            print("Classification report for fold " + str(fold + 1))
            print(classification_report(y_true, y_pred,  target_names = classes))
            
            '''
            Comment out the following code to disable display of confusion matrix after every fold
    
            
            df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix) *10, index = [i for i in classes],
                     columns = [i for i in classes])
            plt.figure(figsize = (12,7))
            sn.heatmap(df_cm, annot=True)
            plt.show()
            '''

            # Print accuracy
            #print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
            print('--------------------------------')
            #results[fold] = 100.0 * (correct / total)


        # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
    print('--------------------------------')
    cf_final = confusion_matrix(y_true_total, y_pred_total)
    print("Total Confusion Matrix Over " + str(fold+1) + " Folds:")
    print(cf_final)
    df_cm = pd.DataFrame(cf_final/np.sum(cf_final) *10, index = [i for i in classes],
             columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))
    
    sn.heatmap(df_cm, annot=True)
    plt.savefig("Output.png")
    plt.show()
    print("Classification Report for " + str(fold+1) + "Folds:")
    print(classification_report(y_true_total , y_pred_total, target_names = classes))
    
    
    