import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

import numpy as np

import h5py

from tqdm import tqdm

training_filename = dir+'/data_processed/seqs_max_length_1022_esm2_t6_8M_UR50D_training_finalversion.h5'
validation_filename = dir+'/data_processed/seqs_max_length_1022_esm2_t6_8M_UR50D_validation_finalversion.h5'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HDF5Dataset(Dataset):
  def __init__(self, filepath):
    self.filepath = filepath
    with h5py.File(self.filepath, 'r') as f:
        self.len = len(f['entry_name'])

  def __getitem__(self, index):
    with h5py.File(self.filepath, 'r') as f:
      entry_name = f['entry_name'][index].tobytes().decode('utf-8')
      accession = f['accession'][index].tobytes().decode('utf-8')
      site_residue = f['site_residue'][index].tobytes().decode('utf-8')
      position = f['position_in_protein'][index]
      OGlcNAc = f['OGlcNAc'][index]
      sequence = f['sequence'][index].tobytes().decode('utf-8')
      species = f['species'][index].tobytes().decode('utf-8')
      embedding = f['embedding'][index]
    return {
      'entry_name': entry_name,
      'accession': accession,
      'residue': site_residue,
      'position': position,
      'OGlcNAc': OGlcNAc,
      'sequence': sequence,
      'species': species,
      'embedding': torch.from_numpy(embedding).float() # Convert numpy array to PyTorch tensor
    }

  def __len__(self):
    return self.len

training_dataset = HDF5Dataset(training_filename)
validation_dataset = HDF5Dataset(validation_filename)

train_dataloader = DataLoader(training_dataset, batch_size=32, shuffle=True)
valid_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size1, output_size, dropout_rate=0.5): # Added dropout_rate parameter
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate) # Added dropout layer
        self.fc2 = nn.Linear(hidden_size1, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x) # Applied dropout
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

input_size = 320
hidden_size1 = 18
output_size = 1
dropout_rate = 0.5

model = SimpleMLP(input_size, hidden_size1, output_size, dropout_rate)
model.to(device)

criterion = nn.BCELoss() # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001) # Adam optimizer with learning rate 0.001

# Lists to store loss and metrics
train_losses = []
valid_losses = []
train_accuracies = []
valid_accuracies = []
train_sensitivities = []
valid_sensitivities = []
train_specificities = []
valid_specificities = []
train_f1_scores = []
valid_f1_scores = []
train_mcc_scores = []
valid_mcc_scores = []

# Define the number of epochs
num_epochs = 75

for epoch in range(num_epochs):
    if epoch == 24:
      for param_group in optimizer.param_groups:
          param_group['lr'] = 0.0001
      print(f"Epoch {epoch+1}: Learning rate changed to 0.0001")
    if epoch == 49:
      for param_group in optimizer.param_groups:
          param_group['lr'] = 0.00001
      print(f"Epoch {epoch+1}: Learning rate changed to 0.0001")

    # Training phase
    model.train()  # Set the model to training mode
    running_train_loss = 0.0
    all_train_labels = []
    all_train_predictions = []
    for i, data in tqdm(enumerate(train_dataloader), total = len(train_dataloader), leave=False):
        inputs = data['embedding'].to(device)
        labels = data['OGlcNAc'].float().unsqueeze(1).to(device)

        optimizer.zero_grad()


        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_train_loss += loss.item()

        predicted = (outputs > 0.5).squeeze().long()
        all_train_labels.extend(data['OGlcNAc'].tolist())
        all_train_predictions.extend(predicted.tolist())


    epoch_train_loss = running_train_loss / len(train_dataloader)
    train_losses.append(epoch_train_loss)

    # Calculate training metrics
    all_train_labels = np.array(all_train_labels)
    all_train_predictions = np.array(all_train_predictions)

    train_accuracy = accuracy_score(all_train_labels, all_train_predictions)
    train_accuracies.append(train_accuracy)

    train_sensitivity = recall_score(all_train_labels, all_train_predictions) if (np.sum(all_train_labels) > 0) else 0 # Calculate training sensitivity
    train_sensitivities.append(train_sensitivity)

    train_specificity = np.sum((all_train_predictions == 0) & (all_train_labels == 0)) / np.sum(all_train_labels == 0) if (np.sum(all_train_labels == 0) > 0) else 0 # Calculate training specificity
    train_specificities.append(train_specificity)

    train_f1 = f1_score(all_train_labels, all_train_predictions) if (np.sum(all_train_predictions) + np.sum(all_train_labels) > 0) else 0 # Calculate training f1 score
    train_f1_scores.append(train_f1)

    train_mcc = matthews_corrcoef(all_train_labels, all_train_predictions) if (len(all_train_labels) > 0) else 0 # Calculate training mcc score
    train_mcc_scores.append(train_mcc)


    # Evaluation phase
    model.eval()
    running_valid_loss = 0.0
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), leave=False):
            inputs = data['embedding'].to(device)
            labels = data['OGlcNAc'].to(device)
            outputs = model(inputs)

            valid_loss = criterion(outputs, labels.float().unsqueeze(1))
            running_valid_loss += valid_loss.item()

            predicted = (outputs > 0.5).squeeze().long()

            all_labels.extend(labels.tolist())
            all_predictions.extend(predicted.tolist())

    epoch_valid_loss = running_valid_loss / len(valid_dataloader)
    valid_losses.append(epoch_valid_loss)

    # Calculate evaluation metrics
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)

    accuracy = accuracy_score(all_labels, all_predictions)
    valid_accuracies.append(accuracy)

    TP = np.sum((all_predictions == 1) & (all_labels == 1))
    TN = np.sum((all_predictions == 0) & (all_labels == 0))
    FP = np.sum((all_predictions == 1) & (all_labels == 0))
    FN = np.sum((all_predictions == 0) & (all_labels == 1))

    sensitivity = recall_score(all_labels, all_predictions) if (TP + FN) > 0 else 0
    valid_sensitivities.append(sensitivity)

    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0
    valid_specificities.append(specificity)

    f1 = f1_score(all_labels, all_predictions) if (TP + FP + FN) > 0 else 0
    valid_f1_scores.append(f1)

    mcc = matthews_corrcoef(all_labels, all_predictions) if (TP + TN + FP + FN) > 0 else 0
    valid_mcc_scores.append(mcc)

    #save the model
    torch.save(model.state_dict(), 'models/'+'model_MLP_320_18_1_with_dropout_epoch_'+str(epoch+1)+'_weights.pth')

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_train_loss:.4f}, valid Loss: {epoch_valid_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, valid Accuracy: {accuracy:.4f}, Train Sensitivity: {train_sensitivity:.4f}, valid Sensitivity: {sensitivity:.4f}, Train Specificity: {train_specificity:.4f}, valid Specificity: {specificity:.4f}, Train F1: {train_f1:.4f}, valid F1: {f1:.4f}, Train MCC: {train_mcc:.4f}, valid MCC: {mcc:.4f}")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(12, 6)) # 2x3 grid

# Plot Training and valid Loss
axes[0, 0].plot(range(1, len(train_losses) + 1), train_losses, label='Training Loss', color='blue') # Changed to blue
axes[0, 0].plot(range(1, len(valid_losses) + 1), valid_losses, label='valid Loss', color='orange') # Changed to orange
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].set_title('Training and valid Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot Training and valid Accuracy
axes[0, 1].plot(range(1, len(train_accuracies) + 1), train_accuracies, label='Training Accuracy', color='blue') # Already blue
axes[0, 1].plot(range(1, len(valid_accuracies) + 1), valid_accuracies, label='valid Accuracy', color='orange') # Changed to orange
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].set_title('Training and valid Accuracy')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_ylim([0, 1])

# Plot Training and valid Sensitivity
axes[0, 2].plot(range(1, len(train_sensitivities) + 1), train_sensitivities, label='Training Sensitivity', color='blue') # Changed to blue
axes[0, 2].plot(range(1, len(valid_sensitivities) + 1), valid_sensitivities, label='valid Sensitivity', color='orange') # Changed to orange
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Score')
axes[0, 2].set_title('Training and valid Sensitivity')
axes[0, 2].legend()
axes[0, 2].grid(True)
axes[0, 2].set_ylim([0, 1])

# Plot Training and valid Specificity
axes[1, 0].plot(range(1, len(train_specificities) + 1), train_specificities, label='Training Specificity', color='blue') # Changed to blue
axes[1, 0].plot(range(1, len(valid_specificities) + 1), valid_specificities, label='valid Specificity', color='orange') # Changed to orange
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Training and valid Specificity')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_ylim([0, 1])

# Plot Training and valid F1 Score
axes[1, 1].plot(range(1, len(train_f1_scores) + 1), train_f1_scores, label='Training F1 Score', color='blue') # Changed to blue
axes[1, 1].plot(range(1, len(valid_f1_scores) + 1), valid_f1_scores, label='valid F1 Score', color='orange') # Changed to orange
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Training and valid F1 Score')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_ylim([0, 1])

# Plot Training and valid MCC
axes[1, 2].plot(range(1, len(train_mcc_scores) + 1), train_mcc_scores, label='Training MCC', color='blue') # Changed to blue
axes[1, 2].plot(range(1, len(valid_mcc_scores) + 1), valid_mcc_scores, label='valid MCC', color='orange') # Changed to orange
axes[1, 2].set_xlabel('Epoch')
axes[1, 2].set_ylabel('Score')
axes[1, 2].set_title('Training and valid MCC')
axes[1, 2].legend()
axes[1, 2].grid(True)
axes[1, 2].set_ylim([0, 1])


plt.tight_layout()
plt.show()
