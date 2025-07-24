import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

import h5py

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
dropout_rate = 0.5 # Define dropout rate

model = SimpleMLP(input_size, hidden_size1, output_size, dropout_rate)
model_file = 'models/'+'model_MLP_320_18_1_with_dropout_epoch_47_weights.pth'
model.load_state_dict(torch.load(model_file))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

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

training_file = 'data_processed/03_seqs_esm2_t6_8M_UR50D_training.h5'

training_dataset = HDF5Dataset(training_file)
training_dataloader = DataLoader(training_dataset, batch_size=1024, shuffle=False)

new_rows = []
for x in tqdm(training_dataloader):
  x['embedding'] = x['embedding'].to(device)
  result = model(x['embedding'])
  for i in range(len(result)):
    new_row = {'entry_name': x['entry_name'][i], 'accession': x['accession'][i],
               'position': x['position'][i].item(), 'OGlcNAc': x['OGlcNAc'][i].item(),
               'result': round(result[i].item())==x['OGlcNAc'][i].item(), 'probability':result[i].item(),
               'species':x['species'][i], 'sequence': x['sequence'][i]}
    new_rows.append(new_row)

df_train = pd.DataFrame(new_rows)

testing_file = 'data_processed/03_seqs_esm2_t6_8M_UR50D_testing.h5'
testing_dataset = HDF5Dataset(testing_file)
testing_dataloader = DataLoader(testing_dataset, batch_size=1024, shuffle=False)

new_rows = []
for x in tqdm(testing_dataloader):
  x['embedding'] = x['embedding'].to(device)
  result = model(x['embedding'])
  for i in range(len(result)):
    new_row = {'entry_name': x['entry_name'][i], 'accession': x['accession'][i],
               'position': x['position'][i].item(), 'OGlcNAc': x['OGlcNAc'][i].item(),
               'result': round(result[i].item())==x['OGlcNAc'][i].item(), 'probability':result[i].item(),
               'species':x['species'][i], 'sequence': x['sequence'][i]}
    new_rows.append(new_row)

df_test = pd.DataFrame(new_rows)

validation_file = 'data_processed/03_seqs_esm2_t6_8M_UR50D_validation.h5'
validation_dataset = HDF5Dataset(validation_file)
validation_dataloader = DataLoader(validation_dataset, batch_size=1024, shuffle=False)

new_rows = []
for x in tqdm(validation_dataloader):
  x['embedding'] = x['embedding'].to(device)
  result = model(x['embedding'])
  for i in range(len(result)):
    new_row = {'entry_name': x['entry_name'][i], 'accession': x['accession'][i],
               'position': x['position'][i].item(), 'OGlcNAc': x['OGlcNAc'][i].item(),
               'result': round(result[i].item())==x['OGlcNAc'][i].item(), 'probability':result[i].item(),
               'species':x['species'][i], 'sequence': x['sequence'][i]}
    new_rows.append(new_row)

df_val = pd.DataFrame(new_rows)

#map True to Correct and False and Wrong
df_test['result'] = df_test['result'].map({True: 'Correct', False: 'Wrong'})

species_correct_rate = {}
species_list = ['human', 'mouse', 'rat', 'yeast', 'wheat']
for sp in species_list:
  correct_rate = df_test[df_test['species']==sp]['result'].value_counts(normalize=True)['Correct'].item()
  species_correct_rate[sp] = correct_rate
  
#plot bar species verus correct rate
plt.bar(species_correct_rate.keys(), species_correct_rate.values())
plt.xlabel('Species')
plt.ylabel('Correct Prediction')
#plt.title('Percentage of Correct Prediction by Species')
formatter = mticker.FuncFormatter(lambda x, p: f'{x*100:.0f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

for df in [df_train, df_test, df_val]:
  df['prediction'] = 'Not Defined'
  df['prediction'] = df['probability'].apply(lambda x: 1 if x>=0.5 else 0)
  
for df in [df_train, df_test, df_val]:
  df = df[['entry_name', 'accession', 'position', 'OGlcNAc', 'prediction', 'result', 'probability', 'species', 'sequence']]
for i, df in enumerate([df_train, df_test, df_val]):
  y_true = df['OGlcNAc']
  y_pred = df['prediction']
  class_names = ['Positive', 'Negative']


  cm = confusion_matrix(y_true, y_pred)
  print("Confusion Matrix:\n", cm)

  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
  print("\nNormalized Confusion Matrix (Recall):\n", cm_normalized)

  TP = cm[1,1]
  TN = cm[0,0]
  FP = cm[0,1]
  FN = cm[1,0]

  accuracy = (TP + TN) / (TP + TN + FP + FN)
  precision = TP / (TP + FP)
  recall = TP / (TP + FN)
  f1_score = 2 * (precision * recall) / (precision + recall)

  if i == 0:
    print("\nTraining Set:")
  if i == 1:
    print("\nTesting Set:")
  if i == 2:
    print("\nValidation Set:")

  print(f'Accuracy: {accuracy*100:.2f}%')
  print(f'Precision: {precision*100:.2f}%')
  print(f'Recall: {recall*100:.2f}%')
  print(f'F1 Score: {f1_score*100:.2f}%')

TP = cm[1,1]
TN = cm[0,0]
FP = cm[0,1]
FN = cm[1,0]

accuracy = (TP + TN) / (TP + TN + FP + FN)
precision = TP / (TP + FP)
recall = TP / (TP + FN)
f1_score = 2 * (precision * recall) / (precision + recall)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1_score)

def plot_confusion_matrix(cm, class_names, title, cmap=plt.cm.Blues, normalize=False, fmt='d'):
    """
    Plots a confusion matrix using seaborn.

    Args:
        cm (array): The confusion matrix.
        class_names (list): List of class names (labels).
        title (str): Title of the plot.
        cmap (matplotlib.colors.Colormap): Colormap for the heatmap.
        normalize (bool): If True, normalize the matrix by rows (recall).
        fmt (str): Format string for the annotations (e.g., 'd' for integers, '.2f' for floats).
    """
    plt.figure(figsize=(8, 6)) # Adjust figure size as needed
    sns.heatmap(cm,
                annot=True,     # Show the values in the cells
                fmt=fmt,        # Format of the annotations
                cmap=cmap,      # Colormap
                cbar=True,      # Show color bar
                xticklabels=class_names,
                yticklabels=class_names,
                linewidths=.5,  # Add lines between cells for better separation
                linecolor='black')

    plt.title(title, fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12) # Rotate for long labels
    plt.yticks(rotation=0, fontsize=12) # Keep y-axis labels horizontal
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    plt.show()

# Plot the raw confusion matrix
plot_confusion_matrix(cm, class_names, 'Confusion Matrix (Counts)')

# Plot the normalized confusion matrix (recall)
plot_confusion_matrix(cm_normalized, class_names, 'Normalized Confusion Matrix (Recall)',
                      normalize=True, fmt='.2f', cmap=plt.cm.Greens) # Use a different cmap for normalized

df_entire = pd.read_csv(dir+'data_processed/seqs_max_length_1022_esm2_t6_8M_UR50D_entire_finalversion.csv')
df_entire = df_entire.drop_duplicates()

df_entire['embedding'] = df_entire['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
df_entire.head()

df_entire['probability'] = 'Not Defined'
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
   df_entire.loc[i, 'probability'] = model(torch.from_numpy(row['embedding']).float().unsqueeze(0).to(device)).item()

df_entire['prediction'] = df_entire['probability'].apply(lambda x: 1 if x>=0.5 else 0)
df_entire['result'] = df_entire['prediction'] == df_entire['OGlcNAc']
df_entire['result'] = df_entire['result'].map({True: 'Correct', False: 'Wrong'})

df_entire = df_entire[['entry_name', 'accession', 'site_residue',
                   'position_in_protein', 'OGlcNAc', 'prediction',
                   'result', 'probability', 'species', 'sequence']]

for prob_level in [0.99, 0.95, 0.9, 0.5]:
  df_export = df_entire[(df_entire['probability']>=prob_level) & (df_entire['OGlcNAc']==0)]
  df_export = df_export[['entry_name', 'accession', 'site_residue', 'position_in_protein', 'probability', 'species', 'sequence']]
  df_export.rename(columns={'site_residue':'residue', 'entry_name':'entry', 'position_in_protein':'position'}, inplace=True)
  df_export.sort_values(by=['entry', 'accession', 'position'], inplace=True)
  df_export.reset_index(drop=True, inplace=True)
  df_export.to_excel('/data_processed/list_of_new_discoveries_with_probability_greater_or_equal_to_'+str(prob_level)+'.xlsx')

  print(f'{len(df_export)} entries saved to, with probability greater of equal to {prob_level} of being a new discovery.')

species_list = ['human', 'mouse', 'rat', 'yeast', 'wheat', 'Arabidopsis']
for prob_level in [0.99, 0.95, 0.9, 0.5]:
  for sp in species_list:
    df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= prob_level)
    prop = df_entire[df_entire['species']==sp]['discovery'].value_counts(normalize=True)[True]
    print(f'{sp} proportion of new discoveries: {prop*100:.2f}% at probability level {prob_level}')
  print()

species_list = ['human', 'mouse', 'rat', 'yeast', 'wheat']
for i, prob_level in enumerate([0.99, 0.95, 0.9]):
  prop_list = []
  for sp in species_list:
    df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= prob_level)
    prop = df_entire[df_entire['species']==sp]['discovery'].value_counts(normalize=True)[True]
    prop_list.append(prop)
    x = np.arange(len(species_list))
    w = 0.25
  plt.bar(x+(i-1)*w, prop_list, label=prob_level, width=w)
plt.xticks(x, species_list)
plt.xlabel('Species')
plt.ylabel('Percentage of New Discoveries')
#plt.title('Percentage of New Discoveries by Species and Probability Level')
plt.legend(title='Probability Level')
formatter = mticker.FuncFormatter(lambda x, p: f'{x*100:.1f}')
plt.gca().yaxis.set_major_formatter(formatter)
plt.show()

df_entire['discovery'] = False
df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= 0.9)

new_rows = []
previous_entry = ''
st_num = 0
no_discovery = 0
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
  if row['entry_name'] != previous_entry:
    if no_discovery > 0:
      new_row = {'entry':previous_entry, 'accession':previous_acc, 'st_residues':st_num, 'discoveries':no_discovery, 'discovery_ratio':no_discovery/st_num}
      new_rows.append(new_row)
    st_num = 0
    no_discovery = 0

  st_num += 1
  if row['discovery']==True:
    no_discovery += 1
  previous_entry = row['entry_name']
  previous_acc = row['accession']

df_disc_09 = pd.DataFrame(new_rows)

df_entire['discovery'] = False
df_entire['discovery_prob_90_percent'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= 0.9)

new_rows = []
previous_entry = ''
st_num = 0
no_discovery = 0
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
  if row['entry_name'] != previous_entry:
    if no_discovery > 0:
      new_row = {'entry':previous_entry, 'accession':previous_acc, 'st_residues':st_num, 'discoveries':no_discovery, 'discovery_ratio':no_discovery/st_num}
      new_rows.append(new_row)
    st_num = 0
    no_discovery = 0

  st_num += 1
  if row['discovery']==True:
    no_discovery += 1
  previous_entry = row['entry_name']
  previous_acc = row['accession']

df_disc_09 = pd.DataFrame(new_rows)

df_entire['discovery'] = False
df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= 0.95)

new_rows = []
previous_entry = ''
st_num = 0
no_discovery = 0
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
  if row['entry_name'] != previous_entry:
    if no_discovery > 0:
      new_row = {'entry':previous_entry, 'accession':previous_acc, 'st_residues':st_num, 'discoveries':no_discovery, 'discovery_ratio':no_discovery/st_num}
      new_rows.append(new_row)
    st_num = 0
    no_discovery = 0

  st_num += 1
  if row['discovery']==True:
    no_discovery += 1
  previous_entry = row['entry_name']
  previous_acc = row['accession']

df_disc_095 = pd.DataFrame(new_rows)

df_entire['discovery'] = False
df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= 0.99)

new_rows = []
previous_entry = ''
st_num = 0
no_discovery = 0
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
  if row['entry_name'] != previous_entry:
    if no_discovery > 0:
      new_row = {'entry':previous_entry, 'accession':previous_acc, 'st_residues':st_num, 'discoveries':no_discovery, 'discovery_ratio':no_discovery/st_num}
      new_rows.append(new_row)
    st_num = 0
    no_discovery = 0

  st_num += 1
  if row['discovery']==True:
    no_discovery += 1
  previous_entry = row['entry_name']
  previous_acc = row['accession']

df_disc_099 = pd.DataFrame(new_rows)

df_entire['discovery'] = False
df_entire['discovery'] = (df_entire['OGlcNAc']==0) & (df_entire['probability'] >= 0.9999)

new_rows = []
previous_entry = ''
st_num = 0
no_discovery = 0
for i, row in tqdm(df_entire.iterrows(), total=len(df_entire)):
  if row['entry_name'] != previous_entry:
    if no_discovery > 0:
      new_row = {'entry':previous_entry, 'accession':previous_acc, 'st_residues':st_num, 'discoveries':no_discovery, 'discovery_ratio':no_discovery/st_num}
      new_rows.append(new_row)
    st_num = 0
    no_discovery = 0

  st_num += 1
  if row['discovery']==True:
    no_discovery += 1
  previous_entry = row['entry_name']
  previous_acc = row['accession']

df_disc_09999 = pd.DataFrame(new_rows)

#selects only human sequences
df_test_human = df_test[df_test['species']=='human']
#drop duplicates
df_test_human = df_test_human.drop_duplicates(subset=['position', 'accession', 'entry_name'])

#open results mannually obtained from O-GlcNAcPRED-DL server
df_oglcnacpred = pd.read_csv('other_models/prediction_result_all_O-GlcNAcPRED-DL.csv')

#organize the datases
df_oglcnacpred['entry_name']=df_oglcnacpred['ID'].apply(lambda x: x.split('|')[0])
df_oglcnacpred['accession']=df_oglcnacpred['ID'].apply(lambda x: x.split('|')[1])
df_oglcnacpred.drop(columns=['ID'], inplace=True)
df_oglcnacpred.drop(columns=['Confidence level'], inplace=True)

#ensure that the same entries are present in both dataframes
existing_combinations = set(df_test_human[['position', 'accession', 'entry_name']].apply(tuple, axis=1))
df_oglcnacpred = df_oglcnacpred[df_oglcnacpred[['Position', 'accession', 'entry_name']].apply(tuple, axis=1).isin(existing_combinations)]
existing_combinations_2 = set(df_oglcnacpred[['Position', 'accession', 'entry_name']].apply(tuple, axis=1))
df_test_human = df_test_human[df_test_human[['position', 'accession', 'entry_name']].apply(tuple, axis=1).isin(existing_combinations_2)]

df_oglcnacpred['result'] = df_oglcnacpred['O-GlcNAc prediction score'].apply(lambda x: x>=0.5)

df_test_human = df_test_human.sort_values(by=['entry_name', 'accession', 'position'])
df_oglcnacpred = df_oglcnacpred.sort_values(by=['entry_name', 'accession', 'Position'])

df_test_human.reset_index(drop=True, inplace=True)
df_oglcnacpred.reset_index(drop=True, inplace=True)

for id1, row1 in tqdm(df_oglcnacpred.iterrows(), total=len(df_oglcnacpred)):
  if df_test_human.loc[id1, 'entry_name'] == row1['entry_name'] and df_test_human.loc[id1, 'accession'] == row1['accession'] and df_test_human.loc[id1, 'position'] == row1['Position']:
    df_oglcnacpred.loc[id1, 'OGlcNAc'] = int(df_test_human.loc[id1, 'OGlcNAc'])
  else:
    print('Error!') #ensure each entry is being compared to the corresponding one

df_oglcnacpred['result'] = 'Not Defined'
for id, row in df_oglcnacpred.iterrows():
  if (row['O-GlcNAc prediction score'] >= 0.5 and row['OGlcNAc']==1) or (row['O-GlcNAc prediction score'] <= 0.5 and row['OGlcNAc']==0):
    df_oglcnacpred.loc[id, 'result'] = 'Correct'
  else:
    df_oglcnacpred.loc[id, 'result'] = 'Wrong'

df_oglcnacpred['prediction'] = df_oglcnacpred['O-GlcNAc prediction score'].apply(lambda x: 1 if x>=0.5 else 0)

TP_our = len(df_test_human[(df_test_human['prediction']==1) & (df_test_human['OGlcNAc']==1)])
TN_our = len(df_test_human[(df_test_human['prediction']==0) & (df_test_human['OGlcNAc']==0)])
FP_our = len(df_test_human[(df_test_human['prediction']==1) & (df_test_human['OGlcNAc']==0)])
FN_our = len(df_test_human[(df_test_human['prediction']==0) & (df_test_human['OGlcNAc']==1)])

TP_them = len(df_oglcnacpred[(df_oglcnacpred['prediction']==1) & (df_oglcnacpred['OGlcNAc']==1)])
TN_them = len(df_oglcnacpred[(df_oglcnacpred['prediction']==0) & (df_oglcnacpred['OGlcNAc']==0)])
FP_them = len(df_oglcnacpred[(df_oglcnacpred['prediction']==1) & (df_oglcnacpred['OGlcNAc']==0)])
FN_them = len(df_oglcnacpred[(df_oglcnacpred['prediction']==0) & (df_oglcnacpred['OGlcNAc']==1)])

accuracy_our = (TP_our + TN_our) / (TP_our + TN_our + FP_our + FN_our)
accuracy_them = (TP_them + TN_them) / (TP_them + TN_them + FP_them + FN_them)
precision_our = TP_our / (TP_our + FP_our)
precision_them = TP_them / (TP_them + FP_them)
recall_our = TP_our / (TP_our + FN_our)
recall_them = TP_them / (TP_them + FN_them)
F1_our = 2 * (precision_our * recall_our) / (precision_our + recall_our)
F1_them = 2 * (precision_them * recall_them) / (precision_them + recall_them)

print(f'Our model accuracy: {accuracy_our*100:.2f}%')
print(f'Our model precision: {precision_our*100:.2f}%')
print(f'Our model recall: {recall_our*100:.2f}%')
print(f'Our model F1 score: {F1_our*100:.2f}%')
print()
print(f'O-GlcNAcPRED-DL accuracy: {accuracy_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL precision: {precision_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL recall: {recall_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL F1 score: {F1_them*100:.2f}%')

df_test_mouse = df_test[df_test['species']=='mouse']
df_oglcnacpred_mouse = pd.read_csv('other_models/prediction_result_all_mouse_O-GlcNAcPRED-DL.csv')

df_oglcnacpred_mouse['entry_name']=df_oglcnacpred_mouse['ID'].apply(lambda x: x.split('|')[0])
df_oglcnacpred_mouse['accession']=df_oglcnacpred_mouse['ID'].apply(lambda x: x.split('|')[1])
df_oglcnacpred_mouse.drop(columns=['ID'], inplace=True)
df_oglcnacpred_mouse.drop(columns=['Confidence level'], inplace=True)

#drop duplicates
df_test_mouse = df_test_mouse.drop_duplicates(subset=['position', 'accession', 'entry_name'])
df_oglcnacpred_mouse = df_oglcnacpred_mouse.drop_duplicates(subset=['Position', 'accession', 'entry_name'])

existing_combinations = set(df_test_mouse[['position', 'accession', 'entry_name']].apply(tuple, axis=1))
df_oglcnacpred_mouse = df_oglcnacpred_mouse[df_oglcnacpred_mouse[['Position', 'accession', 'entry_name']].apply(tuple, axis=1).isin(existing_combinations)]
existing_combinations_2 = set(df_oglcnacpred_mouse[['Position', 'accession', 'entry_name']].apply(tuple, axis=1))
df_test_mouse = df_test_mouse[df_test_mouse[['position', 'accession', 'entry_name']].apply(tuple, axis=1).isin(existing_combinations_2)]

df_test_mouse = df_test_mouse.sort_values(by=['entry_name', 'accession', 'position'])
df_oglcnacpred_mouse = df_oglcnacpred_mouse.sort_values(by=['entry_name', 'accession', 'Position'])

df_test_mouse.reset_index(drop=True, inplace=True)
df_oglcnacpred_mouse.reset_index(drop=True, inplace=True)

for id1, row1 in tqdm(df_oglcnacpred_mouse.iterrows(), total=len(df_oglcnacpred_mouse)):
  if df_test_mouse.loc[id1, 'entry_name'] == row1['entry_name'] and df_test_mouse.loc[id1, 'accession'] == row1['accession'] and df_test_mouse.loc[id1, 'position'] == row1['Position']:
    df_oglcnacpred_mouse.loc[id1, 'OGlcNAc'] = int(df_test_mouse.loc[id1, 'OGlcNAc'])
  else:
    print('Error!') #ensure each entry is being compared to the corresponding one

df_oglcnacpred_mouse['result'] = 'Not Defined'
for id, row in df_oglcnacpred_mouse.iterrows():
  if (row['O-GlcNAc prediction score'] >= 0.5 and row['OGlcNAc']==1) or (row['O-GlcNAc prediction score'] <= 0.5 and row['OGlcNAc']==0):
    df_oglcnacpred_mouse.loc[id, 'result'] = 'Correct'
  else:
    df_oglcnacpred_mouse.loc[id, 'result'] = 'Wrong'

df_test_mouse['result'].value_counts(normalize=True)

df_oglcnacpred_mouse['result'].value_counts(normalize=True)

df_test_mouse['prediction'] = df_test_mouse['probability'].apply(lambda x: 1 if x>=0.5 else 0)
df_oglcnacpred_mouse['prediction'] = df_oglcnacpred_mouse['O-GlcNAc prediction score'].apply(lambda x: 1 if x>=0.5 else 0)

TP_our = len(df_test_mouse[(df_test_mouse['prediction']==1) & (df_test_mouse['OGlcNAc']==1)])
TN_our = len(df_test_mouse[(df_test_mouse['prediction']==0) & (df_test_mouse['OGlcNAc']==0)])
FP_our = len(df_test_mouse[(df_test_mouse['prediction']==1) & (df_test_mouse['OGlcNAc']==0)])
FN_our = len(df_test_mouse[(df_test_mouse['prediction']==0) & (df_test_mouse['OGlcNAc']==1)])

TP_them = len(df_oglcnacpred_mouse[(df_oglcnacpred_mouse['prediction']==1) & (df_oglcnacpred_mouse['OGlcNAc']==1)])
TN_them = len(df_oglcnacpred_mouse[(df_oglcnacpred_mouse['prediction']==0) & (df_oglcnacpred_mouse['OGlcNAc']==0)])
FP_them = len(df_oglcnacpred_mouse[(df_oglcnacpred_mouse['prediction']==1) & (df_oglcnacpred_mouse['OGlcNAc']==0)])
FN_them = len(df_oglcnacpred_mouse[(df_oglcnacpred_mouse['prediction']==0) & (df_oglcnacpred_mouse['OGlcNAc']==1)])

accuracy_our = (TP_our + TN_our) / (TP_our + TN_our + FP_our + FN_our)
accuracy_them = (TP_them + TN_them) / (TP_them + TN_them + FP_them + FN_them)
precision_our = TP_our / (TP_our + FP_our)
precision_them = TP_them / (TP_them + FP_them)
recall_our = TP_our / (TP_our + FN_our)
recall_them = TP_them / (TP_them + FN_them)
F1_our = 2 * (precision_our * recall_our) / (precision_our + recall_our)
F1_them = 2 * (precision_them * recall_them) / (precision_them + recall_them)

print(f'Our model accuracy: {accuracy_our*100:.2f}%')
print(f'Our model precision: {precision_our*100:.2f}%')
print(f'Our model recall: {recall_our*100:.2f}%')
print(f'Our model F1 score: {F1_our*100:.2f}%')
print()
print(f'O-GlcNAcPRED-DL accuracy: {accuracy_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL precision: {precision_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL recall: {recall_them*100:.2f}%')
print(f'O-GlcNAcPRED-DL F1 score: {F1_them*100:.2f}%')
