import pandas as pd
import numpy as np

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

import os
import h5py

from tqdm import tqdm

df_entire = pd.read_csv('data_processed/02_entire_dataset.csv')

df_train = pd.read_csv(dir+'data_processed/02_training_dataset_balanced.csv')
df_valid = pd.read_csv(dir+'data_processed/02_validation_dataset_balanced.csv')
df_test = pd.read_csv(dir+'data_processed/02_testing_dataset_balanced.csv')

df_list = [df_train, df_valid, df_test]

#load model
model_name = 'facebook/esm2_t6_8M_UR50D' #smallest model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()

batch_size = 32

for df in df_list:
  seqs = list(df['sequence'].unique())
  df['embedding'] = None

  seq_embs = {}

  for i in tqdm(range(0, len(seqs), batch_size)):
    batch_seqs = seqs[i:i+batch_size]

    inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
      outputs = model(**inputs, output_hidden_states=True) #** unpacks inputs
    last_hidden_states = outputs.hidden_states[-1] #shape is (batch size, max length of sequence in batch + 2, embedding dimension)

    #for loop only in the rows of df where df['sequence'] is one of batch_seqs
    for seq in batch_seqs:
      for j, row in df[df['sequence'] == seq].iterrows():
        df.at[j, 'embedding'] = last_hidden_states[batch_seqs.index(seq), df.at[j, 'position_in_protein'], :].cpu().numpy()

for i, df in enumerate(df_list):
  if i == 0:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_training.csv'
  if i == 1:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_validation.csv'
  if i == 2:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_testing.csv'

  pd.DataFrame(df).to_csv(output_filename, index=False)
  print(f"DataFrame saved to {output_filename}")

for i, df in enumerate(df_list):
  if i == 0:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_training.h5'
  if i == 1:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_validation.h5'
  if i == 2:
    output_filename = 'data_processed/'+'03_seqs_'+model_name[9:]+'_testing.h5'

  if not os.path.exists(output_filename):
    with h5py.File(output_filename, 'w') as f:
      # Save other columns as datasets
      f.create_dataset('entry_name', data=df['entry_name'].values.astype('S')) # Convert to fixed-width string
      f.create_dataset('accession', data=df['accession'].values.astype('S'))
      f.create_dataset('site_residue', data=df['site_residue'].values.astype('S'))
      f.create_dataset('position_in_protein', data=df['position_in_protein'].values)
      f.create_dataset('OGlcNAc', data=df['OGlcNAc'].values)
      f.create_dataset('species', data=df['species'].values.astype('S'))
      f.create_dataset('sequence', data=df['sequence'].values.astype('S')) # Convert to fixed-width string


      # Save the 'embedding' column as a dataset of fixed-size arrays
      f.create_dataset('embedding', data=np.vstack(df['embedding'].values))

    print(f"DataFrame saved to {output_filename}")

batch_size = 64

for df in [df_entire]:
  seqs = list(df['sequence'].unique())
  df['embedding'] = None

  seq_embs = {}

  for i in tqdm(range(0, len(seqs), batch_size)):
    batch_seqs = seqs[i:i+batch_size]

    inputs = tokenizer(batch_seqs, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
      outputs = model(**inputs, output_hidden_states=True) #** unpacks inputs
    last_hidden_states = outputs.hidden_states[-1] #shape is (batch size, max length of sequence in batch + 2, embedding dimension)

    #for loop only in the rows of df where df['sequence'] is one of batch_seqs
    for seq in batch_seqs:
      for j, row in df[df['sequence'] == seq].iterrows():
        df.at[j, 'embedding'] = last_hidden_states[batch_seqs.index(seq), df.at[j, 'position_in_protein'], :].cpu().numpy()

output_filename = 'data_processed/'+'03_seqs_  '+model_name[9:]+'_entire.csv'
pd.DataFrame(df_entire).to_csv(output_filename, index=False)
print(f"DataFrame saved to {output_filename}")
