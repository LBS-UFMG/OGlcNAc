import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('data_processed/01_cleaned_dataset.csv')

#keep important columns
df = df[['entry_name', 'accession', 'site_residue', 'position_in_protein', 'sequence', 'species']]

#mark every entry as a positive example
df['OGlcNAc'] = 1

def find_ST(seq):
  return [pos+1 for pos, aa in enumerate(seq) if (aa == 'S' or aa == 'T')]

#dictionaries that relate accession to sequence, species and entry name
acc_seq = {}
acc_spc = {}
acc_ent = {}
for index, row in df.iterrows():
  if row['accession'] not in acc_seq:
    acc_seq[row['accession']] = row['sequence']
  if row['accession'] not in acc_spc:
    acc_spc[row['accession']] = row['species']
  if row['accession'] not in acc_ent:
    acc_ent[row['accession']] = row['entry_name']

#it creates a new entry for each serine or threonine of each sequence that
#is not a known O-Glc-NAc site
new_rows = []
existing_combinations = set(df[['position_in_protein', 'accession']].apply(tuple, axis=1))
for acc in df['accession'].unique():
  seq = acc_seq[acc]
  for pos in find_ST(seq):
    #check if there exists a row with this combination of Sequence and Position
    if (pos, acc) not in existing_combinations:
      new_row = {'entry_name': acc_ent[acc], 'accession' : acc, 'site_residue': seq[pos-1], 
                 'position_in_protein': pos, 'sequence': seq, 'species':acc_spc[acc], 'OGlcNAc': 0}
      new_rows.append(new_row)

df_new_rows = pd.DataFrame(new_rows)
df = pd.concat([df, df_new_rows])

#Verifiy, if for each species, indeed, there are more positive examples than negative examples
for sp in df['species'].unique():
  if len(df[(df['species']==sp) & (df['OGlcNAc']==0)]) < len(df[(df['species']==sp) & (df['OGlcNAc']==1)]):
    print(sp)

#creates training, validation and testing datasets
#it keeps the balance of positive/negative cases for each dataset
df_train_balanced = pd.DataFrame(columns=df.columns)
df_valid_balanced = pd.DataFrame(columns=df.columns)
df_test_balanced = pd.DataFrame(columns=df.columns)

for sp in df['species'].unique():
  df_sp_pos_temp = df[(df['species']==sp) & (df['OGlcNAc']==1)]
  len_temp = len(df_sp_pos_temp)
  df_sp_neg_temp = df[(df['species']==sp) & (df['OGlcNAc']==0)].sample(n=len_temp, random_state=1)

  if len_temp>=10:
    df_train_sp_pos, df_temp_sp_pos = train_test_split(df_sp_pos_temp, test_size=0.2, random_state=1)
    df_valid_sp_pos, df_test_sp_pos = train_test_split(df_temp_sp_pos, test_size=0.5, random_state=1)

    df_train_sp_neg, df_temp_sp_neg = train_test_split(df_sp_neg_temp, test_size=0.2, random_state=1)
    df_valid_sp_neg, df_test_sp_neg = train_test_split(df_temp_sp_neg, test_size=0.5, random_state=1)

    df_train_sp = pd.concat([df_train_sp_pos, df_train_sp_neg])
    df_valid_sp = pd.concat([df_valid_sp_pos, df_valid_sp_neg])
    df_test_sp = pd.concat([df_test_sp_pos, df_test_sp_neg])
  else:
    df_train_sp = pd.concat([df_sp_pos_temp, df_sp_neg_temp])
    df_valid_sp = pd.DataFrame(columns=df.columns)
    df_test_sp = pd.DataFrame(columns=df.columns)

  df_train_balanced = pd.concat([df_train_balanced, df_train_sp])
  df_valid_balanced = pd.concat([df_valid_balanced, df_valid_sp])
  df_test_balanced = pd.concat([df_test_balanced, df_test_sp])

df.to_csv('data_processed/02_entire_dataset.csv', index=False)
df_train_balanced.to_csv('data_processed/02_training_dataset_balanced.csv', index=False)
df_valid_balanced.to_csv('data_processed/02_validation_dataset_balanced.csv', index=False)
df_test_balanced.to_csv('data_processed/02_testing_dataset_balanced.csv', index=False)
