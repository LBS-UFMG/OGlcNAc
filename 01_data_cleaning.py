import pandas as pd
import requests
from tqdm import tqdm

df = pd.read_csv(dir+'data/Atlas 4.0_unambiguous sites_20241217.csv')

total_length = len(df)
print('Total number of entries:', total_length)

#count number of rows where site_residue is NOT S or T
not_S_or_T = total_length - len(df[df['site_residue'].isin(['S', 'T'])])
print(f'{not_S_or_T} sequences were excluded out of {total_length}.This is {not_S_or_T/total_length*100:.2f}% of the dataset.')

#keep only rows where site_residue is S or T
df = df[df['site_residue'].isin(['S', 'T'])]

#keep only entries where accession_source is UniProt or Uniprot
df = df[df['accession_source'].isin(['UniProt', 'Uniprot'])]
excluded_by_not_being_uniprot = total_length - len(df)
print(f'{total_length-len(df)} sequences were excluded out of {total_length}.This is {((total_length-len(df))/total_length)*100:.2f}% of the dataset.')

#number of rows where position_in_protein is NaN
position_in_protein_nan = df['position_in_protein'].isna().sum()
print(f'{position_in_protein_nan} sequences were excluded out of {total_length}.This is {position_in_protein_nan/total_length*100:.2f}% of the dataset.')
df = df.dropna(subset=['position_in_protein'])

#fetch sequences from UniProt
acc_seq = {}
for acc in tqdm(df['accession'].unique()):
  #fetch in uniprot sequence with accession acc
  url = 'https://rest.uniprot.org/uniprotkb/'+acc+'.fasta'
  response = requests.get(url)
  fasta_content  = response.text
  lines = fasta_content.splitlines()
  seq = ''.join(lines[1:])
  acc_seq[acc] = seq

#match for each row in df the acc to sequence
df['sequence'] = df['accession'].map(acc_seq)

#check if each accession is uniquely mapped to a sequence
for id, row in df.iterrows():
  if row['sequence'] != acc_seq[row['accession']]:
    print(row)

#check for mismatching glycosilation sites
out_of_range=0
bad_position=0
for id, row in df.iterrows():
  pos = int(row['position_in_protein'])
  if pos > len(row['sequence']):
    out_of_range+=1
    df = df.drop(id)
  else:
    if row['sequence'][pos-1] != row['site_residue']:
      bad_position += 1
      df = df.drop(id)

print(f'Out of {len(df)} entries, there are {out_of_range} sequences where the indicated glycosaliton position is outside the length of the protein.')
print(f'Out of {len(df)} entries, there are {bad_position} sequences where the indicated glycosaliton position does not match the residue.')
print(f'In total, {out_of_range+bad_position} entries were dropped, this is {(out_of_range+bad_position)/total_length*100:.2f}% of the dataset.')

#number of entries with nan sequence
print('Not fetched sequences:', df['sequence'].isna().sum())

#convert position_in_protein to integer
df['position_in_protein'] = df['position_in_protein'].astype(int)

#small report
print(f'There were {total_length} entries initially')
print(f'{not_S_or_T} were excluded because the glycosilation site residue was nor S or T')
print(f'{position_in_protein_nan} were excluded because the glycosilation position was NaN')
print(f'{out_of_range} were excluded because the glycosilation position was outside the length of the protein')
print(f'{bad_position} were excluded because the glycosilation position did not match the indicated residue')
print(f'There are {len(df)} entries left, corresponding to {len(df)/total_length*100:.2f}% of the original dataset')

#keep only sequences of length up to 1022
old_num_entries = len(df)
old_num_seqs = len(df['sequence'].unique())
df = df[df['sequence'].str.len() <= 1022]
num_entries = len(df)
num_seqs = len(df['sequence'].unique())
print(f'{num_entries} were kept out of {old_num_entries}. This is {num_entries/old_num_entries*100:.2f}% of the original dataset.')
print(f'{num_seqs} were kept out of {old_num_seqs}. This is {num_seqs/old_num_seqs*100:.2f}% of the original dataset.')

#save to csv file
df.to_csv(dir+'data_processed/01_cleaned_dataset.csv', index=False)
