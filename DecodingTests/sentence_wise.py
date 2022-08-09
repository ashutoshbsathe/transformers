import pickle 
import evaluate 
from transformers import MBart50Tokenizer
from tqdm import tqdm
root_dir = '/mnt/a99/d0/absathe/infonas/absathe/DecodingTests/Translation/OPUS/triple_test/'

tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', use_fast=False)

A = 'fr'
B = 'en'
C = 'de'

# Find number of sentences where A->C has a higher BLEU than A->B->C 

a_to_c_path = root_dir + f'/{A}->{C}.pkl'
a_to_b_to_c_path = root_dir + f'/{A}->{B}->{C}.pkl'

with open(a_to_c_path, 'rb') as f:
    ac_data = pickle.load(f)

with open(a_to_b_to_c_path, 'rb') as f:
    abc_data = pickle.load(f)

if 'bleu' not in ac_data[0].keys():
    for i, x in enumerate(tqdm(ac_data)):
        ac_data[i]['bleu'] = []
        for pred in x['translations']:
            bleu = evaluate.load('bleu')
            results = bleu.compute(references=[[x['tgt'].strip()]], predictions=[pred], tokenizer=tokenizer.tokenize)
            ac_data[i]['bleu'].append(results['bleu'])
    with open(a_to_c_path, 'wb') as f:
        pickle.dump(ac_data, f)

if 'bleu' not in abc_data[0].keys():
    compute_bleu = True 
for i, x in enumerate(tqdm(abc_data)):
    if compute_bleu:
        abc_data[i]['bleu'] = []
    for j, pred in x['translations']:
        if compute_bleu:
            bleu = evaluate.load('bleu')
            results = bleu.compute(references=[[x['tgt'].strip()]], predictions=[pred], tokenizer=tokenizer.tokenize)
            abc_data[i]['bleu'].append(results['bleu'])
        if abc_data[i]['bleu'][j] < ac_data[i]['bleu'][j]:
            print(pred)
if compute_bleu:
    with open(a_to_b_to_c_path, 'wb') as f:
        pickle.dump(abc_data, f)
