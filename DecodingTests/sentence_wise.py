import pickle 
from multiprocessing import Pool
from copy import deepcopy
import evaluate 
from transformers import MBart50Tokenizer
from tqdm import tqdm
import numpy as np
root_dir = '/mnt/a99/d0/absathe/infonas/absathe/DecodingTests/Translation/OPUS/triple_test/'

tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50-many-to-many-mmt', use_fast=False)
bleu_global = evaluate.load('bleu')

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

def get_bleu_scores(x):
    ret = x
    ret['bleu'] = []
    for pred in x['translations']:
        bleu = deepcopy(bleu_global)
        try:
            results = bleu.compute(references=[[x['tgt'].strip()]], predictions=[pred], tokenizer=tokenizer.tokenize)
        except Exception as e:
            print(str(e))
            results = {'bleu': 0}
        ret['bleu'].append(results['bleu'])
        del bleu, results
    return ret 

if 'bleu' not in ac_data[0].keys():
    with Pool(64) as p:
        ac_data = list(tqdm(p.imap(get_bleu_scores, ac_data), total=len(ac_data)))
    with open(a_to_c_path, 'wb') as f:
        pickle.dump(ac_data, f)

if 'bleu' not in abc_data[0].keys():
    with Pool(64) as p:
        abc_data = list(tqdm(p.imap(get_bleu_scores, abc_data), total=len(abc_data)))
    with open(a_to_b_to_c_path, 'wb') as f:
        pickle.dump(abc_data, f)

ac_bleu = [bleu_ij for bleu_i in ac_data for bleu_ij in bleu_i['bleu']]
abc_bleu = [bleu_ij for bleu_i in abc_data for bleu_ij in bleu_i['bleu']]

count = 0
overall_bleu = []
for i, x in enumerate(tqdm(ac_data)):
    for j, pred in enumerate(x['translations']):
        if ac_data[i]['bleu'][j] > abc_data[i]['bleu'][j]:
            count += 1
        overall_bleu.append(max(ac_data[i]['bleu'][j], abc_data[i]['bleu'][j]))
print(count)
print(np.mean(ac_bleu))
print(np.mean(abc_bleu))
print(np.mean(overall_bleu))
