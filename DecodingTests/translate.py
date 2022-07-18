import torch 
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from tqdm import tqdm
from new_from_hf import beam_search
from copy import deepcopy
import pickle 

root_dir = '/mnt/a99/d0/absathe/infonas/absathe/DecodingTests/Translation/OPUS/triple_test/'
fname = 'opus.de-fr-test'

A = 'de'
B = 'en'
C = 'fr'

gen_args = dict(
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    output_scores=True,
    return_dict_in_generate=True,
    return_args=True,
)


model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)

a_to_c = False  
a_to_b_to_c = True 
a_to_b_joint_to_c = False

a_sentences = open(root_dir + fname + f'.{A}', encoding='utf-8').readlines()
c_sentences = open(root_dir + fname + f'.{C}', encoding='utf-8').readlines()

with torch.no_grad():
    if a_to_c:
        data = []
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
        tokenizer.src_lang = src_key 
        for src, tgt in tqdm(zip(a_sentences, c_sentences), total=len(a_sentences)):
            encoded_src = tokenizer(src, return_tensors='pt')
            out, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            data.append({
                'src': src,
                'tgt': tgt, 
                'translations': tokenizer.batch_decode(out['sequences'], skip_special_tokens=True),
            })
        with open(root_dir + f'{A}->{C}.pkl', 'wb') as f:
            pickle.dump(data, f)
    elif a_to_b_to_c:
        data = []
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
        tokenizer.src_lang = src_key
        b_sentences = []
        for src, tgt in tqdm(zip(a_sentences, c_sentences), total=len(a_sentences)):
            encoded_src = tokenizer(src, return_tensors='pt')
            out, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            b_sentences.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:2])
        translates = []
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
        tokenizer.src_lang = src_key
        for src in tqdm(b_sentences):
            encoded_src = tokenizer(src, return_tensors='pt')
            out, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            translates.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:2])
        for i, (src, tgt) in enumerate(tqdm(zip(a_sentences, c_sentences), total=len(a_sentences))):
            data.append({
                'src': src,
                'tgt': tgt,
                'translations': translations[i*4:(i+1)*4],
            })
        with open(root_dir + f'{A}->{B}->{C}.pkl', 'wb') as f:
            pickle.dump(data, f)
