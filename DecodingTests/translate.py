import torch 
import datasets
import evaluate
from transformers import MBart50Tokenizer, MBartForConditionalGeneration
from tqdm import tqdm
import numpy as np
from copy import deepcopy
import pickle
import os
from new_from_hf import beam_search

root_dir = '/mnt/a99/d0/absathe/infonas/absathe/DecodingTests/Translation/OPUS/triple_test/'
fname = 'opus.de-fr-test'

A = 'fr'
B = 'en'
C = 'de'
num_b = 2

gen_args = dict(
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    output_scores=True,
    return_dict_in_generate=True,
    return_args=True,
)

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt").cuda()
tokenizer = MBart50Tokenizer.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)

a_to_c = False 
a_to_b_to_c = False 
a_to_b_joint_to_c = False
a_to_b_multiple_to_c = True

a_sentences = open(root_dir + fname + f'.{A}', encoding='utf-8').readlines()
c_sentences = open(root_dir + fname + f'.{C}', encoding='utf-8').readlines()

num_samples = len(a_sentences)

with torch.no_grad():
    if a_to_c:
        data = []
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
        tokenizer.src_lang = src_key 
        for src, tgt in tqdm(zip(a_sentences[:num_samples], c_sentences[:num_samples]), total=num_samples):
            encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
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
    if a_to_b_to_c:
        data = []
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
        tokenizer.src_lang = src_key
        b_sentences = []
        for src, tgt in tqdm(zip(a_sentences[:num_samples], c_sentences[:num_samples]), total=num_samples):
            encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
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
            encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
            out, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            translates.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:2])
        for i, (src, tgt) in enumerate(tqdm(zip(a_sentences[:num_samples], c_sentences[:num_samples]), total=num_samples)):
            data.append({
                'src': src,
                'tgt': tgt,
                'translations': translates[i*4:(i+1)*4],
            })
        with open(root_dir + f'{A}->{B}->{C}.pkl', 'wb') as f:
            pickle.dump(data, f)
    if a_to_b_joint_to_c:
        # Going A to C for collecting args
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
        tokenizer.src_lang = src_key
        a_to_c_args = [] 
        # Need only args 
        for src in tqdm(a_sentences[:num_samples]):
            encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
            _, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            a_to_c_args.append(args)

        # Going A to B 
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
        tokenizer.src_lang = src_key
        b_sentences = []
        # Need only 2 of the B 
        for src in tqdm(a_sentences[:num_samples]):
            encoded_src = {k:v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
            out, _ = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                **encoded_src,
                **gen_args,
            )
            b_sentences.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:2])
        
        # Going from B to C and collecting args
        src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
        tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
        tokenizer.src_lang = src_key
        b_to_c_args = []
        for src in tqdm(b_sentences):
            encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
            _, args = model.generate(
                forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key], 
                **encoded_src, 
                **gen_args,
            )
            b_to_c_args.append(args)
        for p in np.arange(-1, 2.5, 0.5):
            for w in [1./2, 1, 2]:
                # Joint A to C and B to C
                print(f'p={p:.3f}, w_ac={w:.3f}, w_bc=1.000')
                data = []
                translates = []
                for i_b, src_b in enumerate(tqdm(b_sentences)):
                    i_a = int(i_b/2)
            
                    ac_args = deepcopy(a_to_c_args[i_a])
                    bc_args = deepcopy(b_to_c_args[i_b])
                    ac_model_kwargs = ac_args.pop('model_kwargs')
                    bc_model_kwargs = bc_args.pop('model_kwargs')

                    out = beam_search(model, **deepcopy(ac_args), model2=model, model2_kwargs=deepcopy(bc_model_kwargs), **deepcopy(ac_model_kwargs), gmf_kwargs={'p': p, 'w1': w, 'w2': 1})

                    translates.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:2])

                predictions = []
                references = []
                bleu = evaluate.load('bleu', keep_in_memory=True)
                for i, (src, tgt) in enumerate(tqdm(zip(a_sentences[:num_samples], c_sentences[:num_samples]), total=len(a_sentences[:num_samples]))):
                    data.append({
                        'src': src,
                        'tgt': tgt,
                        'translations': translates[i*4:(i+1)*4],
                    })
                    for pred in translates[i*4:(i+1)*4]:
                        predictions.append(pred)
                        references.append([tgt])
                assert len(data) == len(a_sentences[:num_samples])
                output = bleu.compute(predictions=predictions, references=references)
                print(output)
                bleu_score = output['bleu']
                with open(root_dir + f'({A}+->{B})->{C}|p={p:.3f}|w_ac={w:.3f}|w_bc=1|bleu={bleu_score:.3f}.pkl', 'wb') as f:
                    pickle.dump(data, f)
    if a_to_b_multiple_to_c:
        save_dir = root_dir + '/multiple_b_joint_to_c/'
        os.makedirs(save_dir, exist_ok=True)

        b_sentences_save_path = save_dir + f'{A}->({B})+->{C}.b_sentences.pkl'
        if os.path.exists(b_sentences_save_path):
            with open(b_sentences_save_path, 'rb') as f:
                b_sentences = pickle.load(f)
        else:
            # Going A to B 
            src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(A))
            tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
            tokenizer.src_lang = src_key
            b_sentences = []
            # Need only 2 of the B 
            for src in tqdm(a_sentences[:num_samples]):
                encoded_src = {k:v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
                out, _ = model.generate(
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key],
                    **encoded_src,
                    **gen_args,
                )
                b_sentences.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True)[:num_b])
            with open(b_sentences_save_path, 'wb') as f:
                pickle.dump(b_sentences, f)

        b_to_c_args_save_path = save_dir + f'{A}->({B})+->{C}.b_to_c_args.pkl'
        if os.path.exists(b_to_c_args_save_path):
            with open(b_to_c_args_save_path, 'rb') as f:
                b_to_c_args = pickle.load(f)
        else:
            # Going from B to C and collecting args
            src_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(B))
            tgt_key = max(tokenizer.lang_code_to_id.keys(), key=lambda x: x.startswith(C))
            tokenizer.src_lang = src_key
            b_to_c_args = []
            for src in tqdm(b_sentences):
                encoded_src = {k: v.cuda() for k, v in tokenizer(src, return_tensors='pt').items()}
                _, args = model.generate(
                    forced_bos_token_id=tokenizer.lang_code_to_id[tgt_key], 
                    **encoded_src, 
                    **gen_args,
                )
                b_to_c_args.append(args)
            with open(b_to_c_args_save_path, 'wb') as f:
                pickle.dump(b_to_c_args, f)

        for p in np.arange(-5, 5.5, 0.5):
            for w in [1]:
                # Joint B1 to C and B2 to C
                print(f'p={p:.3f}, w_b1c={w:.3f}, w_b2c=1.000')
                data = []
                translates = []
                for idx in tqdm(range(len(b_sentences) // num_b)):
                    # TODO: Generalize to more than 2 models
                    idx_1 = idx * 2 
                    idx_2 = idx * 2 + 1

                    b1c_args = deepcopy(b_to_c_args[idx_1])
                    b2c_args = deepcopy(b_to_c_args[idx_2])
                    b1c_model_kwargs = b1c_args.pop('model_kwargs')
                    b2c_model_kwargs = b2c_args.pop('model_kwargs')

                    out = beam_search(model, **deepcopy(b1c_args), model2=model, model2_kwargs=deepcopy(b2c_model_kwargs), **deepcopy(b1c_model_kwargs), gmf_kwargs={'p': p, 'w1': w, 'w2': 1})

                    translates.extend(tokenizer.batch_decode(out['sequences'], skip_special_tokens=True))

                predictions = []
                references = []
                bleu = evaluate.load('bleu', keep_in_memory=True)
                for i, (src, tgt) in enumerate(tqdm(zip(a_sentences[:num_samples], c_sentences[:num_samples]), total=len(a_sentences[:num_samples]))):
                    data.append({
                        'src': src,
                        'tgt': tgt,
                        'translations': translates[i*4:(i+1)*4],
                    })
                    for pred in translates[i*4:(i+1)*4]:
                        predictions.append(pred)
                        references.append([tgt])
                assert len(data) == len(a_sentences[:num_samples])
                output = bleu.compute(predictions=predictions, references=references)
                print(output)
                bleu_score = output['bleu']
                with open(root_dir + f'{A}->({B})+->{C}|p={p:.3f}|w_b1c={w:.3f}|w_b2c=1|bleu={bleu_score:.3f}.pkl', 'wb') as f:
                    pickle.dump(data, f)



files = [
    root_dir + f'{A}->{C}.pkl',
    root_dir + f'{A}->{B}->{C}.pkl',
    root_dir + f'({A}+->{B})->{C}.pkl',
]
for fname in files:
    print(fname)
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    bleu = evaluate.load('bleu', keep_in_memory=True)
    predictions = []
    references = []
    for datum in tqdm(data):
        for pred in datum['translations'][:1]:
            predictions.append(pred)
            references.append([datum['tgt']])
    print('BLEU:', bleu.compute(predictions=predictions, references=references))
    print(64*'-')
