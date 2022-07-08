# Other imports 
from copy import deepcopy 
import torch
# Standalone beam_search 
from new_from_hf import beam_search
# UnifiedSKG
import sys 
sys.path.append('/mnt/a99/d0/absathe/infonas/absathe/DecodingTests/UnifiedSKG')
from models.unified.prefixtuning import Model 
from utils.configue import Configure # ITS GODDAMN "CONFIGUE"

args = Configure.Get('Salesforce/T5_base_prefix_sql2text.cfg')
unifiedskg = Model(args).cuda()
unifiedskg.load('hkunlp/from_all_T5_base_prefix_sql2text2')
unifiedskg.eval()

# Common 
from transformers import AutoTokenizer, T5ForConditionalGeneration
tok = AutoTokenizer.from_pretrained('t5-base')
gen_args = dict(
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    output_scores=True,
    return_dict_in_generate=True,
    return_args=True,
)
# Paraphrase 
paraphrase = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').cuda()
paraphrase.eval()

# Inputs 
sql2text_input = 'SELECT COUNT(prerequisite) FROM course WHERE name = "Advanced Machine Learning" ; structed knowledge: '
paraphrase_input = 'paraphrase: What number of prerequisites exist for the course "Advanced Machine Learning"? </s>'

with torch.no_grad():

    # Standalone UnifiedSKG
    out1, unifiedskg_args = unifiedskg.generate(
        **{k: v.cuda() for k, v in tok(sql2text_input, return_tensors='pt').items()},
        **gen_args,
    )
    print('Standalone UnifiedSKG [HF generate]')
    print(f'Input: ""{sql2text_input}""\nOutput:')
    for i, seq in enumerate(tok.batch_decode(out1['sequences'])):
        print(f'{i+1} {out1["sequences_scores"][i].item():.3f} {seq}')
    print(64*'-')

    unifiedskg_model_kwargs = unifiedskg_args.pop('model_kwargs')
    out2 = beam_search(unifiedskg.pretrain_model, **deepcopy(unifiedskg_args), **deepcopy(unifiedskg_model_kwargs))
    print('Standalone UnifiedSKG [Standalone beam_search]')
    print(f'Input: ""{sql2text_input}""\nOutput:')
    for i, seq in enumerate(tok.batch_decode(out2['sequences'])):
        print(f'{i+1} {out2["sequences_scores"][i].item():.3f} {seq}')
    print(64*'-')
    print(64*'-')

    # Standalone Paraphrase 
    out3, paraphrase_args = paraphrase.generate(
        **{k: v.cuda() for k, v in tok(paraphrase_input, return_tensors='pt').items()},
        **gen_args,
    )
    print('Standalone Paraphrase [HF generate]')
    print(f'Input: ""{paraphrase_input}""\nOutput:')
    for i, seq in enumerate(tok.batch_decode(out3['sequences'])):
        print(f'{i+1} {out3["sequences_scores"][i].item():.3f} {seq}')
    print(64*'-')

    paraphrase_model_kwargs = paraphrase_args.pop('model_kwargs')
    out4 = beam_search(paraphrase, **deepcopy(paraphrase_args), **deepcopy(paraphrase_model_kwargs))
    print('Standalone Paraphrase [Standalone beam_search]')
    print(f'Input: ""{paraphrase_input}""\nOutput:')
    for i, seq in enumerate(tok.batch_decode(out4['sequences'])):
        print(f'{i+1} {out4["sequences_scores"][i].item():.3f} {seq}')
    print(64*'-')
    print(64*'-')

    # Joint decoding 
    out5 = beam_search(paraphrase, **deepcopy(paraphrase_args), model2=unifiedskg.pretrain_model, model2_kwargs=deepcopy(unifiedskg_model_kwargs), **deepcopy(paraphrase_model_kwargs))
    print('Joint decoding: 1 - Paraphrase, 2 - UnifiedSKG: p = 0')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out5['sequences'])):
        print(f'{i+1} {out5["sequences_scores"][i].item():.3f} {seq}\n{out5["sequences"][i]}')
    print(64*'-')

    out6 = beam_search(unifiedskg.pretrain_model, **deepcopy(unifiedskg_args), model2=paraphrase, model2_kwargs=deepcopy(paraphrase_model_kwargs), **deepcopy(unifiedskg_model_kwargs))
    print('Joint decoding: 1 - UnifiedSKG, 2 - Paraphrase: p = 0')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out6['sequences'])):
        print(f'{i+1} {out6["sequences_scores"][i].item():.3f} {seq}\n{out6["sequences"][i]}')
    print(64*'-')

    out7 = beam_search(unifiedskg.pretrain_model, **deepcopy(unifiedskg_args), model2=unifiedskg.pretrain_model, model2_kwargs=deepcopy(unifiedskg_model_kwargs), **deepcopy(unifiedskg_model_kwargs))
    print('Joint decoding: 1 - UnifiedSKG, 2 - UnifiedSKG: p = 0')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out7['sequences'])):
        print(f'{i+1} {out7["sequences_scores"][i].item():.3f} {seq}\n{out7["sequences"][i]}')
    print(64*'-')

    out8 = beam_search(paraphrase, **deepcopy(paraphrase_args), model2=paraphrase, model2_kwargs=deepcopy(paraphrase_model_kwargs), **deepcopy(paraphrase_model_kwargs))
    print('Joint decoding: 1 - Paraphrase, 2 - Paraphrase: p = 0')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out8['sequences'])):
        print(f'{i+1} {out8["sequences_scores"][i].item():.3f} {seq}\n{out8["sequences"][i]}')
    print(64*'-')

    out9 = beam_search(unifiedskg.pretrain_model, **deepcopy(unifiedskg_args), model2=unifiedskg.pretrain_model, model2_kwargs=deepcopy(unifiedskg_model_kwargs), **deepcopy(unifiedskg_model_kwargs), gmf_p=1)
    print('Joint decoding: 1 - UnifiedSKG, 2 - UnifiedSKG: p = 1')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out9['sequences'])):
        print(f'{i+1} {out9["sequences_scores"][i].item():.3f} {seq}\n{out9["sequences"][i]}')
    print(64*'-')

    out10 = beam_search(paraphrase, **deepcopy(paraphrase_args), model2=paraphrase, model2_kwargs=deepcopy(paraphrase_model_kwargs), **deepcopy(paraphrase_model_kwargs), gmf_p=1)
    print('Joint decoding: 1 - Paraphrase, 2 - Paraphrase: p = 1')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out10['sequences'])):
        print(f'{i+1} {out10["sequences_scores"][i].item():.3f} {seq}\n{out10["sequences"][i]}')
    print(64*'-')

    out11 = beam_search(unifiedskg.pretrain_model, **deepcopy(unifiedskg_args), model2=paraphrase, model2_kwargs=deepcopy(paraphrase_model_kwargs), **deepcopy(unifiedskg_model_kwargs), gmf_p=0.5)
    print('Joint decoding: 1 - UnifiedSKG, 2 - Paraphrase: p = 0.5')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out11['sequences'])):
        print(f'{i+1} {out11["sequences_scores"][i].item():.3f} {seq}\n{out11["sequences"][i]}')
    print(64*'-')

    out12 = beam_search(paraphrase, **deepcopy(paraphrase_args), model2=unifiedskg.pretrain_model, model2_kwargs=deepcopy(unifiedskg_model_kwargs), **deepcopy(paraphrase_model_kwargs), gmf_p=0.5)
    print('Joint decoding: 1 - Paraphrase, 2 - UnifiedSKG: p = 0.5')
    print('Output:')
    for i, seq in enumerate(tok.batch_decode(out12['sequences'])):
        print(f'{i+1} {out12["sequences_scores"][i].item():.3f} {seq}\n{out12["sequences"][i]}')
    print(64*'-')
