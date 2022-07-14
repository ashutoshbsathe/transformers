# Run as follows 
# `time TRANSFORMERS_OFFLINE=1 PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python joint_mbart50.py`
# This is slow AF because we use python protobuf instead of C++ one but it works reliabily on every system 
import torch
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from new_from_hf import beam_search
from copy import deepcopy 
article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"
article_ar = "الأمين العام للأمم المتحدة يقول إنه لا يوجد حل عسكري في سوريا."

model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)

gen_args = dict(
    forced_bos_token_id=tokenizer.lang_code_to_id["en_XX"],
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    output_scores=True,
    return_dict_in_generate=True,
    return_args=True,
)

with torch.no_grad():

    # Standalone Hindi --> English 
    tokenizer.src_lang = "hi_IN"
    encoded_hi = tokenizer(article_hi, return_tensors="pt")
    out1, hi_args = model.generate(
        **encoded_hi,
        **gen_args,
    )
    print('Standalone Hindi --> English [HF generate]')
    print(f'Input: ""{article_hi}""\nOutput:')
    for i, seq in enumerate(tokenizer.batch_decode(out1['sequences'], skip_special_tokens=True)):
        print(f'{i+1} {out1["sequences_scores"][i].item():.3f} {seq}')
    
    hi_model_kwargs = hi_args.pop('model_kwargs')
    out2 = beam_search(model, **deepcopy(hi_args), **deepcopy(hi_model_kwargs))
    print('Standalone Hindi --> English [Our beam_search]')
    print(f'Input: ""{article_hi}""\nOutput:')
    for i, seq in enumerate(tokenizer.batch_decode(out2['sequences'], skip_special_tokens=True)):
        print(f'{i+1} {out2["sequences_scores"][i].item():.3f} {seq}')

    # Standalone Arabic --> English
    tokenizer.src_lang = "ar_AR"
    encoded_ar = tokenizer(article_ar, return_tensors="pt")
    out3, ar_args = model.generate(
        **encoded_ar,
        **gen_args,
    )
    print('Standalone Arabic --> English [HF generate]')
    print(f'Input: ""{article_ar}""\nOutput:')
    for i, seq in enumerate(tokenizer.batch_decode(out3['sequences'], skip_special_tokens=True)):
        print(f'{i+1} {out3["sequences_scores"][i].item():.3f} {seq}')
    
    ar_model_kwargs = ar_args.pop('model_kwargs')
    out4 = beam_search(model, **deepcopy(ar_args), **deepcopy(ar_model_kwargs))
    print('Standalone Arabic --> English [Our beam_search]')
    print(f'Input: ""{article_ar}""\nOutput:')
    for i, seq in enumerate(tokenizer.batch_decode(out4['sequences'], skip_special_tokens=True)):
        print(f'{i+1} {out4["sequences_scores"][i].item():.3f} {seq}')


