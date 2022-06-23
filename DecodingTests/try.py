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

# Paraphrase 
paraphrase = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser').cuda()
paraphrase.eval()

# Inputs 
sql2text_input = 'SELECT COUNT(prerequisite) FROM course WHERE name = "Advanced Machine Learning" ; structed knowledge: '
paraphrase_input = 'paraphrase: What number of prerequisites exist for the course "Advanced Machine Learning"? </s>'

# Standalone UnifiedSKG
out, unifiedskg_args = unifiedskg.generate(
    **{k: v.cuda() for k, v in tok(sql2text_input, return_tensors='pt').items()},
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    return_dict_in_generate=True,
    return_args=True,
)
print(64*'-')
print('Standalone UnifiedSKG [HF generate]')
print(f'Input: """{sql2text_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')

unifiedskg_model_kwargs = unifiedskg_args.pop('model_kwargs')
out = beam_search(unifiedskg.pretrain_model, **unifiedskg_args, **unifiedskg_model_kwargs)
print(64*'-')
print('Standalone UnifiedSKG [Standalone beam_search]')
print(f'Input: """{sql2text_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')


# Standalone Paraphrase 
out, paraphrase_args = paraphrase.generate(
    **{k: v.cuda() for k, v in tok(paraphrase_input, return_tensors='pt').items()},
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    return_dict_in_generate=True,
    return_args=True,
)
print(64*'-')
print('Standalone Paraphrase [HF generate]')
print(f'Input: """{paraphrase_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')

paraphrase_model_kwargs = paraphrase_args.pop('model_kwargs')
out = beam_search(paraphrase, **paraphrase_args, **paraphrase_model_kwargs)
print(64*'-')
print('Standalone Paraphrase [Standalone beam_search]')
print(f'Input: """{paraphrase_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')
