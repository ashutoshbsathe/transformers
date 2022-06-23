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
out = unifiedskg.generate(
    **{k: v.cuda() for k, v in tok(sql2text_input, return_tensors='pt').items()},
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    return_dict_in_generate=True,
)
print(64*'-')
print('Standalone UnifiedSKG')
print(f'Input: """{sql2text_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')

# Standalone Paraphrase 
out = paraphrase.generate(
    **{k: v.cuda() for k, v in tok(paraphrase_input, return_tensors='pt').items()},
    num_beams=4,
    num_return_sequences=4,
    max_length=256,
    return_dict_in_generate=True,
)
print(64*'-')
print('Standalone Paraphrase')
print(f'Input: """{paraphrase_input}"""\nOutput:')
for i, seq in enumerate(tok.batch_decode(out['sequences'])):
    print(i+1, seq)
print(64*'-')
