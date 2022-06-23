from transformers import BartTokenizer, BartForConditionalGeneration
import torch
import torch.nn as nn 
import pickle 
import datetime 

model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn").cuda()
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

"""
ARTICLE_TO_SUMMARIZE = (
    "PG&E stated it scheduled the blackouts in response to forecasts for high winds "
    "amid dry conditions. The aim is to reduce the risk of wildfires. Nearly 800 thousand customers were "
    "scheduled to be affected by the shutoffs which were expected to last through at least midday tomorrow."
)
"""

ARTICLE_TO_SUMMARIZE = "Stanford was founded in 1885 by Leland and Jane Stanford in memory of their only child, Leland Stanford Jr., who had died of typhoid fever at age 15 the previous year. Leland Stanford was a U.S. senator and former governor of California who made his fortune as a railroad tycoon. The school admitted its first students on October 1, 1891, as a coeducational and non-denominational institution. Stanford University struggled financially after the death of Leland Stanford in 1893 and again after much of the campus was damaged by the 1906 San Francisco earthquake. Following World War II, provost Frederick Terman supported faculty and graduates' entrepreneurialism to build self-sufficient local industry in what would later be known as Silicon Valley."

pad_token_id = model.config.pad_token_id
eos_token_id = model.config.eos_token_id

num_beams = 8
batch_size = 1

beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device='cuda')
beam_scores[:, 1:] = -1e9 # negative infinity
beam_scores = beam_scores.view((batch_size * num_beams,))

decoder_start_token_id = model.config.decoder_start_token_id

print(pad_token_id, eos_token_id, decoder_start_token_id)

model.eval()

inputs = {k:v.cuda() for k, v in tokenizer([ARTICLE_TO_SUMMARIZE for _ in range(num_beams)], max_length=1024, return_tensors="pt").items()}

decoder_input_ids = torch.ones(num_beams, 1).long().cuda() * decoder_start_token_id

for _ in range(1):
    outputs = model(**inputs, decoder_input_ids=decoder_input_ids)
    print(outputs.logits.size())
    scores = nn.functional.log_softmax(outputs.logits, dim=-1)
    vocab_size = scores.shape[-1]
    scores = scores.view(batch_size, num_beams * vocab_size)

    scores, tokens = torch.topk(
        scores,
        2*num_beams,
        dim=1,
        largest=True,
        sorted=True,
    )
    print(scores)

    indices = torch.div(tokens, vocab_size, rounding_mode='floor')
    tokens = tokens % vocab_size 

    print(indices)
    print(tokens)

