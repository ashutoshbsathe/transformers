from transformers import BartTokenizer, BartForConditionalGeneration
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

inputs = {k:v.cuda() for k, v in tokenizer([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors="pt").items()}
print(inputs)
# Generate Summary
outputs, debug_info = model.generate(inputs["input_ids"], num_beams=8, return_dict_in_generate=True, num_return_sequences=8, min_length=0, max_length=48)
summary_ids = outputs['sequences']
for sequence in tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
    print(sequence)
for sequence in tokenizer.batch_decode(summary_ids):
    print(sequence)
"""
debug_info['article'] = ARTICLE_TO_SUMMARIZE
debug_info['article_to_inputs'] = inputs

stamp = datetime.datetime.now().strftime('%d-%m-%Y_%H%M%S')

with open(f'{stamp}_debug.pkl', 'wb') as f:
    pickle.dump(debug_info, f)
"""
