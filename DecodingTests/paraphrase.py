from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModel 

#tokenizer = T5Tokenizer.from_pretrained('ramsrigouthamg/t5_paraphraser')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
paraphrase = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
sql2text = AutoModel.from_pretrained('hkunlp/from_all_T5_base_prefix_sql2text2')


paraphrase_text_input = 'paraphrase: What number of prerequisites exist for the course "Advanced Machine Learning"? </s>'
sql2text_text_input = 'SELECT COUNT ( prerequisite ) FROM course WHERE name = "Advanced Machine Learning" ; structed knowledge:'

"""
outputs = paraphrase.generate(**tokenizer(paraphrase_text_input, max_length=512, pad_to_max_length=True, return_tensors='pt'), num_beams=4, return_dict_in_generate=True, num_return_sequences=4, do_sample=True, top_k=120, top_p=0.5, max_length=256)
print(paraphrase_text_input)
for seq in tokenizer.batch_decode(outputs['sequences']):
    print(seq)
"""

outputs = sql2text.generate(**tokenizer([sql2text_text_input], max_length=512, pad_to_max_length=True, return_tensors='pt'), num_beams=4, return_dict_in_generate=True, num_return_sequences=4, do_sample=True, top_k=120, top_p=0.5, max_length=256)
print(sql2text_text_input)
for seq in tokenizer.batch_decode(outputs['sequences']):
    print(seq)
