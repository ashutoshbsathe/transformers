from transformers import T5ForConditionalGeneration, AutoTokenizer
import torch

model = T5ForConditionalGeneration.from_pretrained("google/ul2", low_cpu_mem_usage=True, torch_dtype=torch.bfloat16).to("cuda")                                                                                                   
tokenizer = AutoTokenizer.from_pretrained("google/ul2")

input_string = "[S2S] Mr. Dursley was the director of a firm called Grunnings, which made drills. He was a big, solid man with a bald head. Mrs. Dursley was thin and blonde and more than the usual amount of neck, which came in very useful as she spent so much of her time craning over garden fences, spying on the neighbours. The Dursleys had a small son called Dudley and in their opinion there was no finer boy anywhere <extra_id_0>"                                               

inputs = tokenizer(input_string, return_tensors="pt").input_ids.to("cuda")

outputs = model.generate(inputs, max_length=200)

print(tokenizer.decode(outputs[0]))
