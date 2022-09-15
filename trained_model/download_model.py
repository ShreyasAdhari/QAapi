from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_path = 'deepset/bert-base-cased-squad2'

model = AutoModelForQuestionAnswering.from_pretrained( model_path )
tokenizer = AutoTokenizer.from_pretrained( model_path )

model.save_pretrained('./QAS/trained_model/')
tokenizer.save_pretrained('./QAS/trained_model/')