from typing import Any, List, Dict
import torch
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.dataloader import DataLoader
from transformers.data.datasets.glue import GlueDataset, GlueDataTrainingArguments
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from tqdm import tqdm

from datasets import load_dataset
from influence_utils import faiss_utils, glue_utils


MODEL_PATH = "/share/sst2-checkpoint"
FAISS_PATH = "/share/sst2-faiss_index"
tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
glue_utils.freeze_BERT_parameters(model, verbose=False)

num_trainable_params = sum([
    p.numel() for n, p in model.named_parameters()
    if p.requires_grad])

print(f"# trainable parameters is {num_trainable_params}")

def preprocess_function(examples):
    # Do not do padding here, but do it when the dataloader creates batches
    return tokenizer(examples["sentence"],
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True)

data = load_dataset("glue", "sst2")
dataset = data["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=data["train"].column_names)

batch_size = 128
# We want to insert values in the index sequentially
sampler = SequentialSampler(dataset)
# The collator will pad the batches
collator = DataCollatorWithPadding(tokenizer, padding=True)
loader = DataLoader(dataset=dataset, collate_fn=collator, sampler=sampler, batch_size=batch_size)

model.eval()
model.cuda()

# The CLS index will be the same throughout
cls_idx = dataset[0]["input_ids"].index(tokenizer.cls_token_id)

faiss_index = faiss_utils.FAISSIndex(768, "Flat")

for inputs in tqdm(loader):
    for k, v in inputs.items():
        inputs[k] = v.to(model.device)
        
    out = model.distilbert(**inputs)
    features = out[0][:, cls_idx, :].cpu().detach().numpy()
    faiss_index.add(features)

print(f"Built FAISS index with {faiss_index._index.ntotal} elements")
assert faiss_index._index.ntotal == len(data["train"])
faiss_index.save(FAISS_PATH)
print(f"Saved FAISS index at {FAISS_PATH}")
