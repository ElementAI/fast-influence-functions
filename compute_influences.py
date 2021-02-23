from typing import Any, Dict, List
from pathlib import Path
import pickle
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding
from datasets import load_dataset

from experiments import misc_utils
from influence_utils import nn_influence_utils

MODEL_CKPT = "/share/sst2-checkpoint"
RESULTS_FILE_PATH = "/share/sst-2-influences-caching.pkl"
WEIGHT_DECAY = 0.005  # same as used during training
S_TEST_DAMP = 5e-3
S_TEST_SCALE = 1e4
S_TEST_ITERATIONS = 1
BATCH_SIZE_TRAIN_LOADER = 128
# This depends on the size of the training set and batch_size
# The SST-2 training set has 67,349 examples, so 67449/batch_size=128 = 526
S_TEST_NUM_SAMPLES = 500


def compute_s_test(test_inputs, model, train_loader, params_filter, weight_decay_ignores) -> torch.Tensor:
    assert len(test_inputs) == 3 and "input_ids" in test_inputs and "attention_mask" in test_inputs and "labels" in test_inputs
    
    for k, v in test_inputs.items():
        test_inputs[k] = v.to(model.device)
        
    s_test = None

    for _ in range(S_TEST_ITERATIONS):
        _s_test = nn_influence_utils.compute_s_test(
            n_gpu=1,
            device=model.device,
            model=model,
            test_inputs=test_inputs,
            train_data_loaders=[train_loader],
            params_filter=params_filter,
            weight_decay_ignores=weight_decay_ignores,
            weight_decay=WEIGHT_DECAY,
            damp=S_TEST_DAMP,
            scale=S_TEST_SCALE,
            num_samples=S_TEST_NUM_SAMPLES)

        # Sum the values across runs
        if s_test is None:
            s_test = _s_test
        else:
            s_test = [
                a + b for a, b in zip(s_test, _s_test)
            ]
    # Do the averaging
    s_test = [a / S_TEST_ITERATIONS for a in s_test]
    return s_test


def get_influences(train_instance_loader, model, params_filter, weight_decay_ignores, s_test_example):

    influences = {}

    for index, train_inputs in enumerate(tqdm(train_instance_loader)):

        grad_z = nn_influence_utils.compute_gradients(
            n_gpu=1,
            device=model.device,
            model=model,
            inputs=train_inputs,
            params_filter=params_filter,
            weight_decay=WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores)

        with torch.no_grad():
            influence = [
                - torch.sum(x * y)
                for x, y in zip(grad_z, s_test_example)]

        influences[index] = sum(influence).item()
    
    return influences


def custom_collator(features: List[Any]) -> Dict[str, torch.Tensor]:
    """Remove unncessary items from features before passing input to model."""
    assert len(features) == 1, "Only handle batch_size == 1"
    input_dict = default_data_collator(features)
    input_dict.pop("sentence", None)
    # Keep the labels 
    input_dict.pop("idx", None)
    return input_dict


if __name__ == "__main__":

    tokenizer, model = misc_utils.create_tokenizer_and_model(MODEL_CKPT)

    def preprocess_function(examples):
        # Do not do padding here, but do it when the dataloader creates batches
        return tokenizer(examples["sentence"],
            truncation=True,
            return_token_type_ids=False,
            return_attention_mask=True)

    # Load and pre-process data
    data = load_dataset("glue", "sst2")
    val_dataset = data["validation"].map(
        preprocess_function,
        batched=True)

    train_dataset = data["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["idx", "sentence"]) # keep the label in there for gradient calculation

    val_seq_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=custom_collator, sampler=val_seq_sampler, batch_size=1)

    # Need to pad the training batch
    train_collator = DataCollatorWithPadding(tokenizer, padding=True)
    train_random_loader = DataLoader(dataset=train_dataset,
                                    collate_fn=train_collator,
                                    sampler=RandomSampler(train_dataset),
                                    batch_size=BATCH_SIZE_TRAIN_LOADER)
    train_instance_loader = DataLoader(dataset=train_dataset, 
                                   collate_fn=train_collator, 
                                   sampler=SequentialSampler(train_dataset),
                                   batch_size=1)

    # Setup needed parameters
    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
        "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    model.cuda()
    model.eval()

    # Load results so far
    if Path(RESULTS_FILE_PATH).exists():
        with open(RESULTS_FILE_PATH, "rb") as f:
            results = pickle.load(f)
    else:
        results = dict()

    # Iterate through every example in the validation set
    # Compute influences for every incorrect prediction
    for idx, inputs in enumerate(val_loader):
        label = inputs.pop("labels")

        for k, v in inputs.items():        
            inputs[k] = v.to(model.device)
    
        out = model(**inputs)
        pred_correct = (out[0].argmax().cpu() == label).numpy()[0]
        
        # Only get influences for those examples not already processed
        if not pred_correct and idx not in results:
            # Put the label back as we needed for gradient calculation
            inputs["labels"] = label

            print(f"Computing influences for index {idx}")
            s_test = compute_s_test(test_inputs=inputs, model=model, train_loader=train_random_loader, params_filter=params_filter, weight_decay_ignores=weight_decay_ignores)
            #print(s_test.shape)
           
            results[idx] = get_influences(train_instance_loader, model, params_filter, weight_decay_ignores, s_test)
        
            # save results so far
            with open(RESULTS_FILE_PATH, "wb") as f:
                pickle.dump(results, f)   

    #TODO We could do nearest neighbor later
