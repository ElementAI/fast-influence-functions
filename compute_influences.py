import argparse
import pickle
import time
from functools import reduce
from pathlib import Path
from typing import Any, Dict, List, Iterable, Optional

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler
from tqdm import tqdm
from transformers.data.data_collator import default_data_collator, DataCollatorWithPadding

from experiments import misc_utils
from influence_utils import nn_influence_utils, faiss_utils
from performance.dask_map import get_client, wait_for_cluster

MODEL_CKPT = "/share/sst2-checkpoint"
RESULTS_FILE_PATH = "/share/sst-2-influences-caching.pkl"
WEIGHT_DECAY = 0.005  # same as used during training
S_TEST_DAMP = 5e-3
S_TEST_SCALE = 1e4
S_TEST_ITERATIONS = 1
BATCH_SIZE_TRAIN_LOADER = 128
NUM_WORKERS = 4
# This depends on the size of the training set and batch_size
# The SST-2 training set has 67,349 examples, so 67449/batch_size=128 = 526
S_TEST_NUM_SAMPLES = 500


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=-1, help="Number of neighbours to use for IF. -1 means no KNN")
    parser.add_argument('--workers', type=int, default=0, help="Number of dask workers to spawn. 0 means no workers")
    parser.add_argument('--faiss_path', type=str, default=None, help="Path to load the faiss index.")
    return parser.parse_args()


def get_train_dataset(tokenizer, indices=None):
    """
    Get a train dataset.
    Args:
        tokenizer: Tokenizer to use in the preprocessing.
        indices: List of indices to select. optional

    Returns:
        a dataset
    """

    def preprocess_function(examples):
        # Do not do padding here, but do it when the dataloader creates batches
        return tokenizer(examples["sentence"],
                         truncation=True,
                         return_token_type_ids=False,
                         return_attention_mask=True)

    data = load_dataset("glue", "sst2")

    # Scatter train_dataset so that everyone has a copy.
    train_dataset = data["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["idx", "sentence"])
    if indices is not None:
        train_dataset = train_dataset.select(indices)
    return train_dataset


def compute_s_test(test_inputs, params_filter, weight_decay_ignores, dataset_len, client, num_workers) -> torch.Tensor:
    assert len(
        test_inputs) == 3 and "input_ids" in test_inputs and "attention_mask" in test_inputs and "labels" in test_inputs

    splitted = np.array_split(np.arange(dataset_len), NUM_WORKERS)
    fut = client.map(compute_s_test_inner, splitted, test_inputs=test_inputs,
                     params_filter=params_filter, weight_decay_ignores=weight_decay_ignores,
                     num_workers=num_workers, pure=False)
    results = client.gather(fut)
    return reduce(update_s_test, results[1:], results[0])


def compute_s_test_inner(indices, test_inputs, params_filter, weight_decay_ignores, num_workers):
    """Function to be sent to the workers, should not be called directly."""
    tokenizer, model = misc_utils.create_tokenizer_and_model(MODEL_CKPT)
    train_dataset = get_train_dataset(tokenizer, indices)
    train_loader = get_train_loader(train_dataset, tokenizer, random=True)
    model.cuda()
    model.eval()

    s_test = None

    for k, v in test_inputs.items():
        test_inputs[k] = v.to(model.device)

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
            num_samples=S_TEST_NUM_SAMPLES // num_workers)

        # Sum the values across runs
        if s_test is None:
            s_test = _s_test
        else:
            s_test = update_s_test(_s_test, s_test)
    # Do the averaging
    s_test = [a / S_TEST_ITERATIONS for a in s_test]
    return [s.cpu() for s in s_test]


def update_s_test(_s_test, s_test):
    return [
        a + b for a, b in zip(s_test, _s_test)
    ]


def get_influences(params_filter, weight_decay_ignores, s_test_example, closest_indices, client, workers):
    """
    Get the influence of the neighbhors on the test example's prediction.
    Args:
        params_filter: Which params to freeze.
        weight_decay_ignores: ...
        s_test_example: s_test for the test example.
        closest_indices: indices of the closest examples.
        client: Dask client connected to the cluster.
        workers: Number of workers in the cluster. TODO Can be infered from the client I think.

    Returns:
        Influence for each indice.
    """

    def mergedict(args):
        output = {}
        for arg in args:
            output.update(arg)
        return output

    splitted = np.array_split(closest_indices, workers)
    fut = client.map(get_influences_inner, splitted, s_test_example=s_test_example,
                     params_filter=params_filter, weight_decay_ignores=weight_decay_ignores, pure=False)
    results = client.gather(fut)
    return mergedict(results)


def get_influences_inner(indices, params_filter, weight_decay_ignores, s_test_example):
    """Function to be sent to the workers. Should not be called directly."""
    tokenizer, model = misc_utils.create_tokenizer_and_model(MODEL_CKPT)
    train_dataset = get_train_dataset(tokenizer, indices)
    train_instance_loader = get_train_loader(train_dataset, tokenizer, random=False)
    model.cuda()
    model.eval()
    s_test_example = [s.cuda() for s in s_test_example]
    influences = {}
    assert len(indices) == len(train_instance_loader), "Needs batch-size=1."
    for index, train_inputs in zip(indices, train_instance_loader):
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


def get_train_loader(train_dataset: 'Dataset', tokenizer: 'Tokenizer', random=False) -> DataLoader:
    """
    Return a DataLoader for the train dataset.
    Args:
        train_dataset: Initial training dataset
        tokenizer: Tokenizer to use on the examples.
        random: Wheter to use a Sequentialsampler of random.

    Returns:
        a DataLoader.
    """
    train_collator = DataCollatorWithPadding(tokenizer, padding=True)
    if random:
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=train_collator,
                                  sampler=RandomSampler(train_dataset),
                                  batch_size=BATCH_SIZE_TRAIN_LOADER)
    else:
        train_loader = DataLoader(dataset=train_dataset,
                                  collate_fn=train_collator,
                                  sampler=SequentialSampler(train_dataset),
                                  batch_size=1)
    return train_loader


def maybe_load_faiss(faiss_path: str) -> faiss_utils.FAISSIndex:
    """
    Will try to load the faiss index if the path is not None.
    Args:
        faiss_path: path to the faiss index.

    Returns:
        Optional[FAISSIndex], a FAISS object.
    """
    if faiss_path is None:
        return faiss_path
    faiss_index = faiss_utils.FAISSIndex(768, "Flat")
    faiss_index.load(faiss_path)
    return faiss_index


def main(args):
    client, cluster = get_client(args.workers, '/scheduler_info')
    wait_for_cluster(cluster, args.workers)

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

    val_seq_sampler = SequentialSampler(val_dataset)
    val_loader = DataLoader(dataset=val_dataset, collate_fn=custom_collator, sampler=val_seq_sampler, batch_size=1)

    # Scatter train_dataset so that everyone has a copy.
    train_dataset = data["train"].map(
        preprocess_function,
        batched=True,
        remove_columns=["idx", "sentence"])
    # The CLS index will be the same throughout
    cls_idx = train_dataset[0]["input_ids"].index(tokenizer.cls_token_id)
    dataset_len = len(train_dataset)

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

    faiss_index = maybe_load_faiss(args.faiss_path)

    # Load results so far
    if Path(RESULTS_FILE_PATH).exists():
        with open(RESULTS_FILE_PATH, "rb") as f:
            results = pickle.load(f)
    else:
        results = dict()

    # Iterate through every example in the validation set
    # Compute influences for every incorrect prediction
    for idx, inputs in tqdm(enumerate(val_loader), desc="Val set", total=len(val_loader)):
        label = inputs.pop("labels")

        for k, v in inputs.items():
            inputs[k] = v.to(model.device)

        out = model(**inputs)
        pred_correct = (out[0].argmax().cpu() == label).numpy()[0]

        # Only get influences for those examples not already processed
        if not pred_correct and idx not in results:
            embedding = model.distilbert(**inputs)[0][:, cls_idx, :].cpu().detach().numpy()
            closest_indices = get_closest(embedding, train_dataset, args.K, faiss_index)
            # Put the label back as we needed for gradient calculation
            inputs["labels"] = label

            print(f"Computing influences for index {idx}...", end=' ')
            s = time.time()
            s_test = compute_s_test(test_inputs=inputs, params_filter=params_filter,
                                    weight_decay_ignores=weight_decay_ignores, dataset_len=dataset_len, client=client,
                                    num_workers=args.workers)
            results[idx] = get_influences(params_filter, weight_decay_ignores, s_test, closest_indices=closest_indices,
                                          client=client, workers=args.workers)
            print(f"Took {time.time() - s} seconds")
            # save results so far
            with open(RESULTS_FILE_PATH, "wb") as f:
                pickle.dump(results, f)

                # TODO We could do nearest neighbor later


def get_closest(embedding: np.ndarray, train_dataset: 'Dataset', num_neighbours: int,
                faiss_index: Optional[faiss_utils.FAISSIndex] = None) -> Iterable[int]:
    """
    Get indices for the `num_neighbours` closest example.

    Args:
        embedding: Array that represents the features.
        train_dataset: Training dataset to search in.
        num_neighbours: Number of neighbours to return, -1 means no KNN used.
        faiss_index: FAISSIndex object to perform KNN.

    Returns:
        List of indices from the train dataset.
    """
    if num_neighbours == -1:
        return np.arange(len(train_dataset))

    assert faiss_index is not None, "If k != -1, faiss_index must be supplied!"
    _, KNN_indices = faiss_index.search(
        k=num_neighbours, queries=embedding)
    return KNN_indices[0]


if __name__ == '__main__':
    args = parse_args()
    main(args)
