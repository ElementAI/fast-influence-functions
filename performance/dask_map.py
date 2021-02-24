import os
import time
from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from datasets import load_dataset
from distributed import LocalCluster, Client, wait
from eai_distributed_toolkit.dask import EAIDaskCluster
from eai_distributed_toolkit.toolkit_utils import get_docker_image, fetch_datas
from tqdm import tqdm
from transformers import DistilBertForSequenceClassification, AutoTokenizer

from influence_utils.parallel import InfluenceHelper

SHR_SST_CHECKPOINT = "/xai_caps_shr/sst2-checkpoint"
WORKERS = 4


def is_process_agent():
    return os.environ.get("EAI_IS_PROCESS_AGENT", False)


def get_client(num_workers, artifact_folder) -> Tuple[Client, "Cluster"]:
    if is_process_agent():
        cluster = EAIDaskCluster(image=get_docker_image(), memory_gib=8, cores=4, gpus=1,
                                 datas=fetch_datas()['data'], artifact_folder=artifact_folder)
    else:
        cluster = LocalCluster()
    cluster.scale(num_workers)
    client = Client(cluster.scheduler_address)
    return client, cluster


def load_model():
    model = DistilBertForSequenceClassification.from_pretrained(SHR_SST_CHECKPOINT)
    params_to_freeze = [
        "distilbert.embeddings.",
        "distilbert.transformer.layer.0.",
        "distilbert.transformer.layer.1.",
        "distilbert.transformer.layer.2.",
        "distilbert.transformer.layer.3.",
    ]
    for name, param in model.named_parameters():
        if any(pfreeze in name for pfreeze in params_to_freeze):
            param.requires_grad = False
    return model


def tokenize(d, tokenizer):
    out = tokenizer(
        d['sentence'],
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        return_attention_mask=True,
        return_tensors='pt',
        truncation=True,
    )
    out['labels'] = torch.LongTensor(d['label'])
    d.pop('idx')
    print(out)
    return out


def get_inputs(indices):
    tokenizer = AutoTokenizer.from_pretrained(SHR_SST_CHECKPOINT, use_fast=False)
    datasets = load_dataset("glue", 'sst2')['train']
    return [tokenize(datasets[i:i + 1], tokenizer) for i in indices]


def compute_influences(train_indices, s_test, use_cuda):
    model = load_model()
    if use_cuda:
        model = model.cuda()
        s_test = [s.cuda() for s in s_test]
    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = ["bias",
                            "LayerNorm.weight"] + [
                               n for n, p in model.named_parameters()
                               if not p.requires_grad]
    helper = InfluenceHelper('list', n_gpu=torch.cuda.device_count(), model=model,
                             progress_bar=True, params_filter=params_filter, weight_decay=0.005,
                             weight_decay_ignores=weight_decay_ignores)
    """
    Xs: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
    s_test: List[torch.FloatTensor]
    """
    Xs = get_inputs(train_indices)
    if use_cuda:
        Xs = [{k: v.cuda() for k, v in x.items()} for x in Xs]
    influences = helper(Xs=Xs, s_test=s_test).detach().cpu().numpy()
    return influences

def main(artifact_path):
    client, cluster = get_client(WORKERS, artifact_path)

    wait_for_cluster(cluster, WORKERS)
    indices = np.arange(1000)
    s_test = [torch.randn(768, 768)]
    report = defaultdict(list)
    for _ in tqdm(range(3)):
        for n_job in tqdm([1, 2, 3, 4]):
            splitted = np.array_split(indices, n_job)
            s = time.time()
            fut = client.map(compute_influences, splitted, s_test=s_test, use_cuda=True, pure=False)
            wait(fut)
            out = fut[0].result()
            timing = time.time() - s
            report[n_job].append(timing)
            print(f"{n_job} split: Took  {timing}")

    for n_jobs, data in report.items():
        print(f"{n_jobs} Jobs: {np.mean(data):.4f}±{np.std(data):.4f}")


def wait_for_cluster(cluster, num_expected):
    if is_process_agent():
        # TODO Rework this loop
        num_workers = len(cluster._list_workers().running)
        while num_workers < num_expected:
            print(f"Waiting for cluster to start: {num_workers} online.")
            time.sleep(5)
            num_workers = len(cluster._list_workers().running)


if __name__ == '__main__':
    main('/scheduler_info')
