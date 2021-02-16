import torch
import sys


from experiments import misc_utils, constants
from experiments.data_utils import (
    glue_output_modes,
    glue_compute_metrics)

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertTokenizer,
    BertForSequenceClassification,
    GlueDataTrainingArguments,
    Trainer,
    DataCollator,
    default_data_collator)

from influence_utils import glue_utils, parallel
from experiments.data_utils import CustomGlueDataset
from influence_utils import nn_influence_utils

if __name__ == '__main__':

    tokenizer, model = misc_utils.create_tokenizer_and_model("/home/orlandom/checkpoints/fast-if")
    print ('Available devices ', torch.cuda.device_count())

    data_args = GlueDataTrainingArguments(
        task_name="mnli",
        data_dir="/home/orlandom/data/MNLI",
        max_seq_length=128)

    train_dataset = CustomGlueDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="train")

    eval_dataset = CustomGlueDataset(
        args=data_args,
        tokenizer=tokenizer,
        mode="dev")

    batch_train_data_loader = misc_utils.get_dataloader(
        train_dataset,
        batch_size=128,
        random=True)

    instance_train_data_loader = misc_utils.get_dataloader(
        train_dataset,
        batch_size=1,
        random=False)

    eval_instance_data_loader = misc_utils.get_dataloader(
        dataset=eval_dataset,
        batch_size=1,
        random=False)

    output_mode = glue_output_modes["mnli"]

    s_test_num_samples = 100
    test_inputs = eval_instance_data_loader.dataset[0]

    params_filter = [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    weight_decay_ignores = [
        "bias",
       "LayerNorm.weight"] + [
        n for n, p in model.named_parameters()
        if not p.requires_grad]

    for test_index, test_inputs in enumerate(eval_instance_data_loader):
        out = parallel.compute_influences_parallel(
            device_ids=[0, 1, 2, 3],
            train_dataset=train_dataset,
            batch_size=1,
            model=model,
            test_inputs=test_inputs,
            params_filter=params_filter,
            weight_decay=constants.WEIGHT_DECAY,
            weight_decay_ignores=weight_decay_ignores,
            s_test_damp=5e-3,
            s_test_scale=1e4,
            s_test_num_samples=s_test_num_samples,
            debug= False,
            return_s_test=False,
            train_indices_to_include=None)

        print(out.shape)
        break
