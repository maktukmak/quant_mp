import json
import math
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from safetensors.torch import load_file
import torch
import torch.distributed as dist
import transformers
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    default_data_collator,
)
from transformers.dynamic_module_utils import get_class_from_dynamic_module
from transformers.models.auto.auto_factory import _get_model_class

from quant_mp.config import qconfig, rconfig
from quant_mp.utils import patch_model


@dataclass
class QuantizationArguments:
    label: str = field(
        default=None,
        metadata={
            "help": "Label name for the quantion. Defaults to {activation_qtype}-{activation_format}-{activation_alg}-{weight_qtype}-{weight_format}-{weight_alg}"
        },
    )  # type: ignore
    activation_qtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["float", "uniform", "nonuniform"],
            "help": "Quantization type for activations.",
        },
    )
    activation_qbits: Optional[int] = field(
        default=None,
        metadata={
            "help": "Quantization bits for activations.",
        },
    )
    activation_format: Optional[str] = field(
        default=None,
        metadata={"help": "Floating point format (e2m1, e3m0, etc.) for activations."},
    )
    activation_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["minmax", "normal", "iterative", "lsq"],
            "help": "Quantization algorithm for activations.",
        },
    )
    weight_qtype: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["float", "uniform", "nonuniform"],
            "help": "Quantization type for weights.",
        },
    )
    weight_qbits: Optional[int] = field(
        default=None,
        metadata={
            "help": "Quantization bits for activations.",
        },
    )
    weight_format: Optional[str] = field(
        default=None,
        metadata={"help": "Floating point format (e2m1, e3m0, etc.) for activations."},
    )
    weight_block_size: Optional[int | str] = field(
        default=None, metadata={"help": "Block size in integer blocks or 'channel'."}
    )
    weight_alg: Optional[str] = field(
        default=None,
        metadata={
            "choices": ["minmax", "normal", "iterative", "lsq"],
            "help": "Quantization algorithm for activations.",
        },
    )

    def __post_init__(self):
        self.label = (
            self.label
            or f"A-{self.activation_qtype}-{self.activation_format}-{self.activation_alg}--W-{self.weight_qtype}-{self.weight_format}-{self.weight_alg}-{self.weight_block_size}"
        )
        if self.activation_qtype is not None:
            assert self.activation_qbits is not None, (
                "Activation qtype set but no qbits set."
            )
            assert self.activation_alg is not None, (
                "Activation qtype set but no alg set."
            )
            if self.activation_qtype == "float":
                assert self.activation_format is not None, (
                    "Activation qtype set but no format set."
                )

    @property
    def activation_qconfig(self):
        if self.activation_qtype is None:
            return None
        return qconfig(
            qtype=self.activation_qtype,
            qbits=self.activation_qbits,
            alg=self.activation_alg,
            format=self.activation_format or "",
        )

    @property
    def weight_qconfig(self):
        if self.weight_qtype is None:
            return None
        assert self.weight_alg is not None, "Weight qtype set but no alg set."
        if self.weight_qtype == "float":
            assert self.weight_format is not None, "Weight qtype set but no format set."
        assert self.weight_qbits is not None, "Weight qtype set but no qbits set."
        return qconfig(
            qtype=self.weight_qtype,
            qbits=self.weight_qbits,
            alg=self.weight_alg,
            format=self.weight_format or "",
            qblock_size=self.weight_block_size,
        )

    @property
    def is_quant(self):
        return self.activation_qtype is not None or self.weight_qtype is not None

    def get_rconfig(self):
        return rconfig(
            label=self.label,
            activation=self.activation_qconfig or qconfig(qtype=None),
            weight=self.weight_qconfig or qconfig(qtype=None),
            grad=qconfig(qtype=None),
        )


@dataclass
class ModelArguments:
    model_name: str = field(
        default="facebook/MobileLLM-125M",
        metadata={"help": "Model name or path, loaded by AutoModelForCausalLM"},
    )
    tokenizer_name: str = field(
        default=None,
        metadata={
            "help": "Tokenizer name or path, loaded by AutoTokenizer. Defaults to model_name."
        },
    )  # type: ignore
    output_model_path: str = field(
        default=None,
        metadata={
            "help": "Path to save the fine-tuned model. If not set, the model will not be saved."
        },
    )  # type: ignore

    def __post_init__(self):
        self.tokenizer_name = self.tokenizer_name or self.model_name


@dataclass
class DataArguments:
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max training samples in number of lines. Used for debugging on smaller training set"
        },
    )
    max_valid_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "Max validation samples in number of lines. Used for debugging on smaller validation set"
        },
    )
    train_ds_path: str = field(
        default="./train.jsonl", metadata={"help": "Path to training dataset"}
    )
    valid_ds_path: Optional[str] = field(
        default=None, metadata={"help": "Path to validation dataset"}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    optim: Optional[str] = field(default="adamw_torch")
    output_dir: Optional[str] = field(default="/tmp/output/")
    model_max_length: int = field(
        default=2048,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated). 512 or 1024"
        },
    )
    qat: Optional[bool] = field(default=False)


def parse_args() -> Tuple[
    QuantizationArguments, ModelArguments, TrainingArguments, DataArguments
]:
    parser = transformers.HfArgumentParser(
        (QuantizationArguments, ModelArguments, TrainingArguments, DataArguments)  # type: ignore
    )
    quant_args, model_args, training_args, data_args = (
        parser.parse_args_into_dataclasses()
    )

    training_args.output_dir = (
        f"./output/{model_args.model_name.split('/')[-1]}/{quant_args.label}"
    )
    print_once(f"Quant Args: {quant_args}")
    print_once(f"Model Args: {model_args}")
    print_once(f"Data Args: {data_args}")
    print_once(f"Training Args:\n{training_args}")
    return quant_args, model_args, training_args, data_args


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, block_size=1024):
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        tokenized_datasets = []
        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]
        self.data = [
            dict(input_ids=self.input_ids[i], labels=self.labels[i])
            for i in range(len(self.input_ids))
        ]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples["text"])

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d.keys():
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [
                t[i : i + self.block_size]
                for i in range(0, total_length, self.block_size)
            ]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result


def read_jsonl_dataset(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def print_once(*args, **kwargs):
    if os.environ["LOCAL_RANK"] == "0":
        print(*args, **kwargs)


def load_quant_model(quant_model_path: str | Path, rconfig: rconfig):
    config = AutoConfig.from_pretrained(quant_model_path, trust_remote_code=True)
    if hasattr(config, "auto_map"):
        model_cls = get_class_from_dynamic_module(
            config.auto_map["AutoModelForCausalLM"], quant_model_path
        )
    elif type(config) in AutoModelForCausalLM._model_mapping.keys():
        model_cls = _get_model_class(config, AutoModelForCausalLM._model_mapping)
    else:
        raise RuntimeError(f"Could not find model class for {quant_model_path}")
    model = model_cls(config)
    patch_model(model, rconfig)
    state_dict = {}
    for state_dict_path in Path(quant_model_path).glob("*.safetensors"):
        state_dict.update(load_file(state_dict_path))
    model.load_state_dict(state_dict, strict=False)
    return model


def main(
    model_args: ModelArguments,
    training_args: TrainingArguments,
    quant_args: QuantizationArguments,
    data_args: DataArguments,
):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name, trust_remote_code=True
    )
    if quant_args.is_quant:
        quant_config = quant_args.get_rconfig()
        print_once(f"Patching model with quant config: {quant_config}")
        patch_model(model, quant_config)
    tokenizer = AutoTokenizer.from_pretrained(model_args.tokenizer_name, use_fast=False)

    train_data = read_jsonl_dataset(data_args.train_ds_path)[
        : data_args.max_train_samples
    ]
    train_ds = CustomJsonDataset(
        train_data, tokenizer, block_size=training_args.model_max_length
    )
    if training_args.do_train and quant_args.activation_alg == "lsq":
        # LSQ initialization
        with torch.no_grad():
            model(torch.tensor(train_ds[0]["input_ids"]).unsqueeze(0))

    valid_ds = None
    if data_args.valid_ds_path is not None:
        valid_data = read_jsonl_dataset(data_args.valid_ds_path)[
            : data_args.max_valid_samples
        ]
        valid_ds = CustomJsonDataset(
            valid_data, tokenizer, block_size=min(training_args.model_max_length, 1024)
        )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_ds if training_args.do_train else None,
        eval_dataset=valid_ds if training_args.do_eval else None,
        data_collator=default_data_collator,
    )

    if training_args.do_train:
        output_path = f"{training_args.output_dir}/best-model"
        if not os.path.exists(output_path):
            train_result = trainer.train()
            trainer.save_state()
            trainer.save_model(output_path)
        else:
            print_once(f"Model found at {output_path}")
            if quant_args.is_quant:
                trainer.model = load_quant_model(
                    output_path,
                    quant_config,  # type: ignore
                )
            else:
                trainer.model = AutoModelForCausalLM.from_pretrained(
                    output_path, trust_remote_code=True
                )

    if training_args.do_eval and valid_ds is not None:
        metrics = trainer.evaluate()
        max_eval_samples = len(valid_ds)
        metrics["eval_samples"] = min(max_eval_samples, len(valid_ds))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    quant_args, model_args, training_args, data_args = parse_args()
    try:
        main(model_args, training_args, quant_args, data_args)
    finally:
        dist.destroy_process_group()
