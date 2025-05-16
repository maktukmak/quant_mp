#!/usr/bin/env python3
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import transformers
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from run_exp_llm import (
    ModelArguments,
    QuantizationArguments,
    load_quant_model,
    print_once,
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

from quant_mp.config import rconfig
from quant_mp.utils import patch_model


@dataclass
class EvalArguments:
    model_path: Path = field(  # type: ignore
        default=None,
        metadata={
            "help": "Path to model directory. If not set, it will default to infer from quantization arguments"
        },
    )
    tasks: List[str] = field(
        default_factory=list,
        metadata={
            "nargs": "+",
            "help": "All tasks to evaluate on, as a comma or space separated list",
        },
    )
    output_dir: str = field(
        default="./output/eval",
        metadata={
            "help": "The output directory where the evaluation results are stored.",
        },
    )
    device: str = field(
        default="cuda",
        metadata={
            "help": "Device model will be loaded on. Passed directly to model.to(device)"
        },
    )

    def __post_init__(self):
        if len(self.tasks) == 1:
            self.tasks = self.tasks[0].split(",")


def parse_args() -> Tuple[QuantizationArguments, EvalArguments, ModelArguments]:
    parser = transformers.HfArgumentParser(
        (QuantizationArguments, EvalArguments, ModelArguments),  # type: ignore
    )

    quant_args, eval_args, model_args = parser.parse_args_into_dataclasses()

    if eval_args.model_path is None:
        eval_args.model_path = Path(
            f"./output/{model_args.model_name.split('/')[-1]}/{quant_args.label}/best-model"
        )
    assert eval_args.model_path.exists(), (
        f"Unable to find model at {eval_args.model_path}"
    )

    return quant_args, eval_args, model_args


class QuantizedLLM(HFLM):
    def __init__(
        self,
        model_path: Path,
        model_name: str,
        device: str,
        rconfig: Optional[rconfig] = None,
    ):
        # TODO: Maybe pull config and use transformers.dynamic_module_utils.get_class_from_dynamic_module
        # More proper model patching without loading all pretrained weights first
        if rconfig is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_path, trust_remote_code=True
            )
        else:
            model = load_quant_model(model_path, rconfig)

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model.to(device)

        super().__init__(
            pretrained=model,
            tokenizer=tokenizer,
            backend="causal",
            device="cuda",
            batch_size="auto:2",
            trust_remote_code=True,
            use_fast_tokenizer=False,
            dtype=torch.bfloat16,
        )


def convert_to_json_serializable(obj):
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {
            key: convert_to_json_serializable(value)
            for key, value in obj.items()
            if not key.startswith("_")
        }  # Skip private attributes
    elif hasattr(obj, "__dict__"):
        return convert_to_json_serializable(obj.__dict__)
    else:
        try:
            # Test if value is JSON serializable
            json.dumps(obj)
            return obj
        except (TypeError, OverflowError):
            return str(obj)  # Convert non-serializable objects to strings


def main():
    quant_args, eval_args, model_args = parse_args()

    # Create the output directory if it doesn't exist
    output_dir = Path(eval_args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        output_dir
        / f"{model_args.model_name.split('/')[-1]}_{quant_args.label}_results.json"
    )

    if output_file.exists():
        print_once(
            f"Results already calculated and stored in {output_file}. Skipping this eval run."
        )
        return

    # Initialize the model
    model = QuantizedLLM(
        model_path=eval_args.model_path,
        model_name=model_args.model_name,
        device=eval_args.device,
        rconfig=quant_args.get_rconfig() if quant_args.is_quant else None,
    )

    # Run evaluation
    print_once(f"Evaluating model on tasks: {eval_args.tasks}")
    results = evaluator.simple_evaluate(
        model=model,
        tasks=eval_args.tasks,  # type: ignore
        confirm_run_unsafe_code=True,
    )

    # Return if no results
    if results is None:
        return

    # Save results
    with open(output_file, "w") as f:
        json.dump(convert_to_json_serializable(results), f)
    print_once(f"Results saved to {output_file}")

    print_once("Summary:")
    for task_name, task_results in results["results"].items():
        print_once(f"{task_name}:")
        for metric_name, metric_value in task_results.items():
            print_once(f"  {metric_name}: {metric_value}")


if __name__ == "__main__":
    main()
