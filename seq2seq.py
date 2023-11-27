from transformers import (
    PreTrainedTokenizerBase,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

import argparse
import math
import os
from dataset import get_dataloader, get_task_tags


from load_model import load_model

from evaluate import (
    evaluate_most_probable,
)


from typing import List
import json

from tqdm.auto import tqdm


from torch.utils.data import DataLoader
import torch
import torch.nn as nn

from accelerate import Accelerator

from transformers import (
    get_scheduler,
    set_seed,
    PreTrainedModel,
)

from torch.optim import AdamW


import wandb


from constrained_generation import constrained_beam_search, unconstrained_beam_search


def gen_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def experiment_done(experiment_dir: str, test_tsvs: List[str]):
    for test_tsv in test_tsvs:
        test_name = os.path.splitext(os.path.basename(test_tsv))[0]
        if not os.path.exists(os.path.join(experiment_dir, f"{test_name}.txt")):
            return False
    return True


def parse_args():
    parser = argparse.ArgumentParser(
        description="Finetune a transformers model on a text classification task"
    )

    parser.add_argument(
        "--train_tsvs",
        nargs="+",
        type=str,
        default=None,
        help="A tsv file in conll format containing the sl training data.",
    )

    parser.add_argument(
        "--dev_tsvs",
        nargs="+",
        type=str,
        default=None,
        help="A tsv file in conll format containing the sl training data.",
    )

    parser.add_argument(
        "--test_tsvs",
        nargs="+",
        type=str,
        default=None,
        help="A tsv file in conll format containing the sl training data.",
    )

    parser.add_argument(
        "--num_beams",
        type=int,
        default=4,
        help="Number of beams to use for evaluation. This argument will be "
        "passed to ``model.generate``, which is used during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of sequences to return. This argument will be "
        "passed to ``model.generate``, which is used during ``predict``.",
    )

    parser.add_argument(
        "--max_source_length",
        type=int,
        default=256,
        help="The maximum total input sequence length after "
        "tokenization.Sequences longer than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--max_target_length",
        type=int,
        default=256,
        help="The maximum total sequence length for target text after "
        "tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."
        "during ``evaluate`` and ``predict``.",
    )

    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
        required=True,
    )

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=8,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=64,
        help="Starting batch size (per device) for evaluation batch size finder. We will start with batch and "
        "reduce it until the batch fits in memory.",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.0, help="Weight decay to use."
    )

    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Total number of training epochs to perform.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--optim",
        type=str,
        default="adafactor",
        help="The optimizer to use. Adafactor is recommended for training T5, mT5 and FLAN models. "
        "AdamW is recommended for LoRA and decoder-only models. Adafactor requires fairseq, you can install it with "
        "pip install fairseq.",
        choices=["adamw", "adamw8bits", "adafactor", "deepspeed"],
    )

    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear",
            "cosine",
            "cosine_with_restarts",
            "polynomial",
            "constant",
            "constant_with_warmup",
        ],
    )

    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=500,
        help="Number of steps for the warmup in the lr scheduler.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Where to store the final model."
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for reproducible training."
    )

    parser.add_argument(
        "--eval_every_epochs",
        type=int,
        default=1,
        help="We will evaluate every X epochs. Set this to 0 to disable evaluation.",
    )

    parser.add_argument(
        "--eval_every_steps",
        type=int,
        default=0,
        help="We will evaluate every X steps. Set this to 0 to disable evaluation.",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="SeqLabeling_w_LLMs",
        help="The project name to use for wandb.",
    )

    parser.add_argument(
        "--use_lora",
        action="store_true",
        help="Use LoRA for efficient training. We will convert the model to 8-bit and use LoRA to train it. "
        "You should be able to train large models in consumer-grade GPUs with this option.",
    )

    parser.add_argument(
        "--lora_r",
        type=int,
        default=16,
        help="The r parameter for LoRA. This is the number of bits to quantize the weights to.",
    )

    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="The alpha parameter for LoRA. This is the learning rate multiplier for the quantized weights.",
    )

    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="The dropout probability for LoRA. This is the probability of dropping a weight during training.",
    )

    parser.add_argument(
        "--lora_target_modules",
        type=str,
        nargs="+",
        default=["all"],
        help="The modules to apply LoRA to. This is a comma-separated list of module names. "
        "If not specified we will add LoRA to all the compatible layers.",
    )

    parser.add_argument(
        "--constrained_generation",
        action="store_true",
        help="Use constrained generation. ",
    )

    parser.add_argument(
        "--unconstrained_generation",
        action="store_true",
        help="Use unconstrained generation.",
    )

    parser.add_argument(
        "--use_flash_attention",
        action="store_true",
        help="Weather to use flash attention ",
    )

    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Weather to use flash attention ",
    )

    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["fp16", "bf16", "no"],
        help="Whether to use mixed precision or not. Models such as mT5 are trained with bf16, if you set fp16 "
        "the loss will probably end up being NaN due to conversion issues. Check how the model was trained "
        "and set this flag accordingly.",
    )

    parser.add_argument(
        "--quantization",
        type=int,
        default=None,
        help="Whether to use '4' or '8' bit quantization. "
        "Requires bitsandbytes library: https://github.com/TimDettmers/bitsandbytes",
    )

    parser.add_argument(
        "--force_auto_device_map",
        type=int,
        default=None,
        help="Whether to force the use of the auto device map. If set to True, the model will be split across "
        "GPUs and CPU to fit the model in memory. If set to False, a full copy of the model will be loaded "
        "into each GPU. Defaults to False.",
    )

    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )

    parser.add_argument(
        "--add_labels_as_tokens",
        action="store_true",
        help="Add the labels as tokens to the tokenizer",
    )

    parser.add_argument(
        "--add_labels_as_prompt",
        action="store_true",
        help="We will append the labels of the task at the start of the input sentence, usefull for multi-task",
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="The prompt to use for the task, "
        "this is a text that will be appended to the start of the input sentence. "
        "Useful for zero-shot inference.",
    )

    parser.add_argument(
        "--source_lang",
        type=str,
        default=None,
        help="The source language, this is useful if you want to use a machine translation model such as"
        "m2m100 or nllb200. If set to None, we will ignore this parameter.",
    )

    parser.add_argument(
        "--target_lang",
        type=str,
        default=None,
        help="The target language, this is useful if you want to use a machine translation model such as"
        "m2m100 or nllb200. If set to None, we will ignore this parameter.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.train_tsvs is not None and args.dev_tsvs is None:
        raise ValueError("You must specify a dev set if you specify a train set.")

    if not args.constrained_generation and not args.unconstrained_generation:
        raise ValueError(
            "You must specify either constrained_generation or unconstrained_generation."
        )

    return args


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"\n---> Trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}\n"
    )

    return trainable_params, all_param, 100 * trainable_params / all_param


def evaluate(
    dataloader: DataLoader,
    constrained_generation: bool,
    accelerator: Accelerator,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    num_beams: int,
    num_return_sequences: int,
    output_dir: str,
    stage: str = "dev",
    epoch: int = -1,
    train_step: int = -1,
    forced_bos_token: int = None,
):
    print(f"***** Evaluating {dataloader.dataset.file_path} *****")
    if epoch != -1:
        print(f"  Epoch = {epoch}")
        print(f"  Train step = {train_step}")
    print(f"  Num examples = {len(dataloader.dataset)}")
    print(
        f"  Gen kwargs = "
        f"{{'constrained_generation' : {constrained_generation}, "
        f"'num_return_sequences': {num_return_sequences}, "
        f"'num_beams': {num_beams}, "
        f"'max_length': {max_length}}}"
    )
    print()
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    model_outputs_txt: List[List[str]] = []
    gold_txt: List[str] = []
    original_txt: List[str] = []
    model_inputs_txt: List[str] = []
    samples_seen: int = 0

    test_name = os.path.splitext(os.path.basename(dataloader.dataset.file_path))[0]
    if stage == "dev":
        filename = f"{test_name}_epoch_{epoch}_step_{train_step}_{'constrained' if constrained_generation else 'unconstrained'}"
    else:
        filename = f"{test_name}_{'constrained' if constrained_generation else 'unconstrained'}"

    with open(os.path.join(output_dir, f"{filename}.jsonl"), "w", encoding="utf8") as f:
        for step, batch in enumerate(
            tqdm(
                dataloader,
                disable=not accelerator.is_local_main_process,
                ascii=True,
                desc=f"{os.path.splitext(os.path.basename(dataloader.dataset.file_path))[0]}",
            )
        ):
            if constrained_generation:
                generated_tokens = constrained_beam_search(
                    model_inputs=batch,
                    model=accelerator.unwrap_model(model),
                    start_labels_ids=dataloader.dataset.start_labels_ids,
                    end_labels_ids=dataloader.dataset.end_labels_ids,
                    start_labels_names=list(
                        range(len(dataloader.dataset.start_labels_ids))
                    ),
                    end_labels_names=list(
                        range(len(dataloader.dataset.end_labels_ids))
                    ),
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    forced_bos_token_id=forced_bos_token,
                )
            else:
                generated_tokens = unconstrained_beam_search(
                    model_inputs=batch,
                    model=accelerator.unwrap_model(model),
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    forced_bos_token_id=forced_bos_token,
                )

            input_tokens = (
                accelerator.gather(
                    accelerator.pad_across_processes(
                        batch.input_ids,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                )
                .cpu()
                .tolist()
            )

            generated_tokens = (
                accelerator.gather(
                    accelerator.pad_across_processes(
                        generated_tokens,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                )
                .cpu()
                .tolist()
            )

            original_sentences = (
                accelerator.gather(
                    accelerator.pad_across_processes(
                        batch.original_sentence_ids,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                )
                .cpu()
                .tolist()
            )

            gold_tokens = (
                accelerator.gather(
                    accelerator.pad_across_processes(
                        batch.labeled_sentence_ids,
                        dim=1,
                        pad_index=tokenizer.pad_token_id,
                    )
                )
                .cpu()
                .tolist()
            )

            if accelerator.is_local_main_process:
                if accelerator.num_processes > 1:
                    # Remove duplicated in last batch if we are in a distributed setting
                    if step == len(dataloader) - 1:
                        generated_tokens = generated_tokens[
                            : (len(dataloader.dataset) - samples_seen)
                            * num_return_sequences
                        ]
                        gold_tokens = gold_tokens[
                            : (len(dataloader.dataset) - samples_seen)
                        ]
                        original_sentences = original_sentences[
                            : (len(dataloader.dataset) - samples_seen)
                        ]
                        input_tokens = input_tokens[
                            : (len(dataloader.dataset) - samples_seen)
                        ]
                    else:
                        samples_seen += len(batch)

                generated_tokens = list(
                    gen_batch(
                        tokenizer.batch_decode(
                            generated_tokens,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ),
                        n=num_return_sequences,
                    )
                )

                gold_tokens = tokenizer.batch_decode(
                    gold_tokens,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                original_sentences = tokenizer.batch_decode(
                    original_sentences,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )

                input_tokens = tokenizer.batch_decode(
                    input_tokens,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )

                model_outputs_txt.extend(generated_tokens)
                gold_txt.extend(gold_tokens)
                original_txt.extend(original_sentences)
                model_inputs_txt.extend(input_tokens)

                for prediction, gold, orig, model_input_txt in zip(
                    generated_tokens, gold_tokens, original_sentences, model_inputs_txt
                ):
                    print(
                        json.dumps(
                            {
                                "model_input": model_input_txt,
                                "input_sentence": orig,
                                "prediction": prediction,
                                "gold": gold,
                            }
                        ),
                        file=f,
                    )

                if step % 100 == 0:
                    f.flush()

        accelerator.wait_for_everyone()

    # f1, f1_upperbound = (-1, -1)
    f1 = -1
    if accelerator.is_main_process:
        f1 = evaluate_most_probable(
            predictions=model_outputs_txt,
            gold=gold_txt,
            output_name=os.path.join(output_dir, f"{filename}"),
            task_labels=dataloader.dataset.task_labels,
        )

        # f1_upperbound = evaluate_best_prediction(
        #    predictions=model_outputs_txt,
        #    gold=gold_txt,
        #    output_name=os.path.join(output_dir, f"{filename}.upperbound"),
        #    task_labels=dataloader.dataset.task_labels,
        # )

        if stage == "dev":
            wandb.log(
                {
                    f"Val/{test_name}/f1_{'constrained' if constrained_generation else 'unconstrained'}": f1,
                    # f"Val/{test_name}/f1_upperbound": f1_upperbound,
                    "epoch": epoch,
                    "step": train_step,
                }
            )
        else:
            wandb.log(
                {
                    f"{test_name}/f1_{'constrained' if constrained_generation else 'unconstrained'}": f1,
                    # f"{test_name}/f1_upperbound": f1_upperbound,
                }
            )

        print(
            f"\n{test_name}\n"
            f"  -- f1_{'constrained' if constrained_generation else 'unconstrained'}: {f1}.\n"
            # f"  -- f1_upperbound: {f1_upperbound}\n"
        )

    return f1


def seq2seq(
    train_tsvs: List[str],
    dev_tsvs: List[str],
    test_tsvs: List[str],
    num_beams: int,
    num_return_sequences: int,
    max_source_length: int,
    max_target_length: int,
    model_name_or_path: str,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    learning_rate: int,
    weight_decay: float,
    num_train_epochs: int,
    gradient_accumulation_steps: int,
    optim: str,
    lr_scheduler_type: str,
    num_warmup_steps: int,
    output_dir: str,
    seed: int,
    eval_every_epochs: int,
    eval_every_steps: int,
    project_name: str,
    use_lora: bool,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    lora_target_modules: List[str],
    constrained_generation: bool,
    unconstrained_generation: bool,
    mixed_precision: str,
    quantization: int,
    local_rank: int,
    add_labels_as_tokens: bool,
    add_labels_as_prompt: bool,
    force_auto_device_map: bool,
    prompt: str,
    source_lang: str,
    target_lang: str,
    use_flash_attention: bool,
    trust_remote_code: bool,
):
    # if experiment_done(experiment_dir=output_dir, test_tsvs=test_tsvs):
    #    print(f"Experiment {output_dir} already done, skipping.")
    #    return True

    trust_remote_code = True  # @todo Remove for release

    if not constrained_generation:
        print(
            f"WARNING!!! Constrained generation is disabled, are you sure you want to do this?\n"
            f"Use --constrained_generation to enable it."
        )

    if constrained_generation and unconstrained_generation:
        print(
            f"We will use constrained generation and unconstrained generation. This means that we will run two "
            f"inference runs for each dataset. This is useful if you want to compare the performance of the model "
            f"with and without the constraints. If you don't want to run unconstrained generation, please remove "
            f"the --unconstrained_generation flag."
        )

    if quantization and train_tsvs is not None and not use_lora:
        raise ValueError(
            "Training with 8 bits or 4 bits quantization is only supported with LORA. If you want to train "
            "in Int8, please add the flag --use_lora. You can only evaluate in 4/8 bits without LoRA."
        )

    if seed is not None:
        set_seed(seed)

    accelerator = Accelerator(mixed_precision=mixed_precision)

    if accelerator.is_local_main_process:
        wandb.init(
            project=project_name,
            name=f"{os.path.basename(output_dir)}",
            resume=None,
            config={
                "max_source_length": max_source_length,
                "max_target_length": max_source_length,
                "per_device_eval_batch_size": 1,
                "output_dir": output_dir,
                "num_beams": num_beams,
                "num_return_sequences": num_return_sequences,
                "constrained_generation": constrained_generation,
                "unconstrained_generation": unconstrained_generation,
                "use_lora": use_lora,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "lora_target_modules": lora_target_modules,
                "model_name_or_path": model_name_or_path,
                "train_tsvs": train_tsvs,
                "dev_tsvs": dev_tsvs,
                "test_tsvs": test_tsvs,
                "numGPU": accelerator.num_processes,
                "quantization": quantization,
            },
        )

        if (
            train_tsvs is not None
        ):  # Do not overwrite the wandb run train info  if we are just evaluating
            wandb.config.per_device_train_batch_size = per_device_train_batch_size
            wandb.config.gradient_accumulation_steps = gradient_accumulation_steps
            wandb.config.learning_rate = learning_rate
            wandb.config.weight_decay = weight_decay
            wandb.config.lr_scheduler_type = lr_scheduler_type
            wandb.config.num_warmup_steps = num_warmup_steps
            wandb.config.seed = seed
            wandb.config.eval_every_epochs = eval_every_epochs
            wandb.config.eval_every_steps = eval_every_steps
            wandb.config.Mixed_precision = accelerator.mixed_precision
            wandb.config.num_train_epochs = num_train_epochs

    if train_tsvs is not None:
        print(f"Loading model from {model_name_or_path}")

        start_labels, end_labels = [], []
        for train_tsv in train_tsvs:
            sl, el = get_task_tags(train_tsv)
            start_labels.extend(sl)
            end_labels.extend(el)

        if use_lora and add_labels_as_tokens:
            extended_model_path = os.path.join(output_dir, "extended_model")
            print(
                f"Using LoRA and add_labels_as_tokens, we will create a new model extending the original one with the "
                f"labels as tokens. It will be saved in {extended_model_path}."
            )
            model, tokenizer, model_type = load_model(
                inference=True,
                model_weights_name_or_path=model_name_or_path,
                use_lora=False,
                quantization=None,
                add_labels_as_tokens=add_labels_as_tokens,
                labels=start_labels + end_labels,
                use_flash_attention=use_flash_attention,
                trust_remote_code=trust_remote_code,
            )

            model.save_pretrained(extended_model_path)
            tokenizer.save_pretrained(extended_model_path)

            model_name_or_path = extended_model_path

        model, tokenizer, model_type = load_model(
            inference=False,
            model_weights_name_or_path=model_name_or_path,
            use_lora=use_lora,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_target_modules=lora_target_modules,
            quantization=quantization,
            add_labels_as_tokens=add_labels_as_tokens,
            labels=start_labels + end_labels,
            force_auto_device_map=force_auto_device_map,
            use_gradient_checkpointing=quantization is not None or use_lora,
            use_flash_attention=use_flash_attention,
            trust_remote_code=trust_remote_code,
        )

        print(f"Model loaded!")

        if source_lang:
            try:
                _ = tokenizer.lang_code_to_id[source_lang]
            except KeyError:
                raise KeyError(
                    f"Language {source_lang} not found in tokenizer. "
                    f"Available languages: {tokenizer.lang_code_to_id.keys()}"
                )
            tokenizer.src_lang = source_lang
        if target_lang:
            try:
                forced_bos_token = tokenizer.lang_code_to_id[target_lang]
            except KeyError:
                raise KeyError(
                    f"Language {target_lang} not found in tokenizer. "
                    f"Available languages: {tokenizer.lang_code_to_id.keys()}"
                )
            tokenizer.tgt_lang = target_lang
        else:
            forced_bos_token = None

        trainable_params, all_param, percent_trainable = print_trainable_parameters(
            model
        )

        if accelerator.is_local_main_process:
            wandb.config.trainable_params = trainable_params
            wandb.config.all_param = all_param
            wandb.config.percent_trainable = percent_trainable

        print(f"Loading training dataset from {train_tsvs}")
        train_dataloader = get_dataloader(
            tokenizer=tokenizer,
            filenames=train_tsvs,
            batch_size=per_device_train_batch_size,
            max_source_len=max_source_length,
            max_target_len=max_target_length,
            is_encoder_decoder=model_type == "seq2seq",
            train=True,
            input_prompt=None if prompt is None else prompt,
            num_workers=min(os.cpu_count(), 8),
            add_labels_as_context=add_labels_as_prompt,
        )

        val_dataloaders = []
        print(
            f"Found {len(dev_tsvs)} validation datasets, we will average their scores for best model selection."
        )
        for dev_tsv in dev_tsvs:
            print(f"Loading validation dataset from {dev_tsv}")
            val_dataloaders.append(
                get_dataloader(
                    tokenizer=tokenizer,
                    filenames=[dev_tsv],
                    batch_size=per_device_eval_batch_size,
                    max_source_len=max_source_length,
                    max_target_len=max_target_length,
                    is_encoder_decoder=model_type == "seq2seq",
                    train=False,
                    input_prompt=None if prompt is None else prompt,
                    num_workers=min(os.cpu_count(), 8),
                    add_labels_as_context=add_labels_as_prompt,
                )
            )

        num_update_steps_per_epoch = math.ceil(
            len(train_dataloader)
            / gradient_accumulation_steps
            / accelerator.num_processes
        )
        max_train_steps = num_train_epochs * num_update_steps_per_epoch

        total_batch_size = (
            per_device_train_batch_size
            * accelerator.num_processes
            * gradient_accumulation_steps
        )

        if accelerator.is_local_main_process:
            wandb.config.total_batch_size = total_batch_size
            wandb.config.max_train_steps = max_train_steps

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if optim.lower() == "adamw8bits":
            import bitsandbytes as bnb

            optimizer = bnb.optim.PagedAdam8bit(
                optimizer_grouped_parameters,
                lr=learning_rate,
                betas=(0.9, 0.995),
            )

        elif optim.lower() == "adamw":
            optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate, eps=1e-7)
        elif optim.lower() == "adafactor":
            try:
                from fairseq.optim.adafactor import Adafactor
            except ImportError:
                raise ImportError(
                    "Please install fairseq to use Adafactor optimizer: "
                    "https://github.com/facebookresearch/fairseq#requirements-and-installation\n"
                    "You can run: pip install fairseq"
                )

            optimizer = Adafactor(
                params=optimizer_grouped_parameters,
                scale_parameter=False,
                relative_step=False,
                warmup_init=False,
                lr=learning_rate,
                clip_threshold=1.0,
                # weight_decay=args.weight_decay,
            )
        elif optim.lower() == "deepspeed":
            from accelerate.utils import DummyOptim

            optimizer = DummyOptim(
                params=optimizer_grouped_parameters,
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(
                f"Unknown optimizer: {optim}. Please choose from adamw, adafactor, adamw8bits"
            )

        model, optimizer, train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )

        lr_scheduler = get_scheduler(
            name=lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=max_train_steps,
        )

        completed_steps = 0
        best_epoch_metric: float = -1
        validation_dir: str = os.path.join(output_dir, "val_logs")
        os.makedirs(validation_dir, exist_ok=True)
        running_loss = 0
        num_batches = 0
        first = True

        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataloader.dataset)}")
        print(f"  Num Epochs = {num_train_epochs}")
        print(f"  Instantaneous batch size per device = {per_device_train_batch_size}")
        print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        print(f"  Total batch size = {total_batch_size}")
        print(f"  Total optimization steps = {max_train_steps}")
        print(f"  Learning rate = {learning_rate}")
        print(f"  Optimizer = {optim}")
        print(f"  Weight decay = {weight_decay}")
        print(f"  Scheduler = {lr_scheduler_type}")
        print(f"  Model = {model_name_or_path}")
        print(f"  Mixed Precision = {accelerator.mixed_precision}")
        print(f"  Num GPUs = {accelerator.num_processes}")
        print(f"  Seed = {seed}")
        print(f"  Eval every epochs = {eval_every_epochs}")
        print(f"  Eval every steps = {eval_every_steps}")
        print(f"  Num return sequences = {num_return_sequences}")
        print(f"  Num beams = {num_beams}")
        print(f"  Max target length = {max_target_length}")
        print(f"  Max source length = {max_source_length}")
        print(f"  Constrained generation = {constrained_generation}")
        print(f"  quantization = {quantization}")
        print(f"  Use LoRA = {use_lora}")
        print(f"  LoRA r = {lora_r}")
        print(f"  LoRA alpha = {lora_alpha}")
        print(f"  LoRA dropout = {lora_dropout}")
        print(f"  LoRA target modules = {lora_target_modules}")
        print(f"  Accelerator State : {{\n{accelerator.state}\n}}\n")
        print()

        progress_bar = tqdm(
            range(max_train_steps),
            disable=not accelerator.is_local_main_process,
            ascii=True,
            desc="Training",
        )

        model_name = accelerator.unwrap_model(model)._get_name()

        for epoch in range(num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                ### DEBUG ###
                if first and accelerator.is_local_main_process:
                    decodeable_inputs = batch.input_ids.clone()
                    decodeable_inputs[
                        decodeable_inputs == -100
                    ] = tokenizer.pad_token_id

                    model_inputs = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_inputs,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    # Labels without -100
                    decodeable_labels = batch.labels.clone()
                    decodeable_labels[
                        decodeable_labels == -100
                    ] = tokenizer.pad_token_id

                    labels = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_labels,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    words_ids = batch.words_ids

                    decodeable_original_sentence = batch.original_sentence_ids.clone()
                    decodeable_original_sentence[
                        decodeable_original_sentence == -100
                    ] = tokenizer.pad_token_id

                    original_sentences = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_original_sentence,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    decodeable_gold = batch.labeled_sentence_ids.clone()
                    decodeable_gold[decodeable_gold == -100] = tokenizer.pad_token_id

                    gold_sentences = "\n".join(
                        tokenizer.batch_decode(
                            decodeable_gold,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )
                    print(f"*** Sample of batch 0 ***")
                    print(f"-- Model inputs --\n{model_inputs}")
                    print(f"-- Labels --\n{labels}")
                    print(f"-- Words ids --\n{words_ids}")
                    print(f"-- Original sentences --\n{original_sentences}")
                    print(f"-- Gold sentences --\n{gold_sentences}")
                    print(f"*** End of sample ***\n")
                    first = False

                if (
                    model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values()
                    or model_name == "PeftModelForCausalLM"
                ):
                    # Decoder-only loss, we only compute loss on the labeled sequence tokens

                    if "labels" in batch:
                        labels = batch.pop("labels")
                    else:
                        raise ValueError(
                            "You should supply a labels key to compute the loss"
                        )

                    if "loss_weight_mask" in batch:
                        loss_weight_mask = batch.pop("loss_weight_mask")
                    else:
                        raise ValueError(
                            "You should supply a loss_weight_mask key to compute the loss"
                        )

                    outputs = model(
                        input_ids=batch.input_ids,
                        attention_mask=batch.attention_mask,
                        use_cache=False,
                    )

                    logits = (
                        outputs["logits"] if isinstance(outputs, dict) else outputs[0]
                    )

                    logits = logits[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                    loss_weight_mask = loss_weight_mask[..., 1:].contiguous()

                    logits = logits.view(-1, logits.size(-1))
                    labels = labels.view(-1)
                    loss_weight_mask = loss_weight_mask.view(-1)
                    loss_fct = nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

                    loss = loss_fct(logits, labels)
                    # print(f"Loss: {loss}")

                    loss = torch.sum(loss * loss_weight_mask) / torch.sum(
                        loss_weight_mask
                    )
                    # print(f"Loss after sum: {loss}")

                    # print(f"Loss weight mask: {loss_weight_mask}")

                else:
                    # Encoder-decoder loss, only computed on the labeled sequence tokens
                    loss = model(
                        input_ids=batch.input_ids,
                        labels=batch.labels,
                        attention_mask=batch.attention_mask,
                    ).loss

                running_loss += loss.item()
                loss = loss / gradient_accumulation_steps
                accelerator.backward(loss)
                num_batches += 1

                if (
                    step % gradient_accumulation_steps == 0
                    or step == len(train_dataloader) - 1
                ):
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1
                    if (
                        accelerator.is_local_main_process
                        and completed_steps > 0
                        and (completed_steps % 100 == 0)
                    ):
                        wandb.log(
                            {
                                "Train/Loss": loss.item(),
                                "Train/Running Loss": loss.item() / num_batches,
                                "Train/Learning Rate": optimizer.param_groups[0]["lr"],
                                "epoch": epoch,
                                "step": completed_steps,
                            }
                        )

                    if eval_every_steps > 0 and (
                        ((completed_steps + 1) % eval_every_steps) == 0
                        or (completed_steps + 1 == max_train_steps)
                    ):
                        f1_scores = []
                        for val_dataloader in val_dataloaders:
                            val_dataloader = accelerator.prepare(val_dataloader)
                            model.eval()
                            f1 = -1.0
                            if unconstrained_generation:
                                f1 = evaluate(
                                    dataloader=val_dataloader,
                                    constrained_generation=False,
                                    accelerator=accelerator,
                                    model=model,
                                    tokenizer=tokenizer,
                                    max_length=max_target_length,
                                    num_beams=num_beams,
                                    num_return_sequences=num_return_sequences,
                                    output_dir=validation_dir,
                                    stage="dev",
                                    epoch=epoch,
                                    train_step=completed_steps,
                                    forced_bos_token=forced_bos_token,
                                )
                            if constrained_generation:
                                f1 = evaluate(
                                    dataloader=val_dataloader,
                                    constrained_generation=True,
                                    accelerator=accelerator,
                                    model=model,
                                    tokenizer=tokenizer,
                                    max_length=max_target_length,
                                    num_beams=num_beams,
                                    num_return_sequences=num_return_sequences,
                                    output_dir=validation_dir,
                                    stage="dev",
                                    epoch=epoch,
                                    train_step=completed_steps,
                                    forced_bos_token=forced_bos_token,
                                )
                            if accelerator.is_local_main_process:
                                f1_scores.append(f1)

                        if accelerator.is_local_main_process:
                            epoch_f1 = sum(f1_scores) / len(f1_scores)
                            if (
                                (epoch_f1 > best_epoch_metric)
                                or (best_epoch_metric < 0)
                                or (math.isnan(best_epoch_metric))
                            ):
                                print(
                                    f"New best model :) Epoch {epoch} Step {completed_steps} "
                                    f"PrevF1 {round(best_epoch_metric * 100, 2)} F1 {round(epoch_f1 * 100, 2)}"
                                )
                                best_epoch_metric = epoch_f1
                                unwrapped_model = accelerator.unwrap_model(model)
                                unwrapped_model.save_pretrained(
                                    output_dir, save_function=accelerator.save
                                )
                                tokenizer.save_pretrained(output_dir)

                            else:
                                print(
                                    f"This epoch did not improve :( Epoch {epoch} Step {completed_steps} "
                                    f"PrevF1 {round(best_epoch_metric * 100, 2)} F1 {round(epoch_f1 * 100, 2)}"
                                )

                        accelerator.wait_for_everyone()
                        model.train()

            if (
                eval_every_epochs > 0
                and ((epoch + 1) % eval_every_epochs) == 0
                or (epoch + 1) == num_train_epochs
            ):
                f1_scores = []
                for val_dataloader in val_dataloaders:
                    val_dataloader = accelerator.prepare(val_dataloader)
                    model.eval()
                    f1 = -1.0
                    if unconstrained_generation:
                        f1 = evaluate(
                            dataloader=val_dataloader,
                            constrained_generation=False,
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            max_length=max_target_length,
                            num_beams=num_beams,
                            num_return_sequences=num_return_sequences,
                            output_dir=validation_dir,
                            stage="dev",
                            epoch=epoch,
                            train_step=completed_steps,
                            forced_bos_token=forced_bos_token,
                        )
                    if constrained_generation:
                        f1 = evaluate(
                            dataloader=val_dataloader,
                            constrained_generation=True,
                            accelerator=accelerator,
                            model=model,
                            tokenizer=tokenizer,
                            max_length=max_target_length,
                            num_beams=num_beams,
                            num_return_sequences=num_return_sequences,
                            output_dir=validation_dir,
                            stage="dev",
                            epoch=epoch,
                            train_step=completed_steps,
                            forced_bos_token=forced_bos_token,
                        )
                    if accelerator.is_local_main_process:
                        f1_scores.append(f1)

                if accelerator.is_local_main_process:
                    epoch_f1 = sum(f1_scores) / len(f1_scores)
                    if (
                        (epoch_f1 > best_epoch_metric)
                        or (best_epoch_metric < 0)
                        or (math.isnan(best_epoch_metric))
                    ):
                        print(
                            f"New best model :) Epoch {epoch} Step {completed_steps} "
                            f"PrevF1 {round(best_epoch_metric*100,2)} F1 {round(epoch_f1*100,2)}"
                        )
                        best_epoch_metric = epoch_f1
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(
                            output_dir, save_function=accelerator.save
                        )
                        tokenizer.save_pretrained(output_dir)

                    else:
                        print(
                            f"This epoch did not improve :( Epoch {epoch} Step {completed_steps} "
                            f"PrevF1 {round(best_epoch_metric*100,2)} F1 {round(epoch_f1*100,2)}"
                        )

                accelerator.wait_for_everyone()
                model.train()

        progress_bar.close()

        if accelerator.is_local_main_process and eval_every_epochs < 0:
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
            tokenizer.save_pretrained(output_dir)

    if test_tsvs is not None:
        print(f"========= TESTING =========")

        print(f"Loading best model from {output_dir}")

        if use_lora:
            if train_tsvs is not None:
                weights_path = model_name_or_path
                lora_weights_name_or_path = output_dir
            else:
                weights_path = (
                    model_name_or_path
                    if not add_labels_as_tokens
                    else os.path.join(model_name_or_path, "extended_model")
                )
                lora_weights_name_or_path = model_name_or_path
        else:
            if train_tsvs is not None:
                weights_path = output_dir
                lora_weights_name_or_path = None
            else:
                weights_path = model_name_or_path
                lora_weights_name_or_path = None

        model, tokenizer, model_type = load_model(
            inference=True,
            model_weights_name_or_path=weights_path,
            quantization=quantization,
            use_lora=lora_weights_name_or_path is not None,
            lora_weights_name_or_path=lora_weights_name_or_path,
            force_auto_device_map=force_auto_device_map,
            use_flash_attention=use_flash_attention,
            trust_remote_code=trust_remote_code,
        )

        if source_lang:
            try:
                _ = tokenizer.lang_code_to_id[source_lang]
            except KeyError:
                raise KeyError(
                    f"Language {source_lang} not found in tokenizer. "
                    f"Available languages: {tokenizer.lang_code_to_id.keys()}"
                )
            tokenizer.src_lang = source_lang
        if target_lang:
            try:
                forced_bos_token = tokenizer.lang_code_to_id[target_lang]
            except KeyError:
                raise KeyError(
                    f"Language {target_lang} not found in tokenizer. "
                    f"Available languages: {tokenizer.lang_code_to_id.keys()}"
                )
            tokenizer.tgt_lang = target_lang
        else:
            forced_bos_token = None

        model = accelerator.prepare(model)

        for test_tsv in test_tsvs:
            print(f"Testing on {test_tsv}...")
            test_dataloader = get_dataloader(
                tokenizer=tokenizer,
                filenames=[test_tsv],
                batch_size=per_device_eval_batch_size,
                max_source_len=max_source_length,
                max_target_len=max_target_length,
                is_encoder_decoder=model_type == "seq2seq",
                train=False,
                input_prompt=None if prompt is None else prompt,
                num_workers=min(os.cpu_count(), 8),
                add_labels_as_context=add_labels_as_prompt,
            )

            test_dataloader = accelerator.prepare(test_dataloader)

            if unconstrained_generation:
                _ = evaluate(
                    dataloader=test_dataloader,
                    constrained_generation=False,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_target_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    output_dir=output_dir,
                    stage="test",
                    forced_bos_token=forced_bos_token,
                )
            if constrained_generation:
                _ = evaluate(
                    dataloader=test_dataloader,
                    constrained_generation=True,
                    accelerator=accelerator,
                    model=model,
                    tokenizer=tokenizer,
                    max_length=max_target_length,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    output_dir=output_dir,
                    stage="test",
                    forced_bos_token=forced_bos_token,
                )


def main():
    args = parse_args()
    seq2seq(**vars(args))


if __name__ == "__main__":
    main()
