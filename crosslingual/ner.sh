#!/bin/bash
#SBATCH --job-name=masakhaner
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --output=crosslingual/masakhaner.out.txt
#SBATCH --error=crosslingual/masakhaner.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate

export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export TRANSFORMERS_NO_ADVISORY_WARNINGS=true
export OMP_NUM_THREADS=16

echo CUDA_VISIBLE_DEVICES "${CUDA_VISIBLE_DEVICES}"


for model in google/gemma-1.1-7b-it meta-llama/Meta-Llama-3-8B-Instruct CohereForAI/aya-23-8B CohereForAI/aya-101 google/mt5-xl bigscience/mt0-xl Qwen/Qwen2-7B-Instruct 01-ai/Yi-1.5-9B-Chat
do
accelerate launch --use_deepspeed --deepspeed_config_file deepspeed_configs/deepspeed_zero2.json --mixed_precision bf16 --num_processes 4 --num_machines 1 --dynamo_backend no seq2seq.py \
--mixed_precision bf16 \
--constrained_generation \
--unconstrained_generation \
--train_tsvs  /ikerlariak/igarcia945/ner_datasets/remove_misc/en/en.conll.train.tsv \
--dev_tsvs /ikerlariak/igarcia945/ner_datasets/remove_misc/en/en.conll.dev.tsv \
--test_tsvs \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/bam/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/bbj/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/ewe/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/fon/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/hau/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/ibo/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/kin/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/lug/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/mos/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/nya/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/pcm/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/sna/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/swa/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/tsn/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/twi/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/wol/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/xho/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/yor/test.tsv \
/ikerlariak/igarcia945/ner_datasets/MasakhaNER2/conll_labels/zul/test.tsv \
/ikerlariak/igarcia945/ner_datasets/remove_misc/en/en.conll.test.tsv \
/ikerlariak/igarcia945/ner_datasets/remove_misc/es/es.conll.test.tsv \
/ikerlariak/igarcia945/ner_datasets/remove_misc/de/de.conll.test.tsv \
/ikerlariak/igarcia945/ner_datasets/remove_misc/it/it.evalita.test.tsv \
--num_beams 4 \
--num_return_sequences 1 \
--model_name_or_path ${model} \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 2 \
--per_device_eval_batch_size 2 \
--learning_rate 0.00007 \
--optim adamw \
--lr_scheduler_type cosine \
--num_warmup_steps 500 \
--num_train_epochs 8 \
--eval_every_epochs 2 \
--max_source_length 256 \
--max_target_length 256 \
--output_dir results/masakhaner2/${model} \
--project_name SeqLabeling_w_LLMs \
--add_labels_as_tokens

done

