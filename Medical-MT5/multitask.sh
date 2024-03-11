#!/bin/bash
#SBATCH --job-name=medMT5-multitask
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=medMT5-multitask.out.txt
#SBATCH --error=medMT5-multitask.err.txt

source /ikerlariak/igarcia945/envs/pytorch2/bin/activate
cd ..
export LC_ALL=en_US.UTF-8
export LANG=en_US.UTF-8
export LANGUAGE=en_US.UTF-8
export TOKENIZERS_PARALLELISM=true
export OMP_NUM_THREADS=16
export PMI_SIZE=1
export OMPI_COMM_WORLD_SIZE=1
export MV2_COMM_WORLD_SIZE=1
export WORLD_SIZE=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS="true"



# Run MultiTask

for model_name in  \
google/flan-t5-large \
google/flan-t5-xl \
/gaueko1/hizkuntza-ereduak/medT5/medT5-large \
/gaueko1/hizkuntza-ereduak/medT5/medT5-xl \
razent/SciFive-large-Pubmed_PMC \
google/mt5-large \
google/mt5-xl
do
  modelpath=/scratch/igarcia945/
  modelparams="${model_name//"$modelpath"}"
  modelpath2=/en_es_fr_it/ms_1024_lr_0.001_constant/epoch_0_step_
  modelparams="${modelparams//"$modelpath2"}"
  modelparams=$(printf '%s' "$modelparams" | sed 's/[0-9]\+$//')
  modelparams="${modelparams//\//_}"

  accelerate launch --mixed_precision bf16 seq2seq.py \
  --mixed_precision bf16 \
  --constrained_generation \
  --train_tsvs \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-e3c-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-e3c-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/fr/fr-e3c-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/it/it-e3c-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-diann-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-diann-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-neoplasm-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-neoplasm-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/fr/fr-neoplasm-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/it/it-neoplasm-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-ncbi-disease-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-bc5cdr_disease-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-bc5cdr_chemical-train.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-pharmaconer-bsc-train.tsv \
  --dev_tsvs \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-e3c-dev.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-ncbi-disease-dev.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-bc5cdr_disease-dev.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-pharmaconer-bsc-dev.tsv \
  --test_tsvs \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-e3c-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-e3c-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/fr/fr-e3c-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/it/it-e3c-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-diann-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-diann-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-neoplasm-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-neoplasm-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/fr/fr-neoplasm-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/it/it-neoplasm-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-ncbi-disease-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-bc5cdr_disease-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/en/en-bc5cdr_chemical-test.tsv \
  /ikerlariak/igarcia945/antidote_mt5-corpus/sequence-labeling-evaluation-datasets/es/es-pharmaconer-bsc-test.tsv \
  --num_beams 4 \
  --num_return_sequences 1 \
  --model_name_or_path "$model_name" \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 12 \
  --output_dir ./Antidote_mT5_5/all/"$modelparams"_all \
  --seed 42 \
  --eval_every_epochs 4 \
  --max_source_length 256 \
  --max_target_length 256 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 500 \
  --project_name "Antidote_mT5_fix" \
  --add_labels_as_prompt \
  --add_labels_as_tokens


  rm -rf ./Antidote_mT5_5/all/"$modelparams"_all/*.bin

done
