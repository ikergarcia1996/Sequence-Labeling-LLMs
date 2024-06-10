  accelerate launch --mixed_precision bf16 seq2seq.py \
  --mixed_precision bf16 \
  --constrained_generation \
  --train_tsvs \
  medial-domain-datasets/en/en-e3c-train.tsv \
  medial-domain-datasets/es/es-e3c-train.tsv \
  medial-domain-datasets/fr/fr-e3c-train.tsv \
  medial-domain-datasets/it/it-e3c-train.tsv \
  medial-domain-datasets/en/en-diann-train.tsv \
  medial-domain-datasets/es/es-diann-train.tsv \
  medial-domain-datasets/en/en-neoplasm-train.tsv \
  medial-domain-datasets/es/es-neoplasm-train.tsv \
  medial-domain-datasets/fr/fr-neoplasm-train.tsv \
  medial-domain-datasets/it/it-neoplasm-train.tsv \
  medial-domain-datasets/en/en-ncbi-disease-train.tsv \
  medial-domain-datasets/en/en-bc5cdr_disease-train.tsv \
  medial-domain-datasets/en/en-bc5cdr_chemical-train.tsv \
  medial-domain-datasets/es/es-pharmaconer-bsc-train.tsv \
  --dev_tsvs \
  medial-domain-datasets/en/en-e3c-dev.tsv \
  medial-domain-datasets/en/en-ncbi-disease-dev.tsv \
  medial-domain-datasets/en/en-bc5cdr_disease-dev.tsv \
  medial-domain-datasets/es/es-pharmaconer-bsc-dev.tsv \
  --test_tsvs \
  medial-domain-datasets/en/en-e3c-test.tsv \
  medial-domain-datasets/es/es-e3c-test.tsv \
  medial-domain-datasets/fr/fr-e3c-test.tsv \
  medial-domain-datasets/it/it-e3c-test.tsv \
  medial-domain-datasets/en/en-diann-test.tsv \
  medial-domain-datasets/es/es-diann-test.tsv \
  medial-domain-datasets/en/en-neoplasm-test.tsv \
  medial-domain-datasets/es/es-neoplasm-test.tsv \
  medial-domain-datasets/fr/fr-neoplasm-test.tsv \
  medial-domain-datasets/it/it-neoplasm-test.tsv \
  medial-domain-datasets/en/en-ncbi-disease-test.tsv \
  medial-domain-datasets/en/en-bc5cdr_disease-test.tsv \
  medial-domain-datasets/en/en-bc5cdr_chemical-test.tsv \
  medial-domain-datasets/es/es-pharmaconer-bsc-test.tsv \
  --num_beams 1 \
  --num_return_sequences 1 \
  --model_name_or_path google/flan-t5-large \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --learning_rate 1e-4 \
  --num_train_epochs 12 \
  --output_dir results/medical-flan-t5-large \
  --seed 42 \
  --eval_every_epochs 4 \
  --max_source_length 256 \
  --max_target_length 256 \
  --lr_scheduler_type cosine \
  --num_warmup_steps 500 \
  --project_name "medical-flan-t5-large" \
  --add_labels_as_prompt \
  --add_labels_as_tokens