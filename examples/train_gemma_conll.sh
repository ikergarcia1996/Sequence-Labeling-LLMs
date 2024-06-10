# This example requires a lot of VRAM. You can use the 'train_gemma_conll_deepspeed.sh' script to 
# split the model across GPUs using DeepSpeed. Or 'train_LLaMA_conll_QLoRA.sh' to use QLoRA with
# 4-bit quantization to fit the training in a single GPU.
accelerate launch seq2seq.py \
--mixed_precision bf16 \
--constrained_generation \
--constrained_generation \
--train_tsvs examples/conll/en.conll.train.tsv \
--dev_tsvs examples/conll/en.conll.dev.tsv \
--test_tsvs examples/conll/en.conll.test.tsv \
--num_beams 1 \
--num_return_sequences 1 \
--model_name_or_path google/gemma-2b-it \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 8 \
--per_device_eval_batch_size 2 \
--learning_rate 0.00007 \
--optim adamw \
--lr_scheduler_type cosine \
--num_warmup_steps 500 \
--num_train_epochs 30 \
--eval_every_epochs 5 \
--max_source_length 256 \
--max_target_length 256 \
--output_dir results/conll/FlanT5large \
--project_name SeqLabeling_w_LLMs \
--add_labels_as_tokens

