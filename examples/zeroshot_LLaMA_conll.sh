accelerate launch seq2seq.py \
--quantization 4 \
--constrained_generation \
--test_tsvs examples/conll/en.conll.test.tsv \
--num_beams 1 \
--num_return_sequences 1 \
--model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
--per_device_eval_batch_size 8 \
--max_source_length 256 \
--max_target_length 256 \
--output_dir results/conll/Llama-3-8B-Instruct \
--project_name SeqLabeling_w_LLMs \
--prompt "Label the Locations, Organizations, and Persons in the following sentences. Use html tags to mark the entities.\nThe president Barack Obama was born in Hawaii many years ago . -> The president <Person> Barack Obama </Person> was born in <Location> Hawaii </Location> many years ago .\nGoogle is a company in California . -> <Organization> Google </Organization> is a company in <Location> California </Location> .\nBill Gates is the founder of Microsoft a large company . -> <Person> Bill Gates </Person> is the founder of <Organization> Microsoft </Organization> a large company .\n"

