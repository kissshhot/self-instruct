batch_dir=data/gpt3_generations/

CUDA_VISIBLE_DEVICES=6,7 python self_instruct/bootstrap_instructions.py \
    --batch_dir ${batch_dir} \
    --num_instructions_to_generate 100000 \
    --seed_tasks_path data/seed_tasks.jsonl \
    --engine "davinci"