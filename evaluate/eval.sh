# Change your evaluation setting here to evaluate the model generation elegantly
python evaluate.py \
        --eval_model llama \
        --baseline_method task_retrieve \
        --version final_eval \
        --eval_set_type robot \
        --eval_fast False