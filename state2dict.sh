for model in gpt2 gpt2-medium gpt2-large gpt2-xl;do
    python state-dict-convert.py --name $model
done