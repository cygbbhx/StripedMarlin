import transformers
import torch

model_id = "meta-llama/Llama-2-7b-chat-hf"

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
)

messages = [
    {"role": "user", "content": "Give me list of around 100 example background sounds which are not related to human voice. Make sure all the noises are unique. Return format should look like list of strings (ex. ['sound1', 'sound2' ....])"}
]

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

outputs = pipeline(
    messages,
    max_new_tokens=4096,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.9,
    top_p=0.9,
)

response = outputs[0]["generated_text"][-1]['content']
print(response)

with open('tmp.txt', 'w') as f:
    f.write(response)
