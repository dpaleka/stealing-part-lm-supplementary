# query_logprobs_batch.py
import csv
import os
import sys
from query_logprobs import (
    query_all_token_logprobs_simple,
)  # Assuming this function is in query_logprobs.py


async def query_logprobs_batch(
    model_name,
    cuda_device,
    prompt_file,
    bias,
    n_tokens_to_downweight,
    model_kwargs,
    batch_size,
    top_logprobs,
    overwrite_cache,
):
    print(
        f"Model: {model_name}, Prompt File: {prompt_file}, Bias: {bias}, Tokens to Downweight: {n_tokens_to_downweight}, Batch Size: {batch_size}, Overwrite Cache: {overwrite_cache}"
    )

    print(f"{os.environ['CUDA_VISIBLE_DEVICES']=}")
    with open(prompt_file, newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            prompt = row["prompt"]
            # Call the function to query all token logprobs for each prompt
            await query_all_token_logprobs_simple(
                model_name=model_name,
                prompt=prompt,
                bias=bias,
                n_tokens_to_downweight=n_tokens_to_downweight,
                model_kwargs=model_kwargs,
                batch_size=batch_size,
                top_logprobs=top_logprobs,
                overwrite_cache=overwrite_cache,
                cuda_device=cuda_device,
            )


if __name__ == "__main__":
    print(sys.argv)
    query_logprobs_batch()
