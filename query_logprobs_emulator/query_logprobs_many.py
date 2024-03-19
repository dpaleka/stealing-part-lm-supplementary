import itertools
import csv
from pathlib import Path
from utils import get_model_enc, token_to_str, hash_params
import concurrent.futures
import asyncio
import torch
import os

from query_logprobs_batch import query_logprobs_batch

# Set the number of prompts and the number of jobs
N = 1000
CUDA_DEVICES: list[int] = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
K = len(
    CUDA_DEVICES
)  # should in principle go more, but sometimes it crashes for some reason
print(f"{CUDA_DEVICES=}", f"{K=}")

models = [
    f"EleutherAI/pythia-{size}"
    for size in ["14m", "31m", "70m", "160m", "410m", "1b", "1.4b"]
]

model_name = "EleutherAI/pythia-410m"
bias = 80.0
n_tokens_to_downweight = 12
batch_size = 4096
top_logprobs = 5
model_kwargs = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
}
# model_kwargs = None

enc = get_model_enc(model_name)
token_str_map = token_to_str(model_name)

# Find all alphanumeric tokens
# alphanumeric_tokens = [token for token, string in token_str_map.items() if string.isalnum()]
# use only ascii
alphanumeric_tokens = [
    token
    for token, string in token_str_map.items()
    if string.isascii() and string.isalnum()
]


# Generator function to create pairs of tokens
def generate_pairs(tokens, max_pairs):
    count = 0
    for pair in itertools.product(tokens, repeat=2):
        if count >= max_pairs:
            return
        yield pair
        count += 1


# Generate the prompts and write to a CSV file
prompts_path = Path("prompts_10000.csv")
seen_prompts = set()
index = 0

with prompts_path.open("w", newline="") as csvfile:
    prompt_writer = csv.writer(csvfile)
    prompt_writer.writerow(["id", "first_token", "second_token", "prompt"])

    for token in alphanumeric_tokens:
        prompt_single = token_str_map[token]
        if prompt_single not in seen_prompts:
            seen_prompts.add(prompt_single)
            prompt_writer.writerow([index, token, "", prompt_single])
            index += 1
        if index >= N:
            break

    for first_token, second_token in generate_pairs(
        alphanumeric_tokens, N - len(seen_prompts)
    ):
        prompt_pair = token_str_map[first_token] + token_str_map[second_token]
        if prompt_pair not in seen_prompts:
            seen_prompts.add(prompt_pair)
            prompt_writer.writerow([index, first_token, second_token, prompt_pair])
            index += 1

# Create a directory to store the prompt lists
prompt_lists_dir = Path("prompt_lists")
prompt_lists_dir.mkdir(exist_ok=True)

# Split the CSV file into K parts
with prompts_path.open("r") as csvfile:
    reader = csv.DictReader(csvfile)
    prompts = list(reader)

# Save each part to a separate file
for i in range(K):
    part_file = prompt_lists_dir / f"part_{i}.csv"
    with part_file.open("w", newline="") as csvfile:
        prompt_writer = csv.writer(csvfile)
        prompt_writer.writerow(["id", "first_token", "second_token", "prompt"])
        for prompt in prompts[i::K]:
            prompt_writer.writerow(
                [
                    prompt["id"],
                    prompt["first_token"],
                    prompt["second_token"],
                    prompt["prompt"],
                ]
            )


def run_query_logprobs_batch(command):
    # Create a new event loop for the process
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Run the coroutine and return the result
    result = loop.run_until_complete(query_logprobs_batch(**command))
    loop.close()
    return result


async def run_async_commands(commands, log_file_paths):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        tasks = []
        for command in commands:
            # Schedule the execution of the wrapper function in the executor
            task = loop.run_in_executor(executor, run_query_logprobs_batch, command)
            tasks.append(task)
        # Wait for all scheduled tasks to complete
        results = await asyncio.gather(*tasks)
    return results


commands = []
log_files = []
# Run parallel jobs using query_logprobs_batch.py
for i in range(K):
    part_file = prompt_lists_dir / f"part_{i}.csv"
    command = {
        "model_name": model_name,
        "model_kwargs": model_kwargs,
        "prompt_file": part_file,
        "bias": bias,
        "n_tokens_to_downweight": n_tokens_to_downweight,
        "top_logprobs": top_logprobs,
        "batch_size": batch_size,
        "cuda_device": i % len(CUDA_DEVICES),
        "overwrite_cache": False,
    }
    print(
        "python query_logprobs_batch.py "
        + " ".join([f"--{k} {str(v)}" for k, v in command.items()])
    )
    commands.append(command)

    # Restarting should append to the log file instead of overwriting it
    model_kwargs_hash = hash_params(model_kwargs)
    log_file_path = Path(
        f'logs/query_logprobs_batch_{model_name.replace("/", "_")}_{bias}_{n_tokens_to_downweight}_{batch_size}_{part_file.stem}.{model_kwargs_hash}.o.log'
    )
    log_files.append(log_file_path)

    # Create the logfile if it does not exist
    log_file_path.parent.mkdir(exist_ok=True, parents=True)
    log_file_path.touch(exist_ok=True)

# Run the jobs in parallel
asyncio.run(run_async_commands(commands, log_files))
