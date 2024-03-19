# %%
"""
Query the OpenAI/LLaMA logprobs endpoint with a given prompt, completion,
and logit_bias values, for all tokens in the prompt.

Structure: query(client, logit_bias, prompt)
returns a dictionary with results of the query,
as in the OpenAI API.
"""
import asyncio
import json
import os
import pickle

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from pathlib import Path
from tqdm import tqdm

load_dotenv()

from llm_server import Interactor
from utils import (
    get_model_enc,
    is_completion,
    is_openai,
    print_logprob_dict,
    remove_keys,
    str_to_token,
    hash_params,
    make_json_serializable,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
CACHE_DIR = "/data/logits/cachier"


# %%
# @cachier(cache_dir=CACHE_DIR, hash_func=stringify_params)
def query_single_token_openai_chat(
    model, messages, top_logprobs: int = 5, logit_bias=None, overwrite_cache=True
):
    assert top_logprobs <= 5
    client = OpenAI(timeout=10.0, max_retries=1)
    # print(f"messages in qstochat: {messages}")

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=42,
        max_tokens=1,
        temperature=0,
        logit_bias={str(k): v for k, v in logit_bias.items()}
        if logit_bias is not None
        else {},
        logprobs=True,
        top_logprobs=top_logprobs,
    )
    logprobs = completion.model_dump(exclude_unset=True)["choices"][0]["logprobs"][
        "content"
    ]
    return logprobs


def make_chat_logprob_dict_from_completion_data(model, completion_data):
    # Want to return:
    """
    'token': str
    'logprob': float
    'top_logprobs': [{'token': str, 'logprob': float} dict]
    """

    # print(f"completion_data: {completion_data}")

    ret = {
        "token": completion_data["tokens"][0],
        "logprob": completion_data["token_logprobs"][0],
        "top_logprobs": [
            {"token": k, "logprob": v}
            for k, v in completion_data["top_logprobs"][0].items()
        ],
        "top_logprobs_dict": completion_data["top_logprobs"][0],
    }
    return ret


def query_single_token_openai_completion(
    model,
    messages: list[dict[str, str]],
    logit_bias: dict[str, float] | None = None,
    top_logprobs: int = 5,
    overwrite_cache=True,
):
    # Untested

    assert top_logprobs <= 5
    client = OpenAI(timeout=10.0, max_retries=1)

    assert len(messages) == 1
    prompt = messages[0]["content"]

    completion = client.completions.create(
        model=model,
        prompt=prompt,
        seed=42,
        max_tokens=1,
        temperature=0,
        logit_bias={str(k): v for k, v in logit_bias.items()}
        if logit_bias is not None
        else {},
        logprobs=top_logprobs,
    )
    # logprobs = completion.model_dump(exclude_unset=True)["choices"][0]["logprobs"]["token_logprobs"]
    logprobs = make_chat_logprob_dict_from_completion_data(
        model, completion.model_dump(exclude_unset=True)["choices"][0]["logprobs"]
    )
    print(f"logprobs: {logprobs}")
    return logprobs


# @cachier(cache_dir=CACHE_DIR, hash_func=stringify_params)
def query_single_token_local(
    model: str,
    conversations: list[list[dict[str, str]]],
    logit_biases: list[dict[str, float]] | None = None,
    top_logprobs: int = 5,
    model_kwargs_hash: str | None = None,
    overwrite_cache: bool = True,
) -> list[dict[str, float]]:
    client = Interactor(model)  # kwargs should already be loaded in the model

    prompts = [client.format_msgs(conversation) for conversation in conversations]
    print(f"prompts: {prompts}")
    print(f"logit_biases: {logit_biases}")
    logprobs = client.query_single_token(
        prompts, logit_biases=logit_biases, top_logprobs=top_logprobs, temperature=0.0
    )
    return logprobs


async def query_single_token(
    model,
    conversations,
    logit_biases=None,
    batch_size=2560,
    top_logprobs=5,
    model_kwargs_hash: str | None = None,
    overwrite_cache=False,
) -> list[dict[str, float]]:
    assert isinstance(conversations, list) and isinstance(conversations[0], list)
    if logit_biases is not None:
        assert len(conversations) == len(logit_biases)

    if is_openai(model):
        if len(conversations) == 1:
            if is_completion(model):
                return query_single_token_openai_completion(
                    model,
                    conversations,
                    logit_biases[0] if logit_biases is not None else None,
                    top_logprobs=top_logprobs,
                    overwrite_cache=overwrite_cache,
                )
            else:
                return query_single_token_openai_chat(
                    model,
                    conversations,
                    logit_biases[0] if logit_biases is not None else None,
                    top_logprobs=top_logprobs,
                    overwrite_cache=overwrite_cache,
                )
        else:
            max_concurrent_queries = batch_size
            # print(f"Using max concurrent queries {max_concurrent_queries}")
            semaphore = asyncio.Semaphore(max_concurrent_queries)

            async def query_single_token_openai_async(messages, logit_bias=None):
                async with semaphore:
                    if is_completion(model):
                        return query_single_token_openai_completion(
                            model,
                            messages,
                            logit_bias,
                            top_logprobs=top_logprobs,
                            overwrite_cache=overwrite_cache,
                        )
                    else:
                        return query_single_token_openai_chat(
                            model,
                            messages,
                            logit_bias,
                            top_logprobs=top_logprobs,
                            overwrite_cache=overwrite_cache,
                        )

            tasks = []
            for i in range(len(conversations)):
                tasks.append(
                    query_single_token_openai_async(
                        conversations[i],
                        logit_biases[i] if logit_biases is not None else None,
                    )
                )
            results = await asyncio.gather(*tasks)
            print(f"results: {results}")
            return results

    else:  # local model
        if len(conversations) == 1:
            return query_single_token_local(
                model=model,
                conversations=conversations,
                logit_biases=logit_biases,
                top_logprobs=top_logprobs,
                model_kwargs_hash=model_kwargs_hash,
                overwrite_cache=overwrite_cache,
            )
        else:

            def query_chunk(chunk, logit_biases_chunk):
                assert isinstance(chunk, list) and isinstance(chunk[0], list)
                return query_single_token_local(
                    model=model,
                    conversations=chunk,
                    logit_biases=logit_biases_chunk,
                    top_logprobs=top_logprobs,
                    model_kwargs_hash=model_kwargs_hash,
                    overwrite_cache=overwrite_cache,
                )

            chunks = [
                conversations[i : i + batch_size]
                for i in range(0, len(conversations), batch_size)
            ]
            logit_biases_chunks = (
                [
                    logit_biases[i : i + batch_size]
                    for i in range(0, len(conversations), batch_size)
                ]
                if logit_biases is not None
                else [None] * len(chunks)
            )
            results = []
            for chunk, logit_biases_chunk in tqdm(
                zip(chunks, logit_biases_chunks), desc="Processing chunks", unit="chunk"
            ):
                results.append(query_chunk(chunk, logit_biases_chunk))
            return [item for sublist in results for item in sublist]  # flatten


# %%
async def query_and_print_logprobs(
    model_name, conversations, logit_biases=None, overwrite_cache=False
):
    logprobs = await query_single_token(
        model_name,
        conversations,
        logit_biases=logit_biases,
        overwrite_cache=overwrite_cache,
    )
    for logprob_dict in logprobs:
        print_logprob_dict(logprob_dict, model=model_name)
    print(json.dumps(remove_keys(logprobs, ["bytes"]), indent=2))


# %%
"""
The math is as follows:
    logsoftmax(x) = log(exp(x) / sum(exp(x))) = x - log(sum(exp(x)))

Let 
    B = sum(exp(x))
    b[S] = sum([exp(x[i]) for i in S])

Let 
    logit_bias[i] = 0 for i not in S, logit_bias[i] = a[i] for i in S

We observe the top few y[i] = logsoftmax(x + logit_bias)[i] for i in S.
    y[i] = logsoftmax(x + logit_bias)[i] = x + logit_bias[i] - log(B - b[S] + sum([exp(x[i] + a[i]) for i in S]))
    y[i] - y[j] = x[i] - x[j] + logit_bias[i] - logit_bias[j]

In case i, j are in S, this simplifies to
    y[i] - y[j] = x[i] - x[j] + a[i] - a[j]
    x[i] - x[j] = y[i] - y[j] - a[i] + a[j]

In case we have an anchor token (e.g. i=r), we can compute x[j] - x[r] = y[j] - y[r] - a[j] + a[r]  for all j in S.
We accomplish i, j in S by setting high enough a[i] for i in S.
"""


# How do we calculate logit biases needed for the algo?
def get_logit_biases(
    model_name="gpt-3.5-turbo-1106",
    MAXN=299,
    top_logprobs=5,
    bias=40.0,
    anchor_token_ids=None,
    vocab_prefix: int | None = None,
    negative_bias_ids=None,
):
    """
    Generate logit bias vectors for tokens in batches for a specified model.
    This function yields a series of dictionaries, each representing a logit bias vector for a batch of tokens.

    Args:
        model_name (str): The name of the model for which to generate logit biases.

    Yields:
        dict: A dictionary where each key is a token index (as a string) and the value
              is the logit bias (40.0) for that token.
              It has to be a string because the OpenAI API expects a JSON object.

    Example:
        for logit_bias in get_logit_biases():
            print(logit_bias)
    """
    enc = get_model_enc(model_name)
    vocab_size = (
        len(enc.token_byte_values())
        if hasattr(enc, "token_byte_values")
        else len(enc.get_vocab())
    )
    if vocab_prefix is not None:
        vocab_size = vocab_prefix

    special_ids = (
        enc.special_ids if hasattr(enc, "special_ids") else enc.all_special_ids
    )
    # vocab is all except special tokens
    vocab_reduced = [i for i in range(vocab_size) if i not in special_ids]

    negative_logit_bias = None
    if negative_bias_ids is not None:
        negative_logit_bias = {}
        for id in negative_bias_ids:
            # assert id in vocab_reduced
            if id in vocab_reduced:
                continue
            negative_logit_bias[id] = -bias

    if anchor_token_ids is None:
        for i in range(0, len(vocab_reduced), top_logprobs):
            logit_bias = (
                negative_logit_bias.copy() if negative_logit_bias is not None else {}
            )
            for j in range(i, min(i + top_logprobs, len(vocab_reduced))):
                logit_bias[vocab_reduced[j]] = bias
            assert len(logit_bias) <= MAXN
            yield logit_bias
    else:
        reserved: int = len(anchor_token_ids)
        assert reserved < top_logprobs
        for anchor_token_id in anchor_token_ids:
            assert anchor_token_id in vocab_reduced
        for i in range(0, len(vocab_reduced), top_logprobs - reserved):
            logit_bias = (
                negative_logit_bias.copy() if negative_logit_bias is not None else {}
            )
            # If the anchor token or the positive logit bias tokens are in the negative logit bias tokens, we just overwrite them, no need to take special care
            for j in range(i, min(i + top_logprobs - reserved, len(vocab_reduced))):
                logit_bias[vocab_reduced[j]] = bias
            for anchor_token_id in anchor_token_ids:
                logit_bias[anchor_token_id] = bias
            assert len(logit_bias) <= MAXN
            yield logit_bias


async def negative_biases(
    model_name,
    conversation,
    n_tokens_to_downweight: int,
    top_logprobs: int,
    bias: float,
    model_kwargs_hash: str | None = None,
    overwrite_cache=False,
):
    conversations = [conversation]

    negative_ids = set()

    while len(negative_ids) < n_tokens_to_downweight:
        results = await query_single_token(
            model_name,
            model_kwargs_hash=model_kwargs_hash,
            conversations=conversations,
            logit_biases=[{negative_id: -bias for negative_id in negative_ids}]
            * len(conversations),
            top_logprobs=top_logprobs,
            overwrite_cache=overwrite_cache,
        )
        new_negative_ids = [
            str_to_token(model_name).get(top_logprob["token"], None)
            for top_logprob in results[0]["top_logprobs"]
        ]
        new_negative_ids = [
            negative_id for negative_id in new_negative_ids if negative_id is not None
        ]
        if (
            len(new_negative_ids) == 0
            or sum([negative_id in negative_ids for negative_id in new_negative_ids])
            >= (len(new_negative_ids) + 1) // 2
        ):
            print("No more negative ids found at", len(negative_ids))
            break
        negative_ids.update(new_negative_ids)
        print(f"negative_ids: {new_negative_ids}")

    return negative_ids


# %%

SAVE_DIR = Path("logits/")


def heuristic_unintended_topk(diffs: np.ndarray, bias: float):
    """
    If np.max(diffs) - np.min(diffs) is approximately bias, and there are no other values in the array,
    then we take the max.
    Otherwise return None
    """
    n_sig_digits = 3
    tolerance = 10 ** (-n_sig_digits)
    # rounding doesn't work in some cases but those are rare, we can survive without them
    if (
        abs(np.max(diffs) - np.min(diffs) - bias) < tolerance
        and len(np.unique(np.round(diffs, n_sig_digits))) == 2
    ):
        # print("Using heuristic for unintended top-k")
        return np.max(diffs)
    else:
        # print("Discarding token because heuristic for unintended top-k doesn't work")
        return "DIFF_VALUES"


async def query_all_token_logprobs_simple(
    model_name: str,
    prompt: str,
    bias: float,
    n_tokens_to_downweight: int,
    batch_size: int,
    model_kwargs: dict | None = None,
    vocab_prefix: int | None = None,
    top_logprobs: int = 5,
    cuda_device: int = 0,
    overwrite_cache: bool = False,
):
    """
    Queries the model for the difference in log probabilities between each token and a reference token.

    Args:
        model_name (str): The name of the model to query.
        prompt (str): The prompt to query against.

    Returns:
        dict: A dictionary of the difference in log probabilities for each token in the vocab.
    """

    if model_kwargs is None:
        SAVE_PARENT = SAVE_DIR / f"{model_name}_qsimple_a"
        model_kwargs_hash = None
    else:
        model_kwargs_hash: str = hash_params(model_kwargs)
        SAVE_PARENT = SAVE_DIR / f"{model_name}_qsimple_a.{model_kwargs_hash}"

    SAVE_PARENT.mkdir(parents=True, exist_ok=True)

    if model_kwargs is None or (
        "load_in_4bit" not in model_kwargs and "load_in_8bit" not in model_kwargs
    ):
        client = Interactor(
            model_name, **model_kwargs if model_kwargs is not None else {}
        )
        device = f"cuda:{cuda_device}"
        client.singleton_model.to_gpu(device)
    else:
        # this is a horrible hack
        CUDA_VISIBLE_DEVICES: list = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES[int(cuda_device)]
        client = Interactor(
            model_name, **model_kwargs if model_kwargs is not None else {}
        )
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(CUDA_VISIBLE_DEVICES)
        pass

    name = prompt.replace(" ", "_")
    print(f"prompt: {prompt}")

    SAVE_FILE = SAVE_PARENT / f"{name}.json"
    SAVE_PKL = SAVE_FILE.with_suffix(".pkl")

    if SAVE_PKL.exists() and not overwrite_cache:
        print(f"Loading from {SAVE_PKL}")
        return pickle.load(open(SAVE_PKL, "rb"))

    conversation = [{"role": "assistant", "content": prompt}]

    negative_bias_ids = await negative_biases(
        model_name,
        model_kwargs_hash=model_kwargs_hash,
        conversation=conversation,
        n_tokens_to_downweight=n_tokens_to_downweight,
        bias=bias,
        top_logprobs=top_logprobs,
        overwrite_cache=False,
    )

    # Get the logit biases list, with ' a' as the anchor token
    anchor_token_str = " a"
    anchor_token_id = str_to_token(model_name)[anchor_token_str]
    print(f"anchor_token_id: {anchor_token_id}")

    logit_biases_list = list(
        get_logit_biases(
            model_name=model_name,
            top_logprobs=top_logprobs,
            anchor_token_ids=[anchor_token_id],
            bias=bias,
            vocab_prefix=vocab_prefix,
            negative_bias_ids=negative_bias_ids,
        )
    )
    # logit_biases_list = logit_biases_list[:500]

    enc = get_model_enc(model_name)

    # Prepare the batch of prompts with the same content but different logit biases
    conversations = [conversation] * len(logit_biases_list)
    print(f"len(conversations): {len(conversations)}")

    # Query the model
    results = await query_single_token(
        model_name,
        model_kwargs_hash=model_kwargs_hash,
        conversations=conversations,
        logit_biases=logit_biases_list,
        batch_size=batch_size,
        top_logprobs=top_logprobs,
        overwrite_cache=False,
    )

    # Calculate the difference in log probabilities from anchor_token_id token
    """
    Issues that may arise:
    - Tokens that get in top-k without being in logit_biases_list
        - This is like <10 tokens, just ignore them?
        - Unfortunately not a solution that holds for all prompts globally
        - Sol: do negative logit bias on the top ~100 tokens of the prompts
        - What if this doesn't help?
        - Use a heuristic to figure out the right value
    - Tokens that are in logit_biases_list but not in top-k
        - If we solve the above, this doesn't happen
        - We don't get the value for them
        - Set it to dunno the average of the other tokens?
    - Strings in logprobs that don't map to tokens
        - Discard those
    - Tokens that aren't ever represented as strings in logprobs
        - Discard those token_ids from the vector
        - This works for all prompts globally
    """

    logprob_differences = {}
    for logit_bias, result in tqdm(zip(logit_biases_list, results)):
        anchor_logprob = None
        for top_logprob in result["top_logprobs"]:
            if top_logprob["token"] == anchor_token_str:
                anchor_logprob = top_logprob["logprob"]
                break
        assert anchor_logprob is not None
        for top_logprob in result["top_logprobs"]:
            try:
                token_id = str_to_token(model_name)[top_logprob["token"]]
            except KeyError:
                # print("WARNING: token not in str_to_token, discarding:", top_logprob['token'])
                token_id = -1
            if token_id != anchor_token_id:
                if token_id not in logprob_differences:
                    logprob_differences[token_id] = []
                logprob_differences[token_id].append(
                    top_logprob["logprob"] - anchor_logprob
                )

    for token_id, diffs in logprob_differences.items():
        if token_id == -1:
            continue
        # assert they are all the same up to 0.0001
        # assert np.allclose(diffs, diffs[0], atol=1e-4)
        if np.allclose(diffs, diffs[0], atol=1e-3):
            logprob_differences[token_id] = np.median(diffs)
        else:
            # print(f"token: {token_id} {token_to_str(model_name)[token_id]}")
            # print(f"WARNING: diffs for token {token_id} are not all the same.")
            logprob_differences[token_id] = heuristic_unintended_topk(diffs, bias)

        # Use the median of the diffs as a robust central tendency measure

    assert anchor_token_id not in logprob_differences
    logprob_differences[anchor_token_id] = 0.0
    vocab_size = (
        len(enc.token_byte_values())
        if hasattr(enc, "token_byte_values")
        else len(enc.get_vocab())
    )
    # assert len(logprob_differences) == vocab_size

    for token_id in range(vocab_size):
        if token_id not in logprob_differences:
            logprob_differences[token_id] = "NOT_ENCOUNTERED"

    # sort by token id, discard -1
    logprob_differences = {
        k: v
        for k, v in sorted(logprob_differences.items(), key=lambda item: item[0])
        if k != -1
    }

    array_results = np.array(
        [
            logprob_differences[i]
            if isinstance(logprob_differences[i], float)
            else np.nan
            for i in range(vocab_size)
        ]
    )

    save_dict = {
        "prompt": prompt,
        "model_name": model_name,
        "model_kwargs": make_json_serializable(model_kwargs),
        "n_tokens": vocab_size,
        "pivot": anchor_token_str,
        "pivot_id": anchor_token_id,
        "bias": bias,
        "len_negative_bias_ids": len(negative_bias_ids),
        "negative_bias_ids": list(negative_bias_ids),
    }
    save_dict.update(logprob_differences)

    # append the results to a file
    # print(f"Saving to {SAVE_FILE}")
    # display those first
    with open(SAVE_FILE, "w") as f:
        json.dump(save_dict, f, indent=2)

    with open(SAVE_PKL, "wb") as f:
        pickle.dump(array_results, f)

    return array_results


# model_name = "EleutherAI/pythia-160m"
# model_name = "EleutherAI/pythia-70m"
# vec4 = query_all_token_logprobs_simple(model_name, "B")
# vec1 = query_all_token_logprobs_simple(model_name, "To be or not")
# vec2 = query_all_token_logprobs_simple(model_name, "To be")
# vec3 = query_all_token_logprobs_simple(model_name, "A")
# nan_ids_1 = np.isnan(vec1)
# nan_ids_2 = np.isnan(vec2)
# nan_ids_3 = np.isnan(vec3)
# print("lens", len(vec1), len(vec2), len(vec3))
# print("nans", sum(nan_ids_1), sum(nan_ids_2), sum(nan_ids_3))
# print("not all nan nor all number", len(vec1) - sum((nan_ids_1 == nan_ids_2) & (nan_ids_1 == nan_ids_3)))

# vec4 = query_all_token_logprobs_simple("EleutherAI/pythia-160m", "B")
# import code; code.interact(local=dict(globals(), **locals()))
"""
>>> len(vec1)
50277
>>> none_ids_1 = vec1 == None
>>> sum(none_ids_1)
2916
>>> none_ids_2 = vec2 == None
>>> sum(none_ids_2)
2916
>>> none_ids_2 = vec2 == None
>>> sum(none_ids_2)
2915
>>> 50277 - sum((none_ids_1 == none_ids_2) & (none_ids_1 == none_ids_3))
9
"""

# %%
# Usage:
# python query_logprobs.py --model_name EleutherAI/pythia-14m --prompt "To be or not to" --bias 80. --n_tokens_to_downweight 30 --batch_size 4096 --overwrite_cache
