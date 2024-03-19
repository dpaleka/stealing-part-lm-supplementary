"""
We want: to get a logprob of token X in a sequence, e.g. [prompt, X, ...]
We have: a model that gives us top_logprobs (usually top_logprobs=5) logprobs per token in the sequence.

Just notice: if B = sum(exp(logprob_0[t] for t in vocab)) is the normalizer for logits, then:
1. logprob_0[top_token] = x[top_token] - log(B)
2. We know x[top_token] - x[token] from the method in query_logprobs.py (in fact we know this for top_logprobs - 1 tokens per query)
3. Hence we can compute logprob_0[token] = (x[token] - x[top_token]) + logprob_0[top_token]
"""

from query_logprobs import query_single_token
from utils import str_to_token, token_to_str, get_model_enc
import asyncio
import json

bias = 100


async def get_logprob_of_token(
    model: str,
    tasks: list[dict[str, str | list[int]]],
    top_logprobs=5,
):
    """
    tasks is a list of dicts with keys:
    {
        'prompt': str,
        'token_list': list[int],
        'top_token': {
            'token': str,
            'idx': int,
            'logprob': float
        } | None
    }
    Describing prompts and tokens to query logprobs for.

    query_single_token takes:
        model : str,
        conversations : list[list[dict[str, str]]],
        logit_biases : list[dict[int, float]] | None = None

    query_single_token returns a list of:
    {
        'token': token_prettify(self.tokenizer, next_token_id),
        'logprob': round(float(logprobs[next_token_id]), 7),
        'top_logprobs': [
            {
            'token': token_prettify(self.tokenizer, top_indices[i]),
            'logprob': round(float(top_logprobs_tensor[i]), 7),
            } for i in range(top_logprobs_tensor.shape[0])]
    }
    """

    if any(task["top_token"] is None for task in tasks):
        double_call_tasks = [
            task for task in tasks if task["top_token"] is None
        ]  # these are references, which is nice, we'll mutate them
        # If you want to skip this steop, precompute the top_token dict for each task
        first_query_results = await query_single_token(
            model=model,
            conversations=[
                [{"role": "assistant", "content": task["prompt"]}]
                for task in double_call_tasks
            ],
            top_logprobs=top_logprobs,
        )
        # print("first_query_results", first_query_results)
        for i, (query_result, task) in enumerate(
            zip(first_query_results, double_call_tasks)
        ):
            print(i)
            print(json.dumps(query_result, indent=2))
            task["top_token"] = {
                "token": query_result["token"],
                "idx": str_to_token(model)[query_result["token"]],
                "logprob": query_result["top_logprobs_dict"][query_result["token"]],
            }

    logit_biases = [
        {token_idx: bias for token_idx in task["token_list"]} for task in tasks
    ]
    for logit_bias, task in zip(logit_biases, tasks):
        logit_bias[task["top_token"]["idx"]] = bias

    # if there exists a task where top_token dict is None, then we need to query for it

    query_results = await query_single_token(
        model=model,
        conversations=[
            [{"role": "assistant", "content": task["prompt"]}] for task in tasks
        ],
        logit_biases=logit_biases,
        top_logprobs=top_logprobs,
    )

    ret = []
    for i, (query_result, task) in enumerate(zip(query_results, tasks)):
        res = {}
        print()
        print(json.dumps(query_result, indent=2))
        print(f"{query_result['token'] = }")
        # print(f"{task['top_token']['token'] = }")
        # print(f"{logit_biases[i] = }")
        # print(f"{[token_to_str(model)[token] for token in logit_biases[i].keys()] = }")
        # print(f"{[token_to_str(model)[token] for token in task['token_list']] = }")
        assert (
            query_result["token"] == task["top_token"]["token"]
        )  # this fails for some reason
        for token in task["token_list"]:
            str_token = token_to_str(model)[token]
            print(f"token: {str_token}")
            # u = query_result['top_logprobs'][token]['logprob']
            u = query_result["top_logprobs_dict"][str_token]
            logprob = u - query_result["logprob"] + task["top_token"]["logprob"]
            res[str_token] = {
                "logprob": logprob,
                "logit_diff_from_top": u - query_result["logprob"],
                "idx": token,
            }
        ret.append(res)
    return ret


async def get_logprob_of_sequence(
    model: str,
    tasks: list[dict[str, str | list[int]]],
    top_logprobs: int = 5,
) -> list[dict]:
    """
    For each task, calculates the log probability of the sequence of tokens specified in 'sentence'.
    The 'sentence' is tokenized and the log probability of each token given the previous tokens is queried.

    Args:
        model: The model identifier string.
        tasks: A list of tasks, where each task is a dictionary with a 'sentence' key.
        top_logprobs: The number of top log probabilities to consider.

    Returns:
        A list of dictionaries, each containing the log probabilities for the tokens in the sequence.
    """

    # Define a tokenizer function
    def tokenize(sentence: str) -> list[str]:
        # This is a placeholder tokenizer function
        # Replace with actual model-specific tokenization logic
        tokenizer = get_model_enc(model)
        return tokenizer.encode(sentence)

    # Tokenize the sentences and create new tasks for each prefix
    new_tasks = []
    for task in tasks:
        sentence = task["sentence"]
        tokenized_sentence = tokenize(sentence)
        for i in range(len(tokenized_sentence)):
            prefix = tokenized_sentence[:i]
            token_list = [tokenized_sentence[i]]
            [token_to_str(model)[token] for token in token_list]
            prefix_str_list = [token_to_str(model)[token] for token in prefix]
            # print("token_str_list", token_str_list)
            new_tasks.append(
                {
                    "prompt": "".join(prefix_str_list),
                    "token_list": token_list,
                    "top_token": None,
                }
            )
            print(f"new task: {new_tasks[-1]}")

    # Get log probabilities for the new tasks
    logprob_results = await get_logprob_of_token(model, new_tasks, top_logprobs)
    return logprob_results



# Write a test for this
if __name__ == "__main__":
    # model = 'gpt-3.5-turbo-instruct'
    model = "EleutherAI/pythia-70m"
    tasks = [
        {
            "prompt": "The quick",
            "token_list": [str_to_token(model)["fox"], str_to_token(model)["dog"]],
            "top_token": None,
        },
        {
            "prompt": "The quick brown fox jumps over the lazy dog.",
            "token_list": [str_to_token(model)[" The"]],
            "top_token": None,
        },
    ]

    tasks_2 = [
        #        {
        #            'sentence': "The quick brown fox jumps over the lazy dog.",
        #        },
        {
            "sentence": "The",
        },
    ]

    # run with asyncio.run(get_logprob_of_token_in_sequence(model, tasks))
    # ret = asyncio.run(get_logprob_of_token(model, tasks))
    ret = asyncio.run(get_logprob_of_sequence(model, tasks_2))
    print(json.dumps(ret, indent=2))
