# %%
import asyncio

from tqdm import tqdm

from query_logprobs import (  # noqa
    get_logit_biases,
    heuristic_unintended_topk,
    negative_biases,
    query_all_token_logprobs_simple,
    query_and_print_logprobs,
    query_single_token,
    query_single_token_local,
    query_single_token_openai_chat,
    query_single_token_openai_completion,
)
from utils import (  # noqa
    format_float,
    get_model_enc,
    get_token_ids,
    is_completion,
    is_openai,
    print_logprob_dict,
    remove_keys,
    str_to_token,
    stringify_params,
    token_prettify,
    token_to_str,
)


# %%
def get_logit_biases_toy(model_name="gpt-3.5-turbo-1106"):
    # return a list of dcts saying what logit bias vector to use for which batch of tokens
    # for tokens "happy", "sad", "neutral"

    # Adjust these for your use-case
    logit_bias_values = {
        " Paris": get_token_ids(model_name, "Paris")[0],
        " San": get_token_ids(model_name, "San")[0],
        " yes": get_token_ids(model_name, "yes")[0],
        " no": get_token_ids(model_name, "no")[0],
        " a": get_token_ids(model_name, "a")[0],
    }
    # make a dict of logit biases for each token in the prompt
    logit_bias = {}
    for token in logit_bias_values:
        logit_bias[logit_bias_values[token]] = 40.0
    return [logit_bias]


# print(get_logit_biases_toy())
# %%
def test_pythia():
    model_name = "EleutherAI/pythia-160m"
    conversations = [
        [
            {
                "role": "assistant",
                "content": "The capital of France is Paris. The capital of Germany is",
            }
        ],
        [
            {
                "role": "assistant",
                "content": "The capital of France is Paris. The capital of Italy is",
            }
        ],
    ] * 20

    # Call the new async function and use asyncio.run to execute it
    asyncio.run(query_and_print_logprobs(model_name, conversations))


# test_pythia()


# %%
def test_openai_chat():
    model_name = "gpt-3.5-turbo-1106"
    conversations = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "How do you feel?"},
        ],
        [
            {"role": "system", "content": ""},
            {
                "role": "user",
                "content": "The capital of France is Paris. The capital of Germany is",
            },
        ],
    ] * 2
    logit_biases = get_logit_biases_toy() * len(conversations)
    asyncio.run(query_and_print_logprobs(model_name, conversations, logit_biases))


# test_openai_chat()
# %%
def test_pythia_2():
    model_name = "EleutherAI/pythia-160m"
    bias = 1000.0
    logit_biases_list = list(
        get_logit_biases(
            model_name=model_name,
            top_logprobs=5,
            bias=bias,
            anchor_token_ids=[get_token_ids(model_name, " a")[0]],
        )
    )

    tokenizer = get_model_enc(model_name)
    conversations = [
        [
            {
                "role": "assistant",
                "content": "The capital of France is Paris. The capital of Germany is",
            }
        ],
        [
            {
                "role": "assistant",
                "content": "The capital of France is Paris. The capital of Italy is",
            }
        ],
    ] * 20
    logit_biases_list = logit_biases_list[: len(conversations)]
    results = asyncio.run(
        query_single_token(model_name, conversations, logit_biases=logit_biases_list)
    )

    for logit_bias, result in tqdm(zip(logit_biases_list, results)):
        pretty_tokens = [
            token_prettify(tokenizer, token_id) for token_id in logit_bias.keys()
        ]
        print(logit_bias)
        print("logit bias:", pretty_tokens)
        print_logprob_dict(result, model=model_name)
        print()

    return logit_biases_list, results


# test_pythia_2()


# %%
def test_pythia_3():
    model_name = "EleutherAI/pythia-160m"
    bias = 1000.0
    conversations = [
        [
            {
                "role": "assistant",
                "content": "The capital of France is Paris. The capital of Italy is",
            }
        ],
    ]

    logit_biases_list = list(
        get_logit_biases(
            model_name=model_name,
            top_logprobs=5,
            bias=bias,
            anchor_token_ids=[get_token_ids(model_name, " a")[0]],
        )
    )

    logit_biases_list = [logit_biases_list[39]]
    results = asyncio.run(
        query_single_token(model_name, conversations, logit_biases=logit_biases_list)
    )

    tokenizer = get_model_enc(model_name)
    for logit_bias, result in tqdm(zip(logit_biases_list, results)):
        print(logit_bias)
        pretty_tokens = [
            token_prettify(tokenizer, token_id) for token_id in logit_bias.keys()
        ]
        print("logit bias:", pretty_tokens)
        print_logprob_dict(result, model=model_name)
        print()

    return logit_biases_list, results


# test_pythia_3()


# %%
def test_correct_batching_pythia():
    logit_biases_list_2, results_2 = test_pythia_2()
    logit_biases_list_2, results_2 = logit_biases_list_2[39:40], results_2[39:40]
    logit_biases_list_3, results_3 = test_pythia_3()

    assert logit_biases_list_2 == logit_biases_list_3
    print("results_2:", results_2)
    print("results_3:", results_3)

    # also check manually if something is off, whether pretty_tokens and the top tokens in print_logprob_dict match


# test_correct_batching_pythia()


# %%


def test_pythia_4():
    model_name = "EleutherAI/pythia-160m"
    bias = 1000.0
    prompt = "The capital of France is Paris. The capital of Italy is"
    conversation = [{"role": "assistant", "content": prompt}]
    n_tokens_to_downweight = 100

    negative_ids = negative_biases(
        model_name, conversation, n_tokens_to_downweight, bias, overwrite_cache=False
    )

    logit_biases_list = list(
        get_logit_biases(
            model_name=model_name,
            top_logprobs=5,
            bias=bias,
            anchor_token_ids=[get_token_ids(model_name, " a")[0]],
            negative_bias_ids=negative_ids,
        )
    )
    logit_biases_list = [logit_biases_list[39]]

    results = asyncio.run(
        query_single_token(
            model_name,
            [[{"role": "assistant", "content": prompt}]],
            logit_biases=logit_biases_list,
        )
    )

    tokenizer = get_model_enc(model_name)
    for logit_bias, result in tqdm(zip(logit_biases_list, results)):
        print(logit_bias)
        pretty_tokens = [
            token_prettify(tokenizer, token_id) for token_id in logit_bias.keys()
        ]
        print("logit bias:", pretty_tokens)
        print_logprob_dict(result, model=model_name)
        print()

    return logit_biases_list, results


# test_pythia_4()
