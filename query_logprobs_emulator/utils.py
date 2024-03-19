import json
from functools import lru_cache
from copy import deepcopy

import hashlib
import tiktoken
import transformers


def format_float(x) -> str:
    return "{:.3f}".format(x)


def is_openai(model: str) -> bool:
    keywords = [
        "gpt-4",
        "gpt-3.5-turbo",
        "ada",
        "babbage",
        "curie",
        "davinci",
        "instruct",
    ]
    return any(keyword in model for keyword in keywords)


def is_llama2_chat(model: str) -> bool:
    keywords = ["llama-2", "pythia"]
    return any(keyword in model for keyword in keywords)


def is_completion(model: str) -> bool:
    """
    This is very hacky. We should probably use a list of models that are completion models.
    """
    keywords = [
        "ada",
        "babbage",
        "curie",
        "davinci",
        "gpt-3.5-turbo-instruct",
        "llama-2-hf",
        "pythia",
    ]
    return any(keyword in model for keyword in keywords)


def format_msgs(model_name, messages: list) -> str:
    if "lama-2" in model_name and model_name.endswith("chat-hf"):
        raise NotImplementedError("LLaMA chat format not implemented yet.")

    if is_openai(model_name) and not is_completion(model_name):
        raise ValueError("You send messages to the model, not a string.")

    # print("messages:", messages)
    if len(messages) == 1:
        return messages[0]["content"]
    elif len(messages) > 1 and messages[-1]["role"] == "assistant":
        return (
            "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages[:-1]])
            + "\n"
            + "assistant: "
            + messages[-1]["content"]
        )
    else:
        return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])


def token_prettify_decode(tokenizer, token_id):
    """
    This might be what the OpenAI logprobs API does (not sure), but printing these tokens crashes my terminal.
    """
    ret = tokenizer.decode([token_id])
    #    print("token_prettify", token_id, ret) # crashes my terminal
    return ret


def token_prettify(tokenizer, token_id):
    """
    Let's just use this one. We lose some tokens but who cares.
    """
    if isinstance(tokenizer, tiktoken.Encoding):
        converted = [tokenizer.decode([token_id])]
    else:
        converted = tokenizer.convert_ids_to_tokens([token_id])
    # print(f"token_id: {token_id}, converted: {converted}")
    assert len(converted) == 1
    replaced = converted[0].replace("Ġ", " ").replace("\u2581", " ").replace("Ċ", "\n")
    # print("token_id", token_id, "converted", converted, "replaced", replaced)
    return replaced


def remove_keys(dct, keys):
    """
    Go recursively through a dictionary. Return a new dictionary without the specified keys.
    """
    if isinstance(dct, dict):
        return {k: remove_keys(v, keys) for k, v in dct.items() if k not in keys}
    elif isinstance(dct, list):
        return [remove_keys(x, keys) for x in dct]
    else:
        return dct


@lru_cache(maxsize=128)
def get_model_enc(model_name):
    print(f"Loading tokenizer for {model_name}...")
    if "gpt-4" in model_name:
        model_enc = tiktoken.encoding_for_model("gpt-4")
    elif "gpt-3.5-turbo" in model_name:
        model_enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
    elif (
        "ada" in model_name
        or "babbage" in model_name
        or "curie" in model_name
        or "davinci" in model_name
    ):
        model_enc = tiktoken.encoding_for_model("davinci")
    elif "llama-2" in model_name:
        model_enc = transformers.AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf"
        )
    elif "pythia" in model_name:
        print("Loading Pythia tokenizer...")
        model_enc = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-160m")
    else:
        raise ValueError("Model family not recognized.")
    return model_enc


@lru_cache(maxsize=128)  # cache returns mutated objects
def token_to_str(model_name) -> dict[int, str]:
    """
    Return a dictionary mapping token IDs to token strings.
    """
    enc = get_model_enc(model_name)
    vocab_size = (
        len(enc.token_byte_values())
        if hasattr(enc, "token_byte_values")
        else len(enc.get_vocab())
    )
    ret = {}
    print("Construct and sanity check token<->string mapping...")
    for token_id in range(vocab_size):
        # token_str = enc.decode([token_id]) #was wrong for pythia, some results might be off on some tokens
        token_str = token_prettify(enc, token_id)
        ret[token_id] = token_str

        if token_str == ",":
            print(f"comma token {token_id}: '{token_str}'")
        if token_str == " ,":
            print(f"space comma token {token_id}: '{token_str}'")
    return ret


@lru_cache(maxsize=128)
def str_to_token(model_name) -> dict[str, int]:
    """
    Return a dictionary mapping token strings to token IDs.
    """
    forward = token_to_str(model_name)
    return {v: k for k, v in forward.items()}


@lru_cache(maxsize=128)
def get_token_ids(model_name, text):
    model_enc = get_model_enc(model_name)
    return model_enc.encode(text)


def print_logprob_dict(logprob_dict, model):
    print(f"Token: {logprob_dict['token']}")
    for top_logprob in logprob_dict["top_logprobs"]:
        token_id = get_token_ids(model_name=model, text=top_logprob["token"])[0]
        print(f"  {top_logprob['logprob']:.5f} {top_logprob['token']} {token_id}")
    print()


def stringify_params(*args, **kwargs):
    # This function we don't modify, it's the hash for caching
    args_stringified = tuple(json.dumps(arg, sort_keys=True) for arg in args)
    kwargs_stringified = {
        key: json.dumps(value, sort_keys=True) for key, value in kwargs.items()
    }
    return (args_stringified, tuple(sorted(kwargs_stringified.items())))


def json_serializable(value):
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False


def make_json_serializable(value):
    if isinstance(value, dict):
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [make_json_serializable(v) for v in value]
    elif isinstance(value, tuple):
        return tuple(make_json_serializable(v) for v in value)
    elif not json_serializable(value):
        return str(value)
    return value


def hash_params(*args, **kwargs):
    # Copy the arguments so we don't modify them
    args = deepcopy(args)
    kwargs = deepcopy(kwargs)

    # Make all values JSON serializable
    args = tuple(make_json_serializable(arg) for arg in args)
    kwargs = {key: make_json_serializable(value) for key, value in kwargs.items()}

    # Stringify the arguments
    str_args, str_kwargs = stringify_params(*args, **kwargs)
    return hashlib.md5(str(str_args).encode() + str(str_kwargs).encode()).hexdigest()[
        0:8
    ]
    # return hashlib.md5(str_params.encode()).hexdigest()[0:8]
