# %%
"""
    Minimal singleton class that can be used to query the model.
    Import the model and tokenizer from transformers. 
    Load the model and tokenizer from the model name the first time the class is instantiated.
    The purpose of this class is to avoid loading the model and tokenizer multiple times.

    We also implement logprobs, top_logprobs, and logit_bias as arguments to the query method.
    All models are transformers.AutoModelForCausalLM.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import print_logprob_dict, token_prettify, format_msgs

# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


class SingletonModel:
    """
    Doesn't support multiple models with the same name but different kwargs.
    """

    _instances = {}
    model_name = None

    def __new__(cls, model_name, **kwargs):
        if model_name not in cls._instances:
            print(f"Loading model {model_name}...")
            print("kwargs:", kwargs)
            instance = super().__new__(cls)
            instance.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
            instance.tokenizer = AutoTokenizer.from_pretrained(model_name, **kwargs)
            cls._instances[model_name] = instance
        return cls._instances[model_name]

    def to_gpu(self, device: str | None = None):
        if device is None:
            device = "cuda"
        self.model.to(device)
        return self

    def to_cpu(self):
        self.model.to("cpu")
        return self


# %%
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16,
}
# model = SingletonModel("meta-llama/Llama-2-7b-hf", load_in_4bit=True)
# model.to_gpu()
# print(SingletonModel._instances)

# %%


class Interactor:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.singleton_model = SingletonModel(model_name, **kwargs)
        self.model = self.singleton_model.model
        self.tokenizer = self.singleton_model.tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def format_msgs(self, messages: list) -> str:
        return format_msgs(self.model_name, messages)

    def query_single_token(
        self,
        prompts: list[str],
        logprobs=True,
        top_logprobs=5,
        logit_biases: list[dict[int, float]] | None = None,
        temperature=0.0,
    ):
        if all(len(prompt) == 0 for prompt in prompts):
            # Hope this fixed the first token issue
            inputs = self.tokenizer(
                [self.tokenizer.pad_token] * len(prompts),
                return_tensors="pt",
                padding=True,
            ).to(device=self.model.device)
        else:
            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
                device=self.model.device
            )

        output = self.model(inputs.input_ids, return_dict=True, output_attentions=False)
        results = []
        for i, prompt in enumerate(prompts):
            logits = output.logits[i, -1, :]

            if logit_biases is not None:
                for token, bias in logit_biases[i].items():
                    logits[token] += bias

            logits.softmax(dim=-1)
            logprobs = logits.log_softmax(dim=-1)
            top_logprobs_tensor, top_indices = logprobs.topk(k=top_logprobs, dim=-1)

            if temperature == 0.0:
                next_token_id = top_indices[0]
            else:
                raise NotImplementedError("Sampling not implemented yet.")

            # Construct the result for the current prompt
            result = {
                "token": token_prettify(self.tokenizer, next_token_id),
                "logprob": round(float(logprobs[next_token_id]), 7),
                "top_logprobs": [
                    {
                        "token": token_prettify(self.tokenizer, top_indices[i]),
                        "logprob": round(float(top_logprobs_tensor[i]), 7),
                    }
                    for i in range(top_logprobs_tensor.shape[0])
                ],
            }
            result["top_logprobs_dict"] = {
                token["token"]: token["logprob"] for token in result["top_logprobs"]
            }
            results.append(result)

        return results

    def logprob_sequence(self, prompts, debug=False):
        """
        Calculate the sum of log probabilities for each given sequence in a list.

        Args:
            prompts (list[str]): The input sequences for which the log probabilities are to be calculated.

        Returns:
            list[float]: The sums of log probabilities for the given sequences.
        """
        # add a pad token to the beginning of each sequence
        prompts = [self.tokenizer.pad_token + prompt for prompt in prompts]
        print(f"Prompts: {prompts}")
        inputs = self.tokenizer(prompts, return_tensors="pt", padding=True).to(
            device=self.model.device
        )
        output = self.model(**inputs, return_dict=True)
        print(output.logits.shape)
        logprobs = output.logits.log_softmax(dim=-1)
        print(logprobs.shape)
        input_ids = inputs.input_ids
        sequence_logprob_sums = np.zeros(len(prompts))
        for i, prompt in enumerate(prompts):
            if debug:
                print(f"Prompt: {prompt}")
            for j in range(len(input_ids[i])):
                if input_ids[i][j] != self.tokenizer.pad_token_id:
                    if debug:
                        print(
                            f"{token_prettify(self.tokenizer, input_ids[i][j])} : {logprobs[i, j-1, input_ids[i][j]].item():.2f}"
                        )
                    sequence_logprob_sums[i] += logprobs[
                        i, j - 1, input_ids[i][j]
                    ].item()

        return sequence_logprob_sums


# %%
if __name__ == "__main__":
    # api = Interactor("meta-llama/Llama-2-7b-hf", **quantization_config)
    # api = Interactor("meta-llama/Llama-2-7b-hf")
    api = Interactor("EleutherAI/pythia-160m")
    api.model.to("cuda")
    prompts = [
        "The capital of France is Paris. The capital of Germany is",
        "The capital of France is Paris. The capital of",
        "The capital of France is Paris. The capital of Italy",
    ]
    output = api.query_single_token(prompts, top_logprobs=5)

    for i, prompt in enumerate(prompts):
        print(f"Prompt: {prompt}")
        print_logprob_dict(output[i], model=api.model_name)
        print()

    print(api.logprob_sequence(prompts, debug=True).round(2))


# %%
