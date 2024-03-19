# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv

load_dotenv()

sns.set_style("darkgrid")

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
auth_token = os.environ.get("HF_TOKEN")


# Initialize tokenizer and model
model_id = "meta-llama/Llama-2-7b-hf"
# model_id = "EleutherAI/pythia-160m"
# model_id = "google/gemma-2b"
# model_id = "EleutherAI/pythia-70m"
tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    truncation_side="left",
    padding_side="right",
)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_id,  # load_in_8bit=True,
    device_map={"": 0},
)

# %%
import pandas as pd

df = pd.read_csv("distribution_logits/prompts_small.csv")
df.columns = ["prompt"]
prompts = df["prompt"].tolist()
# first 10
prompts = prompts[:10]
print(prompts)


# %%
def prompt_to_tokens(tokenizer, system_prompt, instruction, model_output):
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    dialog_content = B_SYS + system_prompt + E_SYS + instruction.strip()
    dialog_tokens = tokenizer.encode(
        f"{B_INST} {dialog_content.strip()} {E_INST} {model_output.strip()}"
    )
    return torch.tensor(dialog_tokens).unsqueeze(0)


system_prompt = "You are a helpful, honest and concise assistant."

logits_list = []
for prompt in prompts:
    # Tokenize the prompt
    if "chat-hf" in model_id:
        inputs = prompt_to_tokens(tokenizer, system_prompt, prompt, "")
    else:
        inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0)

    print(f"\n{tokenizer.decode(inputs[0])}")

    # print(inputs) # just a tensor
    # Move tensors to the same device as the model
    inputs = inputs.to(model.device)

    # Get logits from the model
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
        logits_list.append(logits)

print(
    logits_list[0].shape
)  # (1, 8, 50264) means (batch_size, sequence_length, vocab_size)


# %%
# We only care about the difference of the max token to all others
def additive_normalize(logits):
    mx = logits.max()
    logits = logits - mx
    return logits


def unit_normalize(logits):
    mx = logits.max()
    mn = logits.min()
    logits = (logits - mn) / (mx - mn)
    return logits


def onesided_unit_normalize(logits, width=40.0):
    print(
        "Max:",
        logits.max().item(),
        "Actual width:",
        (logits.max() - logits.min()).item(),
    )
    mx = logits.max()
    mn = mx - width
    logits = (logits - mn) / (mx - mn)
    return logits


# %%
# Get the logits for the last token in each sequence
last_token_logits = [logits[:, -1, :] for logits in logits_list]
# get sorted
# last_token_logits = [additive_normalize(logits) for logits in last_token_logits]
sorted_last_token_logits = [
    logits.sort(descending=True) for logits in last_token_logits
]
print(sorted_last_token_logits)

# convert to numpy
last_token_logits = np.array([logits[0].cpu().numpy() for logits in last_token_logits])
sorted_last_token_logits = np.array(
    [
        (logits[0].cpu().numpy(), logits[1].cpu().numpy())
        for logits in sorted_last_token_logits
    ]
)

print(last_token_logits.shape)


# %%
# Save these to a file
import pickle
from pathlib import Path

filename = Path(f"data/{model_id}-last_token_logits.pkl")
filename.parent.mkdir(parents=True, exist_ok=True)
with open(filename, "wb") as f:
    pickle.dump(last_token_logits, f)

# %%

# %%
for i in range(len(prompts)):
    print(f"Top and worst 15 tokens for prompt: {prompts[i]}")
    print(tokenizer.convert_ids_to_tokens(sorted_last_token_logits[i][1][0][:15]))
    print(sorted_last_token_logits[i][0][0][:15])
    print(tokenizer.convert_ids_to_tokens(sorted_last_token_logits[i][1][0][-15:]))
    print(sorted_last_token_logits[i][0][0][-15:])

# %%
# plot the full distribution of logits for each prompt
plt.figure(figsize=(20, 20))
for i in range(len(prompts)):
    # there are 16 prompts
    plt.subplot(8, 2, i + 1)
    plt.title(prompts[i])
    # vspace between plots
    plt.subplots_adjust(hspace=0.5)
    plt.hist(last_token_logits[i].flatten(), bins=100, density=True)
    plt.xlim(0, 1)
    # now plot the gaussian of best fit
    # get the mean and std
    xmin, xmax = plt.xlim()
    # mu, std = np.mean(last_token_logits[i].cpu().numpy().flatten()), np.std(last_token_logits[i].cpu().numpy().flatten())
    import scipy.stats as stats

    mu, std = stats.norm.fit(
        last_token_logits[i].flatten()
    )  # this is the actual mu and std
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, "k", linewidth=2, label="Gaussian fit")
    import scipy.stats as stats

    print(f"Mu: {mu}, Std: {std}")

    # now a bit sharper gaussian fit, without outliers
    OUTLIERS = int(0.01 * len(last_token_logits[i].flatten()))
    no_outliers_logits = sorted_last_token_logits[i][0][0][OUTLIERS:-OUTLIERS]
    mu, std = stats.norm.fit(
        no_outliers_logits.flatten()
    )  # this is the actual mu and std
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    plt.plot(x, p, "r", linewidth=2, label="No outliers Gaussian fit")

    plt.legend()

# title
plt.suptitle(
    "Distribution of logits for each prompt, normalizes so that max is 1 and max-30 is 0"
)
fig = plt.gcf()
plt.show()
plt.draw()
fig.savefig("llama2-7b-chat-hf.png", dpi=100)

# %%
