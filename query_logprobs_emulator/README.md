## IMPORTANT NOTE
In response to our vulnerability disclosure, on March 3 2024 or earlier,
OpenAI and Google DeepMind changed their API to make logit bias not affect the logprobs.
Hence, running the original algorithm on those APIs does not work anymore. 
To our best knowledge, no other proprietary model is currently vulnerable to this attack.

No table in the final paper is generated using the code in this directory. 

We discuss logprob-free alternatives in the paper, but all are expected to be significantly more query-intensive than the original algorithm.
Logprob-free attacks (without an existing) are implemented in the [optimize_logit_queries](../optimize_logit_queries) directory.
It is likely other model providers concerned with this attack follow suit.

This means the code in this directory is not of practical or reproducibility interest, 
but only released in case researchers wants a starting point for follow-up work. Feel free to contact the second author of the paper for any questions.

## What is this?
The code in this directory demonstrates the full pipeline from API access to finding the hidden dimension, using a locally hosted small model as an emulated API. 
We do not provide the weight extraction code.

The main usecase of this repo is to play with an emulated API that can be used to optimize attack parameters and methods before running the attack on the real API.
The easier way to do this is to run "queries" a on known logit vector, as is done in `[optimize_logit_queries](../optimize_logit_queries)`. 
However, this has the disadvantage of not testing the full pipeline in terms of API restrictions, tokenization, and other practical issues.

To run the code on the first `N=1000` prompts from `prompts_10000.csv` on `pythia-410m`, execute the following command (warning: it takes a while on bad GPUs):
```bash
python query_logprobs_many.py 
```
Note that the procedure requires `N * vocab_size / (top_logprobs - 1)` queries, so it can take a very long time with the default parameters.
If you want to hurry up this part of the code, set `top_logprobs` to a number closer to `vocab_size`.

Afterwards, to see the SVD of the extracted logit matrix, run
```bash
python svd.py
```
