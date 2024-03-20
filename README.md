## Supplementary code for [Stealing Part of a Production Language Model](https://arxiv.org/abs/2403.06634), Carlini et al., 2024

### `optimize_logit_queries`
Implements *logprob-free* attacks on a known vector of logits (no API calls).
Use `run_attacks.py` to access all implemented attacks, described in the paper in varying levels of detail. 
Running `run_attacks.py` without modification runs several methods on a small random vector of logits.
This directory is useful as a starting point for further research on logprob-free attacks.

### Other, less directly useful code
#### `query_logprobs_emulator`.
Emulates the OpenAI API to test any logprob attack on a local model, as well as possible mitigation strategies.
(Could also use it on the OpenAI API before March 3 2024, but not anymore.)
Defaults to the logprob attack that uses `top_logprobs - 1` tokens per query.
May be useful for research on mitigations and ways to bypass them.

#### `distribution_logits`
Very unpolished script investigating the distribution of logits over the vocabulary for various open-source models.
Pythia seems to be an outlier with very low probabilities on the long tail of the vocabulary.

#### `openai_api_intricacies`
Verifying undocumented properties of `logit_bias` in the OpenAI API that are necessary for the attack to work as described in the paper.



## Disclaimer
Note: this repo does not contain any parameters of OpenAI or Google proprietary models, 
nor any code that can directly extract weights from any API known to the authors.
