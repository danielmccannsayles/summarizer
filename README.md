Exploring benchmarking AI summarization.

The `writer_summaries` data is taken from the paper below.

It's a list of pairs of articles & their summaries (written by freelance writers).

`explore.ipynb` is just exploring the `writer_summaries` data.

`eval.py` runs a simple inspectAI flow. Given a solver (a summarizer) we run it on the data

1. Add OPENAI_API_KEY in .env

Using data from this paper:
@misc{https://doi.org/10.48550/arxiv.2301.13848,
url = {https://arxiv.org/abs/2301.13848},
author = {Zhang, Tianyi and Ladhak, Faisal and Durmus, Esin and Liang, Percy and McKeown, Kathleen and Hashimoto, Tatsunori B.},
title = {Benchmarking Large Language Models for News Summarization},
publisher = {arXiv},
year = {2023},
}
