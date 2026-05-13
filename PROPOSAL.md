### Introduction

Erdogan et al. (2026) introduces an information-theoretic perspective on LLM tokenizers as structured compressors. Notably, it analyzes how well tokenizers fare at compressing different types of corpora, such as English and Code (Figure 3), and studies the conditional entropy structure of tokens under varying training sizes (Figure 4). It further introduces capacity-utilization metrics for understanding how effectively a tokenizer uses its vocabulary (Figure 8). Our project aims to extend these perspectives to SuperBPE, and expand on the entropy and capacity metrics introduced to further explain the efficacy and properties of SuperBPE.

### Goals/Expected Outcomes

We will reproduce core experiments from the information-theoretic paper and recreate the corresponding figures using standard BPE as a baseline and SuperBPE as the main tokenizer of interest. Because SuperBPE is trained in two phases, with the second phase permitting merges across whitespace boundaries, we will also ablate the phase-2 fractional vocabulary budget and study how it affects the resulting tokenizer. First, we will analyze the compression ratio as tokenizer training size varies. Based on Figure 1 of Liu et al. (2025), we expect that a larger phase-2 budget will have better compression. Next, we will analyze the unigram to 5-gram entropy of the tokenized text across varying training sizes. We hypothesize that larger phase-2 budgets will exhibit higher unigram entropy and lower higher-order conditional entropy because there will be a more diverse set of token types, and phrase structure is absorbed into the tokens respectively. Finally, we will measure capacity-utilization as training size varies as another way to interpret the entropy metrics. This will help evaluate under what conditions SuperBPE utilizes its fixed vocabulary best.

### First Milestone

We will first perform the compression-ratio experiment described above and produce its corresponding figure. This will require setting up the tokenizer training pipeline. Namely, we want to make sure we are able to train a SuperBPE tokenizer, and also recreate the setup from the information-theoretic paper by using the same sized C4 and CodeParrot training sets with 10^8/10^6 character train/test splits). We will also determine the phase-2 fractional vocabulary budgets to ablate by choosing 2-3 based on the values shown in Figure 1 of the SuperBPE paper.

### Project Details

We did not find an official repository for information-theoretic paper, but the metrics are described clearly enough to produce our own testing code. The C4 dataset [https://huggingface.co/datasets/allenai/c4](https://huggingface.co/datasets/allenai/c4) is 305GB, so we will stream until our desired size of approximately 10^8 characters. We will use the HuggingFace Tokenizers library for our BPE implementation, and the official SuperBPE repository for the SuperBPE implementation. Training a tokenizer at this scale is lightweight and computationally feasible on personal machines.  In addition to reproducing the main figures, we hope to provide further interpretation of the entropy and capacity-utilization results, which is not explored in as much depth in the original paper.  

### Related Work/Bibliography

\[1\] A. Liu, J. Hayase, V. Hofmann, S. Oh, N. A. Smith, and Y. Choi, "SuperBPE: Space travel for language models," in *Proc. 2nd Conf. Language Modeling (COLM)*, 2025\. \[Online\]. Available: [https://arxiv.org/abs/2503.13423](https://arxiv.org/abs/2503.13423)

\[2\] M. Erdogan, A. Gorle, S. Chandak, M. Pilanci, and T. Weissman, "An information-theoretic perspective on LLM tokenizers," *arXiv preprint arXiv:2601.09039*, 2026\. \[Online\]. Available: [https://arxiv.org/abs/2601.09039](https://arxiv.org/abs/2601.09039)