# On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization

This repository is the official PyTorch implementation of paper: On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization.

**Jiancong Xiao, Ziniu Li, Xingyu Xie, Emily Getzen, Cong Fang, Qi Long, Weijie J. Su**

**arXiv:** [https://arxiv.org/abs/2405.16455](https://arxiv.org/abs/2405.16455) 

## Abstract

Accurately aligning large language models (LLMs) with human preferences is crucial for informing fair, economically sound, and statistically efficient decision-making processes. However, we argue that reinforcement learning from human feedback (RLHF)—the predominant approach for aligning LLMs with human preferences through a reward model—suffers from an inherent algorithmic bias due to its Kullback–Leibler-based regularization in optimization. In extreme cases, this bias could lead to a phenomenon we term preference collapse, where minority preferences are virtually disregarded. To mitigate this algorithmic bias, we introduce preference matching (PM) RLHF, a novel approach that provably aligns LLMs with the preference distribution of the reward model under the Bradley–Terry–Luce/Plackett–Luce model. Central to our approach is a PM regularizer that takes the form of the negative logarithm of the LLM’s policy probability distribution over responses, which helps the LLM balance response diversification and reward maximization. Notably, we obtain this regularizer by solving an ordinary differential equation that is necessary for the PM property. For practical implementation, we introduce a conditional variant of PM RLHF that is tailored to natural language generation. Finally, we empirically validate the effectiveness of conditional PM RLHF through experiments on the OPT-1.3B and Llama-2-7B models, demonstrating a 29% to 41% improvement in alignment with human preferences, as measured by a certain metric, compared to standard RLHF.

## Main Result
1. Standard RLHF is biased. Its algorithmic bias inherits from the reference model.

2. PM RLHF precisely aligns with RM preferences. This is mathematically provable and experimentally corroborated.

## Code

Throughout the experimental process, each stage of RLHF follows the standardized configurations established in the [DeepSpeed-Chat framework](https://github.com/microsoft/DeepSpeed).

## Citation
```
@article{xiao2024algorithmic,
  title={On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization},
  author={Xiao, Jiancong and Li, Ziniu and Xie, Xingyu and Getzen, Emily and Fang, Cong and Long, Qi and Su, Weijie J},
  journal={arXiv preprint arXiv:2405.16455},
  year={2024}
}
```
