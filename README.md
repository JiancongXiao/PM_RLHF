# On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization

This repository is the official PyTorch implementation of paper: On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization.

**Jiancong Xiao, Ziniu Li, Xingyu Xie, Emily Getzen, Cong Fang, Qi Long, Weijie J. Su**

**arXiv:** [https://arxiv.org/abs/2405.16455](https://arxiv.org/abs/2405.16455) 

## Abstract

Accurately aligning large language models (LLMs) with human preferences is crucial for informing fair, economically sound, and statistically efficient decision-making processes. However, we argue that reinforcement learning from human feedback (RLHF)—the predominant approach for aligning LLMs with human preferences through a reward model—suffers from an inherent algorithmic bias due to its Kullback–Leibler-based regularization in optimization. In extreme cases, this bias could lead to a phenomenon we term preference collapse, where minority preferences are virtually disregarded. To mitigate this algorithmic bias, we introduce preference matching (PM) RLHF, a novel approach that provably aligns LLMs with the preference distribution of the reward model under the Bradley–Terry–Luce/Plackett–Luce model. Central to our approach is a PM regularizer that takes the form of the negative logarithm of the LLM’s policy probability distribution over responses, which helps the LLM balance response diversification and reward maximization. Notably, we obtain this regularizer by solving an ordinary differential equation that is necessary for the PM property. For practical implementation, we introduce a conditional variant of PM RLHF that is tailored to natural language generation. Finally, we empirically validate the effectiveness of conditional PM RLHF through experiments on the OPT and Llama‑family models, demonstrating a 29% to 41% improvement in alignment with human preferences, as measured by a certain metric, compared to standard RLHF.

## Required Packages & Versions

- `cuda` ≥ 12.1
- `python` ≥ 3.10  
- `torch` (PyTorch) ≥ 2.0  
- `transformers` ≥ 4.29.0  
- `deepspeed-chat` ≥ 0.9.0
- `datasets` ≥ 2.10.0
- `accelerate` ≥ 0.18.0
- `wandb` ≥ 0.15.0 
- `scipy` ≥ 1.10.0  
- `numpy` ≥ 1.24.0  


## Reproducing the Experiments of Preferences in Helpfulness and Harmlessness

To reproduce the experiments reported in Figure 1 and Table 1 of the paper, run the following shell scripts:

Step 1: SFT:

```bash
bash training_scripts/sft/train_llama2_7b.sh
```

Step 2: Reward Modeling:

```bash
bash training_scripts/rm/train_llama2_7b.sh
```

Step 3: RLHF Finetuning:

```bash
bash training_scripts/po/ppo/train_llama2_7b.sh
```

For standard RLHF, set the parameters 

```bash
   --penalty "kl" \
   --kl_ctl 0.1 \
   --ent_ctl 0 \
   --alpha_ctl 0 \
```

in the file train_llama2_7b.sh.

For the experiments of PM RLHF, set the parameters 

```bash
   --penalty "entropy" \
   --kl_ctl 0 \
   --ent_ctl 0.1 \
   --alpha_ctl 0.1 \
```

Here, kl_ctl and ent_ctl are the beta in the paper, alpha_ctl is the alpha in the paper. By letting ent_ctl = 0.1, 0.5, 1.0 and alpha_ctl = 0.1, 0.3, 0.5, we can obtain the results shown in Figure 1 and Table 1 of the paper.

## Reproducing the Experiments of Preferences in Summarization

Replace train_llama2_7b.sh by train_llama2_7b_tldr.sh, train_llama3_1b_tldr.sh, and train_llama3_3b_tldr.sh in the above shell scripts. We can obtain the results shown in Table 2 of the paper.

## Reproducing the Experiments on the OPT model

Replace train_llama2_7b.sh by train_opt1.3b.sh in the above shell script. We can obtain the results shown in Figure 5,6, and 7, and Table 3 of the paper.

## Reproducing the Experiments on output probabilities

Once the trained model is saved, run the following

```bash
training_scripts/Output_probs.ipynb
```
to obtain the results shown in Figure 2 and 4.

## Training Time

The following table provides the estimated time to run one epoch on the TLDR dataset using 4 A100 GPUs:

| Model       | Step 1 | Step 2 | Step 3 |
|-------------|--------|--------|--------|
| LLaMA 2‑7B  | 12 min | 36 min | 59 min |
| LLaMA 3‑1B  | 2 min  | 5 min  | 8 min  |
| LLaMA 3‑3B  | 5 min  | 15 min | 25 min |


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
