<a name="awesome-few-shot"></a>
# Awesome Few-Shot Learning on Graphs

[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/smufang/fewshotgraph/pulls) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![GitHub stars](https://img.shields.io/github/stars/smufang/fewshotgraph.svg)](https://github.com/smufang/fewshotgraph/stargazers)

This repository provides a curated collection of research papers focused on few-shot learning on graphs. It is derived from our survey paper: [A Survey of Few-Shot Learning on Graphs: From Meta-Learning to Pre-Training and Prompting](https://arxiv.org/abs/2402.01440). We will update this list regularly. If you notice any errors or missing papers, please feel free to open an issue or submit a pull request.

<a name="table-of-contents"></a>
## Table of Contents

- [Awesome Few-Shot Learning on Graphs](#awesome-few-shot)
  - [Table of Contents](#table-of-contents)
  - [Few-Shot Learning on Graphs: Problems](#few-shot-problem)
    - [Label Scarcity](#label-sacrcity)
    - [Structure Scarcity](#structure-sacrcity)
  - [Few-Shot Learning on Graphs: Techniques](#few-shot-technique)
    - [Meta-Learning Approaches](#meta-learning)
      - [Structure-Based Enhancement](#structure-enhancement)
      - [Adaptation-Based Enhancement](#adaptation-enhancement)
    - [Pre-Training Approaches](#pre-training)
      - [Pre-Training Strategies](#pre-training-strategies)
        - [Contrastive Strategies](#contrastive-strategies)
        - [Generative Strategies](#generative-strategies)
      - [Adaptation by Finetuning](#finetuning)
      - [Parameter-efficient Adaptation](#parameter-efficient)
        - [Prompting on Text-free Graphs](#text-free-graph)
        - [Prompting on Text-attributed Graphs](#tag)
    - [Hybrid Approaches](#hybrid) 
  - [Contributing](#contributing)
  - [Citation](#citation)


<a name="meta-learning"></a>
## Meta-Learning Approaches

<a name="structure-enhancement"></a>
### Structure-Based Enhancement
1. **Graph Prototypical Networks for Few-shot Learning on Attributed Networks.** In *CIKM'2020*, [Paper](https://arxiv.org/pdf/2006.12739), [Code](https://github.com/kaize0409/GPN_Graph-Few-shot).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Node-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Protonets-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

2. **Adaptive Attentional Network for Few-Shot Knowledge Graph Completion.** In *EMNLP'2020*, [Paper](https://arxiv.org/pdf/2010.09638), [Code](https://github.com/JiaweiSheng/FAAN).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Node-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Protonets-red)](#) 
[![Task](https://img.shields.io/badge/Task-Edge-yellow)](#)

3. **HMNet: Hybrid Matching Network for Few-Shot Link Prediction.** In *DASFAA'2021*, [Paper](https://link.springer.com/chapter/10.1007/978-3-030-73194-6_21).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Edge-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MN-red)](#) 
[![Task](https://img.shields.io/badge/Task-Edge-yellow)](#)

4. **Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion.** In *EMNLP'2019*, [Paper](https://arxiv.org/pdf/1909.11359), [Code](https://github.com/ZihaoWang/Few-shot-KGC).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Edge-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Edge-yellow)](#)

5. 🌟**Relative and absolute location embedding for few-shot node classification on graph.** In *AAAI'2021*, [Paper](https://zemin-liu.github.io/papers/Relative%20and%20Absolute%20Location%20Embedding%20for%20Few-Shot%20Node%20Classification%20on%20Graph.pdf), [Code](https://github.com/shuaiOKshuai/RALE).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Path-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

6. **Meta-learning on heterogeneous information networks for cold-start recommendation.** In *KDD'2020*, [Paper](https://fangyuan1st.github.io/paper/KDD20_MetaHIN.pdf), [Code](https://github.com/rootlu/MetaHIN).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Path-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

7. **Graph meta learning via local subgraphs.** In *NeurIPS'2020*, [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/412604be30f701b1b1e3124c252065e6-Paper.pdf), [Code](https://github.com/mims-harvard/G-Meta).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Subgraph-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Hybrid-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node,%20Edge-yellow)](#)

8. **Learning to extrapolate knowledge: Transductive few-shot out-of-graph link prediction.** In *NeurIPS'2020*, [Paper](https://proceedings.neurips.cc/paper_files/paper/2020/file/0663a4ddceacb40b095eda264a85f15c-Paper.pdf), [Code](https://github.com/JinheonBaek/GEN).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Subgraph-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Protonets-red)](#) 
[![Task](https://img.shields.io/badge/Task-Edge-yellow)](#)

9. **Towards locality-aware meta-learning of tail node embeddings on networks.** In *CIKM'2020*, [Paper](https://zemin-liu.github.io/papers/CIKM-20-towards-locality-aware-meta-learning-of-tail-node-embeddings-on-network.pdf), [Code](https://github.com/smufang/meta-tail2vec).\
[![Structure enhancement](https://img.shields.io/badge/Structure%20enhancement-Subgraph-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node,%20Edge-yellow)](#)


<a name="adaptation-enhancement"></a>
### Adaptation-Based Enhancement
1. **Graph few-shot learning via knowledge transfer.** In *AAAI'2020*, [Paper](https://grlearning.github.io/papers/13.pdf).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Graph-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

2. **Meta-Inductive Node Classification across Graphs.** In *SIGIR'2021*, [Paper](https://arxiv.org/pdf/2105.06725), [Code](https://github.com/WenZhihao666/MI-GNN).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Graph-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Hybrid-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

3. **Prototypical networks for few-shot learning.** In *NeurIPS'2017*, [Paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/cb8da6767461f2812ae4290eac7cbc42-Paper.pdf), [Code](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Task-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-Protonets-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

4. **Graph Few-shot Learning with Attribute Matching.** In *CIKM'2020*, [Paper](https://www.researchgate.net/profile/Kaize-Ding/publication/346267392_Graph_Few-shot_Learning_with_Attribute_Matching/links/61687a7766e6b95f07cb607d/Graph-Few-shot-Learning-with-Attribute-Matching.pdf).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Task-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Node-yellow)](#)

5. **Adaptive-Step Graph Meta-Learner for Few-Shot Graph Classification.** In *CIKM'2020*, [Paper](https://arxiv.org/pdf/2003.08246).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Step%20Size-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Graph-yellow)](#)

6. **Few-shot link prediction in dynamic networks.** In *WSDM'2022*, [Paper](https://yuanfulu.github.io/publication/WSDM-MetaDyGNN.pdf).\
[![Structure enhancement](https://img.shields.io/badge/Adaptation%20enhancement-Hybrid-brightgreen)](#) 
[![Meta learner](https://img.shields.io/badge/Meta%20learner-MAML-red)](#) 
[![Task](https://img.shields.io/badge/Task-Edge-yellow)](#)

<a name="pre-training"></a>
## Pre-Training Approaches

<a name="pre-training-strategies"></a>
### Pre-Training Strategies

<a name="contrastive-strategies"></a>
#### Contrastive Strategies

<a name="generative-strategies"></a>
#### Generative Strategies

<a name="parameter-efficient"></a>
### Parameter-efficient Adaptation

<a name="text-free-graph"></a>
#### Prompting on Text-free Graphs
1. **GPPT: Graph pre-training and prompt tuning to generalize graph neural networks.** In *KDD'2022*, [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539249), [Code](https://github.com/MingChen-Sun/GPPT).\
[![Template](https://img.shields.io/badge/Template-Subgraph--token_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

2. **Voucher Abuse Detection with Prompt-based Fine-tuning on Graph Neural Networks.** In *CIKM'2023*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3583780.3615505), [Code](https://github.com/WenZhihao666/VPGNN).\
[![Template](https://img.shields.io/badge/Template-Node--token_matching-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Structure_prompt-8A2BE2)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

3. 🌟**Graphprompt: Unifying pre-training and downstream tasks for graph neural networks.** In *WWW'2023*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3543507.3583386), [Code](https://github.com/Starlien95/GraphPrompt).\
[![Template](https://img.shields.io/badge/Template-Subgraph_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Readout-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

4. **Motif-based prompt learning for universal cross-domain recommendation.** In *WSDM'2024*, [Paper](https://dl.acm.org/doi/abs/10.1145/3616855.3635754).\
[![Template](https://img.shields.io/badge/Template-Subgraph_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Readout-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Edge-blue)](#)

5. 🌟**Generalized graph prompt: Toward a unification of pre-training and downstream tasks on graphs.** In *TKDE'2024*, [Paper](https://ieeexplore.ieee.org/abstract/document/10572358),[Code](https://github.com/Starlien95/GraphPrompt).\
[![Template](https://img.shields.io/badge/Template-Subgraph_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-All_layers-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

6. **Non-Homophilic Graph Pre-Training and Prompt Learning.** In *arXiv'2024*, [Paper](https://arxiv.org/abs/2408.12594).\
[![Template](https://img.shields.io/badge/Template-Subgraph_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Readout-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Conditional-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

7. **Text-Free Multi-domain Graph Pre-training: Toward Graph Foundation Models.** In *arXiv'2024*, [Paper](https://arxiv.org/abs/2405.13934).\
[![Template](https://img.shields.io/badge/Template-Node_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Readout-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Pretext_tokens-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

8. **MultiGPrompt for multi-task pre-training and prompting on graphs.** In *WWW'2024*, [Paper](https://dl.acm.org/doi/abs/10.1145/3589334.3645423), [Code](https://github.com/Nashchou/MultiGPrompt).\
[![Template](https://img.shields.io/badge/Template-Node_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-All_layers-red)](#)
[![Multiple pretext tasks](https://img.shields.io/badge/Multiple_pretext_tasks-deeppink)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Pretext_tokens-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

9. **HetGPT: Harnessing the power of prompt tuning in pre-trained heterogeneous graph neural networks.** In *WWW'2024*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645685).\
[![Template](https://img.shields.io/badge/Template-Node_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

10. **Universal prompt tuning for graph neural networks.** In *NeurIPS'2023*, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a4a1ee071ce0fe63b83bce507c9dc4d7-Abstract-Conference.html), [Code](https://github.com/zjunet/GPF).\
[![Template](https://img.shields.io/badge/Template-Universal_feature/spectral_space-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

11. **Inductive Graph Alignment Prompt: Bridging the Gap between Graph Pre-training and Inductive Fine-tuning From Spectral Perspective.** In *WWW'2024*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3589334.3645620).\
[![Template](https://img.shields.io/badge/Template-Universal_feature/spectral_space-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Singal-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Graph-blue)](#)

12. **Sgl-pt: A strong graph learner with graph prompt tuning.** In *arXiv'2023*, [Paper](https://arxiv.org/abs/2302.12449).\
[![Template](https://img.shields.io/badge/Template-Dual--template-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Structure_prompt-8A2BE2)](#) 
[![Multiple pretext tasks](https://img.shields.io/badge/Multiple_pretext_tasks-deeppink)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Graph-blue)](#)

13. **HGPrompt: Bridging homogeneous and heterogeneous graphs for few-shot prompt learning.** In *AAAI'2024*, [Paper](https://arxiv.org/pdf/2312.01878), [Code](https://github.com/Starlien95/HGPrompt).\
[![Template](https://img.shields.io/badge/Template-Dual--template,_graph_template-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Readout-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

14. **PSP: Pre-training and structure prompt tuning for graph neural networks.** In *arXiv'2024*, [Paper](https://arxiv.org/pdf/2310.17394), [Code](https://github.com/gqq1210/PSP).\
[![Template](https://img.shields.io/badge/Template-View_similarity-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Structure_prompt-8A2BE2)](#) 
[![Multiple pretext tasks](https://img.shields.io/badge/Multiple_pretext_tasks-deeppink)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Graph-blue)](#)

15. **ULTRA-DP: Unifying graph pre-training with multi-task graph dual prompt.** In *arXiv'2023*, [Paper](https://arxiv.org/pdf/2310.14845), [Code](https://github.com/Keytoyze/ULTRA-DP).\
[![Template](https://img.shields.io/badge/Template-Node--node/group_similarity-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Feature prompt](https://img.shields.io/badge/Structure_prompt-8A2BE2)](#) 
[![Multiple pretext tasks](https://img.shields.io/badge/Multiple_pretext_tasks-deeppink)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge-blue)](#)

16. **Virtual node tuning for few-shot node classification.** In *KDD'2023*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599541).\
[![Template](https://img.shields.io/badge/Template-Node_attribute_reconstruction,_structure_recovery-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Meta--trained-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

17. 🌟**All in one: Multi-task prompting for graph neural networks.** In *KDD'2023*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3580305.3599256), [Code](https://github.com/sheldonresearch/ProG).\
[![Template](https://img.shields.io/badge/Template-Subgraph_classification-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Structure_prompt-8A2BE2)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Meta--trained-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

18. **DyGPrompt: Learning Feature and Time Prompts on Dynamic Graphs.** In *arXiv'2024*, [Paper](https://arxiv.org/abs/2405.13937).\
[![Template](https://img.shields.io/badge/Template-Temporal_node_similarity-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Conditional-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge-blue)](#)

19. **Prompt learning on temporal interaction graphs.** In *arXiv'2024*, [Paper](https://arxiv.org/abs/2402.06326).\
[![Template](https://img.shields.io/badge/Template-Temporal_node_similarity-brightgreen)](#)
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-Input-red)](#)
[![Prompt Initialization](https://img.shields.io/badge/Prompt_initialization-Time--based-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge-blue)](#)




<a name="tag"></a>
#### Prompting on Text-attributed Graphs
1. 🌟**Augmenting low-resource text classification with graph-grounded pre-training and prompting.** In *SIGIR'2023*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3539618.3591641), 
 [Code](https://github.com/WenZhihao666/G2P2).\
[![Instruction](https://img.shields.io/badge/Instruction-Text-brightgreen)](#)
[![Learnable prompt](https://img.shields.io/badge/Learnable_prompt-Vector-red)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

2. **Prompt tuning on graph-augmented low-resource text classification.** In *TKDE'2024*, [Paper](https://smufang.github.io/paper/TKDE24_G2P2Star.pdf), [Code](https://github.com/WenZhihao666/G2P2-conditional).\
[![Instruction](https://img.shields.io/badge/Instruction-Text-brightgreen)](#)
[![Learnable prompt](https://img.shields.io/badge/Learnable_prompt-Condition--net-red)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

2. **GraphGPT: Graph instruction tuning for large language models.** In *SIGIR'2024*, [Paper](https://dl.acm.org/doi/abs/10.1145/3626772.3657775), [Code](https://github.com/HKUDS/GraphGPT).\
[![Instruction](https://img.shields.io/badge/Instruction-Text,_Graph-brightgreen)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

3. **Natural language is all a graph needs.** In *EACL'2024*, [Paper](https://arxiv.org/abs/2308.07134), [Code](https://github.com/agiresearch/InstructGLM).\
[![Instruction](https://img.shields.io/badge/Instruction-Text,_Graph-brightgreen)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)

4. **GIMLET: A unified graph-text model for instruction-based molecule zero-shot learning.** In *NeurIPS'2023*, [Paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/129033c7c08be683059559e8d6bfd460-Abstract-Conference.html), [Code]( https://github.com/zhao-ht/GIMLET).\
[![Instruction](https://img.shields.io/badge/Instruction-Text,_Graph-brightgreen)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Graph-blue)](#)

5. 🌟**One for all: Towards training one graph model for all classification tasks.** In *ICLR'2024*, [Paper](https://openreview.net/pdf?id=4IT2pgc9v6), [Code]( https://github.com/LechengKong/OneForAll).\
[![Instruction](https://img.shields.io/badge/Instruction-Text,_Graph-brightgreen)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node,_Edge,_Graph-blue)](#)

6. **HiGPT: Heterogeneous graph language model.** In *KDD'2024*, [Paper](https://dl.acm.org/doi/pdf/10.1145/3637528.3671987), [Code](https://github.com/HKUDS/HiGPT).\
[![Instruction](https://img.shields.io/badge/Instruction-Text,_Graph-brightgreen)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_task-Node-blue)](#)


<a name="contributing"></a>
## Contributing

:thumbsup: Contributions to this repository are highly encouraged!

If you have any relevant resources to share, please feel free to open an issue or submit a pull request.

<a name="citation"></a>
## Citation

If you find this repo helpful to you, please feel free to cite these works:

🔥[Survey Paper](https://arxiv.org/abs/2402.01440)

```bibtex
@article{yu2024few,
  title={Few-Shot Learning on Graphs: from Meta-learning to Pre-training and Prompting},
  author={Yu, Xingtong and Fang, Yuan and Liu, Zemin and Wu, Yuxia and Wen, Zhihao and Bo, Jianyuan and Zhang, Xinming and Hoi, Steven CH},
  journal={arXiv preprint arXiv:2402.01440},
  year={2024}
}
```

🔥[**GraphPrompt**](https://arxiv.org/abs/2302.08043) A Representative GNN Prompting Framework. One of the Most Influential Papers in WWW'23 by Paper Digest (2023-09 Version)

```bibtex
@inproceedings{liu2023graphprompt,
  title={Graphprompt: Unifying pre-training and downstream tasks for graph neural networks},
  author={Liu, Zemin and Yu, Xingtong and Fang, Yuan and Zhang, Xinming},
  booktitle={WWW},
  year={2023}
}
```

