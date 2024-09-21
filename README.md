<a name="awesome-few-shot"></a>
# Awesome Few-Shot Learning on Graphs

[![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-brightgreen.svg)](https://github.com/smufang/fewshotgraph/pulls) [![Awesome](https://awesome.re/badge.svg)](https://awesome.re) [![GitHub stars](https://img.shields.io/github/stars/smufang/fewshotgraph.svg)](https://github.com/smufang/fewshotgraph/stargazers)

This repository provides a curated collection of research papers focused on few-shot learning on graphs. It is derived from our survey paper: [A Survey of Few-Shot Learning on Graphs: From Meta-Learning to Pre-Training and Prompting](#). We will update this list regularly. If you notice any errors or missing papers, please feel free to open an issue or submit a pull request.

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

5. **Relative and absolute location embedding for few-shot node classification on graph.** In *AAAI'2021*, [Paper](https://zemin-liu.github.io/papers/Relative%20and%20Absolute%20Location%20Embedding%20for%20Few-Shot%20Node%20Classification%20on%20Graph.pdf), [Code](https://github.com/shuaiOKshuai/RALE).\
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
1. **GPPT: Graph pre-training and prompt tuning to generalize graph neural networks.** In *KDD'2022*, [Paper](https://dl.acm.org/doi/abs/10.1145/3534678.3539249).\
[![Template](https://img.shields.io/badge/Template-subgraph--token_similarity-brightgreen)](#) 
[![Feature prompt](https://img.shields.io/badge/Feature_prompt-input-red)](#) 
[![Prompt Initialization](https://img.shields.io/badge/Prompt_Initialization-random-yellow)](#)
[![Downstream Task](https://img.shields.io/badge/Downstream_Task-node-blue)](#)

<a name="tag"></a>
#### Prompting on Text-attributed Graphs

<a name="contributing"></a>
## Contributing

:thumbsup: Contributions to this repository are highly encouraged!

If you have any relevant resources to share, please feel free to open an issue or submit a pull request.

