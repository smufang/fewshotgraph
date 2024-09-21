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
      - [Adaptation by Finetuning](#finetuning)
      - [Parameter-efficient Adaptation](#parameter-efficient)
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

4. **Tackling Long-Tailed Relations and Uncommon Entities in Knowledge Graph Completion.** In *EMNLP'2019*, [Paper](https://arxiv.org/pdf/1909.11359), [Code].\
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

<a name="pre-training"></a>
## Pre-Training Approaches


<a name="pre-training-strategies"></a>
### Pre-Training Strategies

<a name="parameter-efficient"></a>
### Parameter-efficient Adaptation

<a name="contributing"></a>
## Contributing

:thumbsup: Contributions to this repository are highly encouraged!

If you have any relevant resources to share, please feel free to open an issue or submit a pull request.

