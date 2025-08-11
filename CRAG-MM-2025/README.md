[![banner image](https://images.aicrowd.com/raw_images/challenges/social_media_image_file/1155/3d44411079169ec5776a.jpg)](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)
[![Discord](https://img.shields.io/discord/565639094860775436.svg)](https://discord.gg/yWurtB2huX)

# [Meta CRAG-MM: Comprehensive RAG Benchmark for Multi-Modal, Multi-Turn Dialogue Challenge](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)

This repository is the **Submission template and Starter kit** for the Meta CRAG-MM challenge (KDD Cup 2025)! Clone the repository to compete now!

**This repository contains**:
*  **Documentation** on how to submit your agents to the leaderboard
*  **The procedure** for best practices and information on how we evaluate your agent
*  **Starter code** for you to get started!

# Table of Contents

1. [Competition Overview](#-competition-overview)
2. [Dataset](#-dataset)
3. [Tasks](#-tasks)
4. [Evaluation Metrics](#-evaluation-metrics)
5. [Getting Started](#-getting-started)
   - [How to write your own agent?](#️-how-to-write-your-own-agent)
   - [How to start participating?](#-how-to-start-participating)
      - [Setup](#setup)
      - [How to make a submission?](#-how-to-make-a-submission)
      - [What hardware does my code run on?](#-what-hardware-does-my-code-run-on-)
      - [Baselines](#baselines)
6. [Frequently Asked Questions](#-frequently-asked-questions)
7. [Important Links](#-important-links)

# 📖 Competition Overview

Have you tried asking smart glasses to tell you the history of a landmark when travelling to a new country? Have you used wearable devices to translate foreign languages reali-time to order food in a foreign resturant? Have you ever forgoten where you parked your car and thankfully found the location stored in an image remidner on your glasses? Wearable devices are revolutionizing the way people communicate, work, and entertain. To make wearable devices truly valuable in daily life, they must provide relevant and accurate information tailored to users' needs.

Vision Large Language Models (VLLMs) have undergone significant advancements in recent years, empowering multi-modal understanding and visual question answering (VQA) capabilities behind smart glasses. Despite the progress, VLLMs still face a major challenge: generating hallucinated answers. Studies have shown that VLLMs encounter substantial difficulties in handling queries involving long-tail entities; these models also encounter challenges for handling complex queries that require integration of different capabilities: recognition, ocr, knowledge, and generation.

The Retrieval-Augmented Generation (RAG) paradigm has expanded to accommodate multi-modal (MM) input, and demonstrated promise in addressing the knowledge limitation of VLLM. Given an image and a question, an MM RAG system constructs a search query by synthesizing information from the image and the question, searches external sources to retrieve relevant information, and then provides grounded answers to address the question.

# 📊 Dataset

CRAG-MM contains three parts of data: the image set, the QA set, and the contents for retrieval.

The datasets can be accessed as follows:
- **Single-Turn:** [https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-single-turn-public)
- **Multi-Turn:** [https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public](https://huggingface.co/datasets/crag-mm-2025/crag-mm-multi-turn-public)

## 🖼️ Image set
CRAG-MM contains two types of images: egocentric images and normal images. The egocentric images were collected using RayBan Meta Smart Glasses 4 from first-person perspective. The normal images were collected from publicly available images on the web.

## 📝 Question Answer Pairs
CRAG-MM covers 14 domains: Book, Food, General object recognition, Math and science, Nature, Pets, Plants and Gardening, Shopping, Sightseeing, Sports and games, Style and fashion, Text understanding, Vehicles, and Others, representing popular use cases that wearable device users would like to engage with. It also includes 4 types of questions, ranging from simple questions that can be answered based on the image to complex questions that require retrieving multiple sources and synthesizing an answer.

## 📁 Retrieval Contents
The dataset includes a mock image search API and a mock web search API to simulate real-world knowledge sources from which RAG solutions retrieve from.

You can download the mock APIs with 
```
pip install -U cragmm-search-pipeline
```

[docs/search_api.md](docs/search_api.md) contains the documentations to the mock APIs, and [agents/rag_agent.py](agents/rag_agent.py) shows a sample usage of the APIs. 

# 👨‍💻👩‍💻 Tasks

We designed three competition tasks:

## Task #1: Single-Source Augmentation
Task #1 provides an image mock API to access information from an underlying image-based mock KG. The mock KG is indexed by the image, and stores structured data associated with the image; answers to the questions may or may not exist in the mock KG. The mock API takes an image as input, and returns similar images from the mock KG along with structured data associated with each image to support answer generation. This task aims to test basic answer generation capability of MM-RAG systems.

## Task #2: Multi-Source Augmentation
Task #2 in addition provides a web search mock API as a second retrieval source. The web pages are likely to provide useful information for answering the question, but meanwhile also contain noises. This task aims to test how well the MM-RAG system synthesizes information from different sources.

## Task #3: Multi-Turn QA
Task #3 tests the system's ability to conduct multi-turn conversations. Each conversation contains 2–6 turns. Except the first turn, questions in later turns may or may not need the image for answering the questions. Task #3 tests context understanding for smooth multi-turn conversations.


# 📏 Evaluation Metrics

For tasks #1 and #2, we adopt exactly the same metrics and methods used in the CRAG competition ([KDD Cup 2024](https://www.aicrowd.com/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2024)) to assess the performance of the MM RAG systems.

## Single-Turn QA (Tasks #1 and #2)
For each question in the evaluation set, we score the answer with:
- Perfect (fully correct): Score 1
- Acceptable (useful w. minor non-harmful errors): Score 0.5
- Missing (e.g., "I don't know"): Score 0
- Incorrect (wrong or irrelevant): Score -1

We then use Truthfulness as the average score from all examples in the evaluation set for a given MM-RAG system. We compute an average score for each domain, and take the weighted average across all domains for the final score.

## Multi-Turn QA
We adapt the method in [1], which is closest to the information-seeking flavor of conversations. In particular, we stop a conversation when the answers in two consecutive turns are wrong and consider answers to all remaining questions in the same conversation as missing–mimicking the behavior of real users when they lose trust or feel frustrated after repeated failures. We then take the average score of all multi-turn conversations.

[1] Bai et al., "MT-Bench-101: A Fine-Grained Benchmark for Evaluating Large Language Models in Multi-Turn Dialogues". Available at: https://aclanthology.org/2024.acl-long.401/

# 🏁 Getting Started

1. **Sign up** to join the competition [on the AIcrowd website](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025).
2. **Fork** this starter kit repository. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.
3. **Clone** your forked repo and start developing your agent.
4. **Develop** your agent(s) following the template in [how to write your own agent](#-how-to-write-your-own-agent) section.
5. [**Submit**](#-how-to-make-a-submission) your trained models to [AIcrowd Gitlab](https://gitlab.aicrowd.com) for evaluation.

# ✍️ How to write your own agent?

Please follow the instructions in [agents/README.md](agents/README.md) for instructions and examples on how to write your own agents for this competition.

# 🚴 How to start participating?

## Setup

1. **Add your SSH key** to AIcrowd GitLab

You can add your SSH Keys to your GitLab account by going to your profile settings [here](https://gitlab.aicrowd.com/-/user_settings/ssh_keys). If you do not have SSH Keys, you will first need to [generate one](https://docs.gitlab.com/ee/user/ssh.html).

2. **Fork the repository**. You can use [this link](https://gitlab.aicrowd.com/aicrowd/challenges/meta-comprehensive-rag-benchmark-kdd-cup-2025/meta-comprehensive-rag-benchmark-starter-kit/-/forks/new) to create a fork.

3. **Clone the repository**
    ```bash
    git clone git@gitlab.aicrowd.com:<YOUR-AICROWD-USERNAME>/meta-crag-submission.git
    cd meta-crag-submission
    ```

4. **Install** competition specific dependencies!
    ```bash
    cd meta-crag-submission
    pip install -r requirements.txt
    ```
**Note**: The installation of vLLM may depend on specific CUDA or PyTorch versions, so it is possible that `pip install -r requirements.txt` fails. If that happens, please find an appropriate version on the [vLLM website](https://docs.vllm.ai/en/latest/). To run LLaMA-3.2-Vision, we need at least `vllm>=0.6.2`. 

5. Write your own agent as described in [agents/README.md](agents/README.md).

6. Test your agent locally using `python local_evaluation.py`.

7. Accept the Challenge Rules on the main [challenge page](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025) by clicking on the **Participate** button. Also accept the Challenge Rules on the Task specific page (link on the challenge page) that you want to submit to.

8. Make a submission as described in [How to make a submission](#-how-to-make-a-submission) section.

## 📮 How to make a submission?

Please follow the instructions in [docs/submission.md](docs/submission.md) to make your first submission.
This also includes instructions on [specifying your software runtime](docs/submission.md#specifying-software-runtime-and-dependencies), [code structure](docs/submission.md#code-structure-guidelines), [submitting to different tracks](docs/submission.md#submitting-to-different-tracks).

For detailed instructions on securely setting up your submissions as Public Gated Hugging Face models, please refer to [Using Gated Hugging Face Models in Your Submission 🔒](huggingface-gated-models.md).

**Note**: **Remember to accept the Challenge Rules** on the challenge page, **and** the task page before making your first submission.

## 💻 What hardware does my code run on?
All submissions will be run on a single `g6e.2xlarge` instance with an `NVIDIA L40s GPU` with `48GB of GPU memory` on AWS. Please note that:
- `LLaMA 3.2 11B-Vision` and `Pixtral 12B` in full precision can run directly
- `Llama 3.2 90B-Vision` in full precision cannot be directly run on this GPU instance. Quantization or other techniques need to be applied to make the model runnable

Moreover, the following restrictions will also be imposed:
- Network connection will be disabled
- Each conversation-turn will have a time-out limit of 10 seconds, and a batch of N turns (as configured by you `.get_batch_size()` function) will have a timeout of `N * 10 seconds`. 
- To encourage concise answers, each answer will be truncated to 75 bpe tokens in the auto-eval

## 🏁 Baseline
We include three baselines for demonstration purposes:
1. RandomAgent: A simple agent that generates random responses
2. LlamaVisionModel: A vision-language agent based on Meta's LLaMA 3.2 11B Vision Instruct model
3. SimpleRAGAgent: A RAG-based agent that uses the unified search pipeline for retrieving relevant information

You can read more about them in [docs/baselines.md](docs/baselines.md).

# ❓ Frequently Asked Questions

## Which track is this starter kit for?
This starter kit can be used to submit to any of the three tasks. You can find more information in [docs/submission.md#submitting-to-different-tracks](docs/submission.md#submitting-to-different-tracks).

## Where can I know more about the dataset schema?
The dataset schema is described in [docs/dataset.md](docs/dataset.md).

**Best of Luck** :tada: :tada:

# 📎 Important links

- 💪 Challenge Page: https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025
- 🗣 Discussion Forum: https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/discussion
- 🏆 Leaderboard: https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025/leaderboards
- 📧 Contact: crag-kddcup-2025@meta.com
