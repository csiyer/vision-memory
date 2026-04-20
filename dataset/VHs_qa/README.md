---
license: mit
---

# Visual Haystacks Dataset Card

## Dataset details
1. Dataset type: Visual Haystacks (VHs) is a benchmark dataset specifically designed to evaluate the Large Multimodal Model's (LMM's) capability to handle long-context visual information. It can also be viewed as the first vision-centric Needle-In-A-Haystack (NIAH) benchmark dataset. Please also download COCO-2017's training set validation set.

2. Data Preparation and Benchmarking
  - Download the VQA questions:
    ```
    huggingface-cli download --repo-type dataset tsunghanwu/visual_haystacks --local-dir dataset/VHs_qa
    ```
  - Download the COCO 2017 dataset and organize it as follows, with the default root directory ./dataset/coco:
    ```
    dataset/
    ├── coco
    │   ├── annotations
    │   ├── test2017
    │   └── val2017
    └── VHs_qa
        ├── single_needle
        │   ├── VHs_large
        │   └── VHs_small
        └── multi_needle
    ```
  - Follow the instructions in https://github.com/visual-haystacks/vhs_benchmark to run the evaluation.

3. We utilized the full dataset from `single_needle/VHs_large` and `multi_needle`, which includes 1,000 test cases, to conduct all experiments depicted in Figures 2 and 3 with fewer than 100 images, and a third of this dataset for experiments with more than 100 input images. Additionally, the `single_needle/VHs_small` dataset, comprising 100 test cases, was employed specifically for the experiments on positional biases (Figure 4).

4. Please check out our [project page](https://visual-haystacks.github.io) for more information. You can also send questions or comments about the model to [our github repo](https://github.com/visual-haystacks/vhs_benchmark/issues).

5. This is the updated VHs dataset, enhanced for greater diversity and balance. The original dataset can be found at [tsunghanwu/visual_haystacks_v0](https://huggingface.co/datasets/tsunghanwu/visual_haystacks_v0).

## Intended use
Primary intended uses: The primary use of VHs is research on large multimodal models and chatbots.

Primary intended users: The primary intended users of the model are researchers and hobbyists in computer vision, natural language processing, machine learning, and artificial intelligence.
