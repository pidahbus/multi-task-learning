# ARJUN: A Parameter-Efficient Multi-Objective Pre-Training Framework for Language Models

The proliferation of large-scale language models has significantly advanced natural language processing capabilities; yet their substantial computational requirements create barriers for deployment in resource-constrained environments and limit accessibility for under-represented language communities. To address this fundamental issue, we present a novel multi-objective pre-training approach for developing parameter-efficient language models. Our method, implemented in our Arjun model family, strategically combines four complementary pre-training objectives within a unified transformer encoder-decoder framework: (1) Masked Token Prediction (MT) following BERT's bidirectional understanding paradigm, (2) Next Token Prediction (NT) incorporating autoregressive generation capabilities, (3) Masked Span Prediction (MS) that works on spans of text instead of just words, and (4) Complete Sentence Prediction (CS) adapted from T5 that bridge discriminative and generative tasks. Using this multi-objective approach, Arjun achieves competitive or superior performance for downstream applications with only 69M parameters, compared to models that are 2.5-4 times larger. Evaluations on 14 tasks across three languages---Bangla (Bengali), Tamil, and Hindi---show state-of-the-art results, with average macro-F1 gains of 1.3-1.7 percentage points (pp) on discriminative tasks and a 13.02 pp sacreBLEU improvement on Bangla generative tasks. Ablation studies confirm that joint training with all four objectives consistently outperforms single-objective and partial combinations, with improvements ranging from 0.5-2.0 pp across tasks, thereby validating the synergistic effects of our multi-objective framework.

## Pre-Trained Models
Below table are Arjun models pre-trained on multiple monolingual corpus,

| Model Name        | Parameters | Pre-training Corpus |
|-------------------|------------|---------|
| Vac-Arjun (Base)    | 69M        | [Vacaspati](https://bangla.iitk.ac.in/Vacaspati.html) |
| Vac-Arjun (Small)    | 17M        | [Vacaspati](https://bangla.iitk.ac.in/Vacaspati.html) |
| Indic-BangArjun (Base)    | 69M        | [IndicCorp v2 (Bangla Subset)](https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data) |
| Indic-TamArjun (Base)    | 69M        | [IndicCorp v2 (Tamil Subset)](https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data) |
| Indic-HindArjun (Base)    | 69M        | [IndicCorp v2 (Hindi Subset)](https://huggingface.co/datasets/ai4bharat/IndicCorpV2/tree/main/data) |

## Arjun Model Card

### Setup
To run pre-training and fine-tuning scripts, please install the required packages using the following command:
```bash
$ conda create -n arjun python=3.9 -y
$ conda activate arjun
$ pip install -r requirements.txt
```