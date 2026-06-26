# nano-Geneformer

A minimal, fast, and faithful reimplementation of [Geneformer](https://huggingface.co/ctheodoris/Geneformer/tree/main/geneformer) for single-cell foundation model inference, supporting all official Geneformer checkpoints (Geneformer-V1, Geneformer-V2-104M, Geneformer-V2-104M_CLcancer, and Geneformer-V2-316M), with planned support for fine-tuning and training from scratch.

nano-Geneformer faithfully reproduces the original Geneformer architecture in a lightweight, easy-to-read codebase. nano-Geneformer aims to provide:
- A clean and minimal implementation
- Faithful reproduction of the original Geneformer architecture
- Faster inference with modern PyTorch optimizations
- A codebase suitable for experimentation, fine-tuning, and future training from scratch

![figure1](assets/umap_nano_geneformer_vs_geneformer.png)

## The nano-scFMs Project
Single-cell foundation models (scFMs) are one of the most promising directions in AI for biology, yet many existing repositories remain difficult to read, extend, benchmark, or use as educational resources.

nano-Geneformer is part of **nano-scFMs**, a collection of lightweight reimplementations of popular single-cell foundation models. The goal is to make state-of-the-art scFMs easier to understand, extend, benchmark, and use as educational resources.

All repositories are implemented in pure, modern PyTorch and follow a consistent coding style, making it straightforward to install, compare, and experiment with different models using the same environment and shared [requirements file](https://github.com/huynguyen250896/nano-Geneformer/blob/main/requirements.txt). 

Available Models:

- [X] [nano-scBERT](https://github.com/huynguyen250896/nano-scBERT)
- [X] nano-Geneformer
- [ ] nano-scFoundation
- [ ] nano-CellFM
- [ ] nano-UCE
- [ ] nano-scPRINT

**NOTE:** Danqi Liao has already created an excellent minimal implementation of scGPT, so I chose not to duplicate that effort. If you're looking for a lightweight version of scGPT, check out [nano-scGPT](https://github.com/Danqi7/nano-scGPT).

If you know of another single-cell foundation model that should be included, feel free to open an issue or send me a message. To keep the collection focused on established methods, I currently only plan to include models that have been published in peer-reviewed journals.

## Benchmark 
I carefully benchmarked nano-Geneformer across different settings to give future users confidence in adopting nano-Geneformer as a drop-in alternative to the official implementation. Full benchmark details are available in [benchmark_geneformer_vs_nano.ipynb](benchmark_Geneformer_vs_nano.ipynb).

#### Inference Runtime

nano-Geneformer faithfully reproduces the official Geneformer implementation while delivering up to **1.15× faster inference** and reducing **peak GPU memory by up to 56.8%**, enabling substantially more memory-efficient inference for larger Geneformer models.


| Model | Total (15,681 cells) | Per cell | Throughput | Peak GPU | % Reduced Peak GPU | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **nano-Geneformer-V1** | **3.65 s** | **0.233 ms** | **4295.03 cells/s** | **15.398 GB** | **8.7%** | **1.15×** |
| Geneformer-V1 | 4.19 s | 0.267 ms | 3742.51 cells/s | 16.860 GB | — | 1.00× |
| **nano-Geneformer-V2-104M** | **22.23 s** | **1.417 ms** | **705.52 cells/s** | **14.691 GB** | **39.8%** | **1.06×** |
| Geneformer-V2-104M | 23.56 s | 1.502 ms | 665.62 cells/s | 24.420 GB | — | 1.00× |
| **nano-Geneformer-V2-316M** | **56.51 s** | **3.604 ms** | **277.47 cells/s** | **17.094 GB** | **56.8%** | **1.06×** |
| Geneformer-V2-316M | 60.07 s | 3.831 ms | 261.06 cells/s | 39.602 GB | — | 1.00× |

#### Cell-level Embedding Reproducibility
nano-Geneformer reproduces the original Geneformer embedding space almost exactly, preserving both local and global structure.

![figure2](assets/umap_overlay_nano_geneformer_vs_geneformer.png)

| Model | Metric | Value |
| --- | --- | ---: |
| **Geneformer-V1** | Mean cosine similarity | **0.9999999** |
|  | Median cosine similarity | **0.9999999** |
|  | Minimum cosine similarity | **0.9999908** |
|  | Mean absolute difference | **2.83e-05** |
|  | Maximum absolute difference | **1.21e-03** |
|  | Distance correlation | **0.9999990** |
| **Geneformer-V2-104M** | Mean cosine similarity | **1.0000000** |
|  | Median cosine similarity | **1.0000000** |
|  | Minimum cosine similarity | **0.9999993** |
|  | Mean absolute difference | **6.28e-05** |
|  | Maximum absolute difference | **1.31e-03** |
|  | Distance correlation | **0.99999997** |
| **Geneformer-V2-316M** | Mean cosine similarity | **1.0000000** |
|  | Median cosine similarity | **0.9999999** |
|  | Minimum cosine similarity | **0.9999995** |
|  | Mean absolute difference | **5.25e-05** |
|  | Maximum absolute difference | **7.41e-04** |
|  | Distance correlation | **0.99999986** |

The PCA spectrum and pairwise distance structure are nearly identical between nano-Geneformer and the original implementation.

> Benchmarked on a single NVIDIA A100 (80 GB) GPU with batch size 256 on the Pancreas dataset (15,681 cells).

## Install
```bash
git clone https://github.com/huynguyen250896/nano-Geneformer.git
cd nano-Geneformer

pip install -r requirements.txt
```

## Quick Start

### Using nano-Geneformer in Python
#### Generate Cell Embeddings from Raw-count `.h5ad` with nano-Geneformer (Geneformer-V2-316M)
```python
import torch

# Select GPU if available; otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Choose one of the supported pretrained Geneformer checkpoints.
# Available options:
#   - Geneformer-V1
#   - Geneformer-V2-104M
#   - Geneformer-V2-104M_CLcancer
#   - Geneformer-V2-316M
model_name = "Geneformer-V2-316M"

# Load the pretrained model and switch to evaluation mode.
from model import GeneformerForMaskedLM
model = GeneformerForMaskedLM.from_pretrained(model_name, compile_model=False).to(device)
model.eval()

# Tokenize the raw-count .h5ad file 
# Run the pretrained Geneformer model.
# Returns a tensor of cell embeddings with shape (n_cells, hidden_size).
embs = model.encode(
    "./pancreas.h5ad",
    model_name=model_name,
    batch_size=256, # Increase for faster inference if GPU memory allows.
)
```

### Using nano-Geneformer from the Terminal
#### Generate Cell Embeddings from Raw-count `.h5ad` with nano-Geneformer (Geneformer-V1)
```sh
python tasks/embedding.py \
  --input data/pancreas.h5ad \
  --output geneformer_v1_embeddings.npy \
  --model Geneformer-V1 \
  --batch-size 256 \
  --save-tokenized pancreas_v1_tokenized.pt \
  --mode raw \
  --no_compile
```

#### Generate Cell Embeddings from tokenized `.h5ad` with nano-Geneformer (Geneformer-V2-104M)
```sh
python tasks/embedding.py \
  --input pancreas_v2_104M_tokenized.pt \
  --output geneformer_v2_104M_embeddings_model_only.npy \
  --model Geneformer-V2-104M \
  --batch-size 256 \
  --mode tokenized \
  --no_compile
```

## Roadmap
- [X] Embedding .h5ad scRNA data with Geneformer
- [ ] Finetuning for cell classification
- [ ] Finetuning for gene classification
- [ ] Finetuning for in-silico perturbation
- [ ] Training from scratch

Let me know what tasks you'd like to see next!

## Acknowledgments
1. If you find this repo interesting and/or use nano-Geneformer in your work, please cite the original papers:
> C V Theodoris, L Xiao, A Chopra, M D Chaffin, Z R Al Sayed, M C Hill, H Mantineo, E Brydon, Z Zeng, X S Liu, P T Ellinor. Transfer learning enables predictions in network biology. Nature, 31 May 2023.

> H Chen, M S Venkatesh, J Gomez Ortega, S V Mahesh, T Nandi, R Madduri, K Pelka†, C V Theodoris. Scaling and quantization of large-scale foundation model enables resource-efficient predictions in network biology. Nature Computational Science, 27 Mar 2026.

> Y Zhang, M S Venkatesh, and C V Theodoris. Discovery of candidate therapeutic targets with Geneformer. Nature Protocols, 23 Apr 2026.

and STAR⭐ my repo. Thanks!

2. nano-Geneformer is inspired by Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanogpt), Chris Hayduk's [minAlphaFold2](https://github.com/ChrisHayduk/minAlphaFold2), and especially Danqi Liao's [nano-scGPT](https://github.com/Danqi7/nano-scGPT).

## License
[MIT LICENSE](LICENSE)
