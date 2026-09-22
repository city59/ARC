# ARC: Adaptive Relation Compression

Implementation accompanying **Adaptive Relation Compression for Knowledge-Aware Recommendation: An Information-Theoretic Perspective** by Cheng Li, Jianfeng Sun, Yong Xu, and Jinde Cao.

ARC formulates personalized relation compression as a conditional information bottleneck. It combines:

- User context learning through information-constrained graph message passing.
- Context-guided masking and hard assignment to a shared relation codebook.
- Knowledge propagation with compressed relations and explicit message information costs.

Training balances BPR ranking loss with relation and message information costs. The context encoder is pretrained and then frozen.

## Setup

Tested with Python 3.9 and PyTorch 1.12.1 (CUDA) / 2.3.0 (CPU). The requirements pin the 2.3.0 reference environment, including its runtime dependencies. Select a CPU or CUDA wheel using the commands in [requirements.txt](requirements.txt), then run from this directory:

```bash
python -m pip install -r requirements.txt
```

## Datasets

The included datasets are stored under `data/`:

| Argument | Dataset |
|---|---|
| `movie` | MovieLens-1M |
| `music` | Last.FM |
| `book` | Book-Crossing |

The loader preserves the supplied train/test split and creates validation data from training positives. Only positive training interactions enter the graph.

## Training and Evaluation

Train a dataset (`movie`, `music`, or `book`):

```bash
python main.py --dataset music --device auto --output-dir runs/arc
```

Run a short check:

```bash
python main.py --dataset music --smoke-test --device auto --output-dir runs/smoke
```

Evaluate a saved model:

```bash
python main.py --eval-only --checkpoint runs/arc/music/train/best.pt --device auto
```

`auto` selects CUDA when available. Use `--device cpu` to force CPU execution. Logs, checkpoints, and metrics are saved under `<output-dir>/<dataset>/train/` (`smoke/` for smoke tests). Choose a new output directory for each run.
