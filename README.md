# Advanced Rock–Paper–Scissors AI — Technical README

**Repository:** Advanced RPS AI
**File:** `advanced_rps_ai.py` (neural + rule-based ensemble)

---

## Overview

This document explains, precisely and technically, how the `advanced_rps_ai.py` implementation works, how the self-play + curriculum trainer is constructed, how the ensemble decision is made at inference time, and how to use an existing `.pt` model file (for example your uploaded checkpoint).

The design goals:

* Produce a compact, fast neural predictor that captures exploitable human patterns in RPS.
* Combine interpretable rule-based predictors (frequency, Markov, pattern) with a learned neural predictor.
* Train the neural predictor by mixed **self-play + curriculum** so it learns to predict non-uniform opponents while staying stable.
* At runtime, adaptively weight predictors by recent empirical accuracy and add controlled exploration.

---

## Components (algorithmic summary)

### 1) Rule-based predictors

* **FrequencyPredictor**: counts absolute occurrence of each move in opponent history and predicts the most frequent move.
* **MarkovPredictor (order=2)**: stores counts `P(next | last_k)` for k=2 (configurable). Predicts the most likely next move conditional on the last `k` moves.
* **PatternPredictor**: searches for repeated suffix patterns up to `max_k` and returns the most frequent move that followed the matched suffix in the history.

These predictors are fast, deterministic, interpretable and excel at common human biases (repeats, momentum, heavy-frequency).

### 2) Neural predictor (MLP)

* Architecture: an MLP taking a flattened one-hot encoding of the last `history_len` opponent moves. Typical default: `history_len=8`, input size `8*3=24`. Hidden sizes: `[128, 64]` (configurable).
* Output: 3 logits (rock/paper/scissors). Training minimizes cross-entropy to predict the opponent's next move.
* Encoding: each move is one-hot; shorter histories are padded with neutral distribution `[1/3,1/3,1/3]`.

This model captures subtle correlations over several past moves that rule-based methods may miss.

### 3) Training method — Self-play + Curriculum

Training aims to produce a predictor that is *useful against real (non-uniform) opponents*. Pure self-play universally converges toward Nash (uniform random) which yields poor exploitative performance. To avoid that, training mixes:

* **Self-play batches** (model samples opponent moves from the current model). This allows the model to bootstrap and refine its internal distribution.
* **Curriculum batches** (rule-based synthetic opponents) that inject non-uniform, human-like biases into training data (repeat tendencies, trend-following, mild patterns). This prevents collapse to trivial uniform policy and teaches the network to exploit biased opponents.

Loss: standard `CrossEntropyLoss` on predicted logits.
Optimizer: Adam (default lr `1e-3`).

Key hyperparameters to tune:

* `history_len` — how many past moves the model sees.
* `batch_size` — affects throughput and gradient variance.
* `selfplay_steps` (total simulated moves) — larger values improve capacity to learn rare patterns but cost compute.
* `curriculum_mix` — fraction of batches that are curriculum vs self-play.

### 4) Ensemble & decision rule at inference

At play time the system computes predictions from all available predictors (`freq`, `markov`, `pattern`, `neural`).

* Each predictor maintains running statistics: `correct / total` (a smoothed or floor-protected accuracy). We use `weight = 0.05 + accuracy` to avoid zero weights.
* Weighted votes are accumulated for each candidate opponent move: `vote[move] += weight_of_predictor_that_predicted_move`.
* Predicted opponent move = `argmax(vote)`. AI’s move = `counter(predicted_opponent_move)`.
* Small exploration probability `epsilon` (e.g. 0.03—0.06) forces occasional random moves to avoid being fully deterministic.

This adaptive ensemble prefers predictors that are currently working well on the current human opponent.

---

## Checkpoint format & using your `.pt` file

The trainer saves checkpoints as a Python `dict` (Torch checkpoint) containing at least:

```python
{
  'model_state': <state_dict of NeuralPredictor>,
  'cfg': <trainer/config dict>
}
```

### How to use your uploaded `*.pt` file (example: `rps_predictor.pt`)

1. Put the file next to the project root, e.g. the same directory as `advanced_rps_ai.py`.
2. When starting the interactive script, pass `--model_path rps_predictor.pt` (or set the default path) — the ensemble loader will attempt to `torch.load()` that file and construct the `NeuralPredictor` with `history_len` from your runtime options. Example:

```bash
python advanced_rps_ai.py --model_path rps_predictor.pt
```

3. The loader expects the checkpoint to contain a `model_state` that matches the MLP architecture (same `history_len` and hidden sizes). If the architecture does not match, the script will raise a `RuntimeError`.

### Verifying compatibility (quick snippet)

Run the following small script in Python to sanity-check the checkpoint:

```python
import torch
from advanced_rps_ai import NeuralPredictor

ckpt = torch.load('rps_predictor.pt', map_location='cpu')
state = ckpt.get('model_state')
print('keys in checkpoint:', ckpt.keys())
# Inspect some keys of a saved state_dict
print(list(state.keys())[:10])
# Quick shape check:
model = NeuralPredictor(history_len=8)
model.load_state_dict(state)  # will raise if incompatible
print('Loaded OK')
```

If `load_state_dict` raises key-shape mismatches, either recreate the MLP hyperparameters to match the checkpoint or re-save the checkpoint using the script's model architecture.

---

## Usage & commands

Prerequisites:

```bash
python >= 3.8
pip install torch tqdm
# (For GPU training, install a torch build with CUDA for your GPU.)
```

Basic interactive play (using an existing `.pt`):

```bash
python advanced_rps_ai.py --model_path rps_predictor.pt
# follow the prompt to enter the number of rounds and play.
```

Train (generate & save a model):

```bash
# Run trainer to simulate self-play + curriculum. Example: 100,000 simulated steps
python advanced_rps_ai.py --selfplay 100000 --batch_size 4096 --model_path rps_predictor.pt
```

Notes: the `--selfplay` argument controls the **total simulated steps** (approx. number of virtual opponent moves created for training). For large values (e.g. 1e7 or 1e9) prefer `--device cuda` and large memory machine.

Resuming / fine-tuning from your `.pt`:

The script currently saves only the `model_state` (and `cfg`). To fine-tune (continue training) on a checkpoint:

1. Load the checkpoint into the `NeuralPredictor`:

```python
ckpt = torch.load('rps_predictor.pt')
model = NeuralPredictor(history_len=ckpt['cfg'].get('history_len', 8))
model.load_state_dict(ckpt['model_state'])
# create trainer/optimizer with the same lr and call training loop
```

2. Note: optimizer state (momentum, Adam buffers) is not saved by default in this checkpoint format, so exact optimizer resumption is not available — you will re-initialize an optimizer and continue training (works fine but may slightly alter dynamics).

If you want true resumeability, re-save checkpoints including `optimizer.state_dict()` and training step counters.

---

## Evaluation & validation

To estimate how exploitative the model is (i.e., how well it predicts non-uniform opponents), do several evaluations:

1. **Rule-based accuracy** — measure prediction accuracy vs rule-based opponents (frequency, markov, pattern) using the `evaluate()` helper in the trainer. This provides a sense of how well the model generalizes to typical human biases.
2. **Human trials** — run 1k+ interactive games against real humans and track AI win-rate and predictor accuracies.
3. **Distribution checks** — sample model predictions on many random contexts and inspect entropy of softmax; uniform high-entropy suggests collapse to Nash.

Example: quick python snippet to compute prediction entropy statistics:

```python
import torch, numpy as np
from advanced_rps_ai import NeuralPredictor, encode_history
model = NeuralPredictor(history_len=8)
model.load_state_dict(torch.load('rps_predictor.pt')['model_state'])
model.eval()
ents = []
for _ in range(2000):
    hist = [random.randrange(3) for _ in range(8)]
    x = torch.tensor([encode_history(hist, 8)], dtype=torch.float32)
    logits = model(x)
    probs = torch.softmax(logits, dim=-1).detach().numpy()[0]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    ents.append(ent)
print('avg entropy', sum(ents)/len(ents))
```

Lower average entropy indicates more confident/exploitable predictions.

---

## Troubleshooting & tips

* **Model fails to load**: Verify `history_len` and MLP hidden sizes match the architecture used when saving. Inspect checkpoint `cfg` if present.
* **Model predicts close to uniform**: either the model was trained purely by self-play (collapsed to Nash) or the curriculum fraction was too small. Train with higher `curriculum_mix` to induce exploitability.
* **Very large checkpoint (1e9 steps)**: such training may overfit synthetic patterns or produce brittle policies. Evaluate against held-out curriculum and humans.
* **Performance**: for training at scale (millions to billions of simulated steps), prefer GPU (`--device cuda`), larger `batch_size`, and consider using mixed precision (AMP) to increase throughput.

---

## Recommended advanced enhancements

1. **Save full optimizer checkpoints** (`optimizer.state_dict()` + scheduler + step counter) to support exact resume.
2. **Replay buffer**: persist diverse curriculum and self-play episodes to disk and sample from them to improve stability.
3. **Temperature control for self-play sampling**: use controlled softmax temperature when sampling opponent moves from the current model to encourage exploration/exploitation balance.
4. **Probabilistic ensemble**: combine predictors' softmax outputs (rather than hard `argmax`) using a calibrated weight vector.
5. **RL policy learning**: shift from supervised next-move prediction to policy optimization (e.g., PPO) to directly maximize win-rate instead of predictive accuracy.

---

## Final notes about your uploaded `*.pt` file

You mentioned: *"I uploaded a file `.pt` with `1000000000` next to the project on GitHub"* — I assume this is a trained checkpoint produced by the same project (trained for \~1e9 simulation steps). To use it:

* Place the file in the project folder and pass its filename to `--model_path`.
* Verify compatibility with the snippet above. If incompatible, inspect the saved `cfg` (if present) to find the `history_len` and other hyperparameters used at save time.
* If the checkpoint is large, download it locally and use `--device cuda` (recommended) when loading/inference to avoid slow CPU inference.

---

## License & credits

This code and README are provided as-is. If you incorporate the model weights or reproduce results, please reference the project and note whether curriculum/self-play settings were used.

---

If you want, I can now:

* produce this text as a `README.md` file in the repo (I can create the file content here),
* or generate a smaller `USAGE.md` with step-by-step commands and explain how to add optimizer checkpoints for resumable training.

Which do you prefer?


By mohammad taha gorji
