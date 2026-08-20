FANC-Torch
--------------------

This directory contains a PyTorch implementation of FANC. This implementation is based on the SAS 2026 artifact for [*Uncovering the Limits of Proof Sharing for Neural Networks*](https://zenodo.org/records/21314767).

Overview
--------------------

For each selected image, FANC creates templates: verified boxes at an intermediate network layer. It then:

1. generates templates from patches using the original network;
2. transfers and validates them on the selected target network; and
3. uses matching templates to verify new specifications, falling back to DeepZ when none match.

Setup and Tests
--------------------

Run from the repository root:

```shell
python -m pip install -r fanc-torch/requirements.txt
PYTHONPATH=fanc-torch python -m unittest discover -s fanc-torch/tests -v
```

Run
--------------------
Example:

```shell
python fanc-torch/scripts/run_suite.py \
  --output output/fcnn7 \
  --network fcnn7 --target-variant float16 \
  --attack patch --attack l0_random \
  --images 10 --transfer-radius 0.001 \
  --batch-size 8 --repetitions 3
```

The output directory contains:

| File | Contents |
| --- | --- |
| `experiment.json` | Selected images and attacks |
| `result-N.json` | Verification counts and timing for one run |
| `summary.json` | Counts and phase timing statistics across runs |


Options
--------------------

| Option | Default | Meaning |
| --- | --- | --- |
| `--output` | `output` | Result directory |
| `--network` | `fcnn7` | Original network |
| `--target-variant` | `quant8` | `float16`, `quant16`, or `quant8` version of that network |
| `--attack` | All | Perturbation type: `patch`, `l0_random`, or `l0_center`; repeat to combine |
| `--transfer-radius` | 0 | Expansion around the target network activation |
| `--seed` | 2022 | Image and attack selection seed |
| `--images` | 2 | Number of images |
| `--spec-limit` | All | Maximum specifications per perturbation type and image |
| `--batch-size` | 8 | Number of specifications verified at a time |
| `--threads` | 1 | PyTorch CPU threads |
| `--repetitions` | 1 | Runs using the same experiment |

Run `python fanc-torch/scripts/run_suite.py --help` for the full command reference.

Networks
--------------------

| Network | Input | Template patch size | Template layer |
| --- | --- | ---: | ---: |
| `fcnn7` (FCN7-MNIST) | 1x28x28 | 7x7 | 4 |
| `fconv4` (CONV2-MNIST) | 1x28x28 | 14x14 | 6 |
| `fcnn7_cifar` (FCN7-CIFAR) | 3x32x32 | 16x16 | 6 |
| `fconv4_cifar` (CONV4-CIFAR) | 3x32x32 | 32x32 | 5 |

Template radii are tested at `1, 1/2, ..., 1/256, 0`, and the first verified value is kept for each patch. The template layer is a zero based PyTorch module index. The transfer radius expands a template around the target network's activation at that layer.
