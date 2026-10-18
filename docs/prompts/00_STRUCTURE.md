Create the project scaffold for a from-scratch implementation of the original
Transformer ("Attention Is All You Need"), and set up its Python environment.
I am writing all model code myself as a learning exercise, so you must create
the STRUCTURE ONLY.

STRICT RULES
- Do NOT write any implementation: no functions, classes, imports, logic, or test cases.
- Every .py file contains ONLY a module docstring describing its purpose (use the
  descriptions in the tree below). __init__.py files may be empty.
- Do NOT run git init or any git command. I will handle git myself.
- Do not download any datasets.

STRUCTURE
transformer-from-scratch/
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── docs/                 # Leave empty. I will add docs/MILESTONES.md myself.
├── transformer/
│   ├── __init__.py
│   ├── attention.py      # Scaled dot-product attention (M1) and multi-head attention (M2)
│   ├── masks.py          # Causal mask, padding mask, and combining them
│   ├── layers.py         # LayerNorm, position-wise feed-forward, sinusoidal positional encoding, token embeddings (M3)
│   ├── encoder.py        # Encoder layer and encoder stack (M4)
│   ├── decoder.py        # Decoder layer and decoder stack (M5)
│   └── model.py          # Full encoder-decoder Transformer (M6)
├── tests/
│   ├── __init__.py
│   ├── test_attention.py
│   ├── test_masks.py
│   ├── test_layers.py
│   ├── test_encoder.py
│   ├── test_decoder.py
│   └── test_model.py
├── experiments/
│   ├── __init__.py
│   ├── reverse_task/
│   │   ├── __init__.py
│   │   └── train.py      # Toy task: learn to reverse digit sequences (M7)
│   └── translation/
│       ├── __init__.py
│       ├── data.py       # Dataset loading, tokenizer, batching with padding (M8)
│       ├── train.py      # Training loop with validation and checkpointing (M8)
│       └── decode.py     # Greedy decoding and evaluation (M8)
├── notebooks/
│   └── scratch.ipynb     # Valid, empty Jupyter notebook
├── notes/
│   ├── milestone-01.md ... milestone-08.md
├── data/
│   └── .gitkeep
└── checkpoints/
    └── .gitkeep

FILE CONTENTS
- pytest.ini: [pytest] section with testpaths = tests and pythonpath = .
- .gitignore: standard Python ignores, virtual envs, .ipynb_checkpoints/,
  *.pt, *.pth, runs/, and data/* and checkpoints/* with exceptions
  !data/.gitkeep and !checkpoints/.gitkeep.
- README.md: title "Transformer from Scratch", a one-line description
  ("The original encoder-decoder Transformer implemented from the paper, without
  tutorials or reference code."), a "Progress" section with an unchecked
  checklist of the 8 milestones below, a "Setup" section with the exact
  commands to recreate the environment, and a "Rules" section stating:
  nn.MultiheadAttention, nn.LayerNorm, nn.Transformer* and
  F.scaled_dot_product_attention are not used inside the transformer/ package;
  they may be used in tests/ only as numerical references.
  Milestones:
    M1 Scaled dot-product attention and masks
    M2 Multi-head attention
    M3 LayerNorm, feed-forward, positional encoding, embeddings
    M4 Encoder
    M5 Decoder
    M6 Full Transformer
    M7 Toy task: sequence reversal
    M8 Translation
- Each notes/milestone-XX.md: a title with the milestone name and these empty
  headings: "Concepts", "Answers to questions", "Bugs I hit and their causes",
  "What I learned".

ENVIRONMENT
- My machine: Ubuntu, NVIDIA RTX 5070 Ti (Blackwell architecture). I use conda.
- Create a conda environment named "transformer-scratch" with the latest Python
  version that current stable PyTorch officially supports.
- Install the latest STABLE (not nightly, not pre-release) PyTorch built for
  CUDA 12.8 or newer, using the official PyTorch install instructions.
  Older CUDA builds do not support Blackwell GPUs.
- Then install the latest stable versions of: pytest, numpy, matplotlib, ipykernel.
- Write requirements.txt with the EXACT versions that were installed for these
  packages only (not a full pip freeze). Add a comment line at the top with the
  PyTorch index URL used for torch.
- Verify the setup and show me the output:
  1. torch version and CUDA version torch was built with
  2. torch.cuda.is_available() is True
  3. the GPU name
  4. a small matrix multiplication on the GPU runs without errors
  5. pytest runs (it will report no tests collected; that is expected)
- If any check fails, stop and explain the problem instead of working around it.

When done, print the tree and the verification output, and stop.