```
ssa-prompt-tuning/
│
├── README.md
├── requirements.txt
├── config.yaml                    # Hyperparameters, paths, LLM settings
│
├── data/
│   ├── raw/
│   │   └── sentiment_dataset.csv
│   ├── processed/
│   │   ├── train.csv
│   │   ├── dev.csv
│   │   ├── validation.csv
│   │   └── test.csv
│   └── results/
│       ├── evolution_logs/
│       ├── best_prompts/
│       └── performance_metrics/
│
├── src/
│   ├── __init__.py
│   │
│   ├── prompt/
│   │   ├── __init__.py
│   │   ├── base_prompt.py         # Base prompt templates
│   │   ├── prompt_parser.py       # Parse prompt into components
│   │   ├── prompt_builder.py      # Reconstruct prompt from components
│   │   └── mutation_libraries.py  # Synonym maps, templates, etc.
│   │
│   ├── genome/
│   │   ├── __init__.py
│   │   ├── genome.py              # Genome class definition
│   │   ├── decoder.py             # Genome → Prompt mutations
│   │   └── validator.py           # Ensure genome validity
│   │
│   ├── ssa/
│   │   ├── __init__.py
│   │   ├── squirrel.py            # Squirrel agent class
│   │   ├── population.py          # Population management
│   │   ├── ssa_optimizer.py       # Main SSA algorithm
│   │   └── movement.py            # SSA movement operators
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── llm_interface.py       # API calls to LLM (OpenAI/Anthropic/etc)
│   │   ├── fitness.py             # Fitness calculation (accuracy, F1, etc)
│   │   ├── evaluator.py           # Run prompts on dataset
│   │   └── cache.py               # Cache LLM responses
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py              # Logging utilities
│   │   ├── metrics.py             # Evaluation metrics
│   │   └── visualization.py       # Plot evolution curves, etc.
│   │
│   └── experiments/
│       ├── __init__.py
│       └── experiment_runner.py   # Orchestrate full experiments
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_prompts.ipynb
│   ├── 03_mutation_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── tests/
│   ├── __init__.py
│   ├── test_genome.py
│   ├── test_decoder.py
│   ├── test_ssa.py
│   └── test_evaluation.py
│
├── scripts/
│   ├── prepare_data.py            # Split dataset
│   ├── run_baseline.py            # Test manual prompts
│   ├── run_optimization.py        # Main SSA optimization
│   └── analyze_results.py         # Statistical analysis
│
└── outputs/
    ├── logs/
    ├── checkpoints/                # Save population states
    └── final_results/
```