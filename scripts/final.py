import os
import sys
import io
import yaml
from pathlib import Path
from typing import Optional

# Force UTF-8 encoding for Windows
if sys.platform == 'win32':
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(
            sys.stdout.buffer,
            encoding='utf-8',
            errors='replace'
        )
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(
            sys.stderr.buffer,
            encoding='utf-8',
            errors='replace'
        )

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig
from src.utils.logger import get_logger


def load_config_file(config_path: str = 'config.yaml') -> dict:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config.yaml
    
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Loaded configuration from {config_path}\n")
        return config
    except FileNotFoundError:
        print(f"[FAIL] Config file not found: {config_path}")
        print("Make sure config.yaml exists in the project root")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"[FAIL] Error parsing config file: {e}")
        sys.exit(1)


def get_user_config(config_data: dict) -> ExperimentConfig:
    """
    Interactive configuration builder using config file defaults.
    Prompts user for experiment settings.
    
    Args:
        config_data: Configuration dictionary from YAML
    
    Returns:
        ExperimentConfig object
    """
    
    print("\n" + "="*60)
    print("Configure Your Experiment")
    print("="*60 + "\n")
    
    # Dataset size
    print("Dataset size options:")
    print(f"  Tiny:    {config_data['dataset']['sizes']['tiny']}")
    print(f"  Small:   {config_data['dataset']['sizes']['small']}")
    print(f"  Medium:  {config_data['dataset']['sizes']['medium']}")
    print(f"  Large:   {config_data['dataset']['sizes']['large']}")
    print(f"  Full:    {config_data['dataset']['sizes']['full']}")
    print("  Or enter custom size (10-31000)")
    
    while True:
        try:
            dataset_size_input = input("\nEnter dataset size (or name: tiny/small/medium/large/full): ").strip().lower()
            
            # Check if it's a preset name
            if dataset_size_input in config_data['dataset']['sizes']:
                dataset_size = config_data['dataset']['sizes'][dataset_size_input]
                break
            else:
                dataset_size = int(dataset_size_input)
                if 10 <= dataset_size <= 31000:
                    break
                else:
                    print("[FAIL] Size must be between 10 and 31000")
        except ValueError:
            print("[FAIL] Invalid input. Enter a number or preset name (tiny/small/medium/large/full)")
    
    # Extract defaults from config file
    default_pop_size = config_data['ssa']['population_size']
    default_iterations = config_data['ssa']['max_iterations']
    default_model = config_data['llm']['default_model']
    default_patience = config_data['ssa']['early_stopping_patience']
    default_gc = config_data['ssa']['Gc']
    default_pdp = config_data['ssa']['Pdp']
    base_url = config_data['llm']['base_url']
    temperature = config_data['llm']['temperature']
    
    # Population size
    print(f"\nPopulation size (default: {default_pop_size}):")
    population_size_input = input("Enter population size (or press Enter for default): ").strip()
    population_size = int(population_size_input) if population_size_input else default_pop_size
    
    # Iterations
    print(f"\nMax iterations (default: {default_iterations}):")
    iterations_input = input("Enter max iterations (or press Enter for default): ").strip()
    max_iterations = int(iterations_input) if iterations_input else default_iterations
    
    print(f"\nAvailable LLM models from config:")
    for model_type, model_name in config_data['llm']['models'].items():
        print(f"  - {model_name} ({model_type})")
    print(f"\nDefault model: {default_model}")
    
    model_mapping = {
        "fast": config_data['llm']['models']['fast'],           # gemma3:270m
        "balanced": config_data['llm']['models']['balanced'],   # llama3.1:8b
        "powerful": config_data['llm']['models']['powerful'],   # qwen2.5:32b
    }

    # User input
    model_input = input("Enter model name (or press Enter for default): ").strip()

    if not model_input:
        ollama_model = config_data['llm']['default_model']  # gemma3:270m
    elif model_input in model_mapping:
        ollama_model = model_mapping[model_input]  # Resolve label to actual name
    else:
        ollama_model = model_input  # User entered actual model name

    print(f"Using model: {ollama_model}")
    
    # Gravitational coefficient
    print(f"\nGravitational coefficient Gc (default: {default_gc}):")
    gc_input = input("Enter Gc value (or press Enter for default): ").strip()
    Gc = float(gc_input) if gc_input else default_gc
    
    # Predation probability
    print(f"\nPredation probability Pdp (default: {default_pdp}):")
    pdp_input = input("Enter Pdp value (or press Enter for default): ").strip()
    Pdp = float(pdp_input) if pdp_input else default_pdp
    
    # Early stopping
    print(f"\nEarly stopping patience (default: {default_patience}):")
    patience_input = input("Enter patience (or press Enter for default): ").strip()
    early_stopping_patience = int(patience_input) if patience_input else default_patience
    
    # Experiment name
    experiment_name = input("\nExperiment name (default: ssa_sentiment_optimization): ").strip()
    if not experiment_name:
        experiment_name = f"ssa_sentiment_{dataset_size}_samples"
    
    print("\n" + "="*60)
    print("Configuration Summary")
    print("="*60)
    print(f"Dataset size:          {dataset_size}")
    print(f"Population size:       {population_size}")
    print(f"Max iterations:        {max_iterations}")
    print(f"LLM model:             {ollama_model}")
    print(f"Base URL:              {base_url}")
    print(f"Temperature:           {temperature}")
    print(f"Gc (gravity):          {Gc}")
    print(f"Pdp (predation):       {Pdp}")
    print(f"Early stopping:        {early_stopping_patience} iterations")
    print(f"Experiment name:       {experiment_name}")
    print("="*60 + "\n")
    
    confirm = input("Proceed with this configuration? (y/n): ").strip().lower()
    if confirm != 'y':
        print("[FAIL] Configuration cancelled")
        sys.exit(0)
    
    # Create config
    config = ExperimentConfig(
        dataset_size=dataset_size,
        population_size=population_size,
        max_iterations=max_iterations,
        Gc=Gc,
        Pdp=Pdp,
        ollama_model=ollama_model,
        ollama_base_url=base_url,
        ollama_temperature=temperature,
        early_stopping_patience=early_stopping_patience,
        use_cache=config_data['evaluation']['use_cache'],
        experiment_name=experiment_name,
        output_dir=config_data['experiment']['output_dir'],
        seed=config_data['experiment']['seed']
    )
    
    return config


def main():
    """Main entry point"""
    print("\n")
    print("=" * 60)
    print("SSA PROMPT OPTIMIZATION EXPERIMENT".center(60))
    print("Sentiment Classification".center(60))
    print("=" * 60)
    
    # Load config file
    print("\nLoading configuration file...")
    config_data = load_config_file('config.yaml')
    
    # Check if Ollama is running
    print("Verifying Ollama connection...")
    try:
        from src.evaluation.llm_interface import OllamaInterface
        test_llm = OllamaInterface(
            base_url=config_data['llm']['base_url'],
            model=config_data['llm']['default_model']
        )
        print("[OK] Ollama is running and ready!\n")
    except Exception as e:
        print(f"\n[FAIL] Error: Cannot connect to Ollama")
        print(f"  {e}")
        print(f"\nPlease start Ollama:")
        print(f"  1. Run: ollama serve")
        print(f"  2. In another terminal, run: ollama pull {config_data['llm']['default_model']}")
        sys.exit(1)
    
    # Get configuration
    config = get_user_config(config_data)
    
    # Run experiment
    try:
        runner = ExperimentRunner(config)
        results = runner.run_full_experiment()
        
        print("\n[OK] Experiment completed successfully!")
        print(f"\nResults saved to: {runner.dirs['results']}")
        print(f"Visualizations saved to: {runner.dirs['figures']}")
        print(f"Logs saved to: {runner.dirs['logs']}")
        
    except KeyboardInterrupt:
        print("\n\n[FAIL] Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()