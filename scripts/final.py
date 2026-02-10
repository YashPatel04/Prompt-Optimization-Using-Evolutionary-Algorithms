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


def load_experiment_config(config_data: dict) -> ExperimentConfig:
    """
    Load experiment configuration directly from config.yaml.
    No user prompts.
    
    Args:
        config_data: Configuration dictionary from YAML
    
    Returns:
        ExperimentConfig object
    """
    
    # Resolve dataset size
    dataset_size_key = config_data['dataset']['default_size']
    if dataset_size_key in config_data['dataset']['sizes']:
        dataset_size = config_data['dataset']['sizes'][dataset_size_key]
    else:
        # Assume it's a custom number
        try:
            dataset_size = int(dataset_size_key)
        except (ValueError, TypeError):
            print(f"[FAIL] Invalid dataset size: {dataset_size_key}")
            sys.exit(1)
    
    # Generate experiment name with dataset size
    experiment_name = f"ssa_sentiment_{dataset_size}_samples"
    
    # Create config from file settings
    config = ExperimentConfig(
        dataset_size=dataset_size,
        population_size=config_data['ssa']['population_size'],
        max_iterations=config_data['ssa']['max_iterations'],
        Gc=config_data['ssa']['Gc'],
        Pdp=config_data['ssa']['Pdp'],
        early_stopping_patience=config_data['ssa']['early_stopping_patience'],
        ollama_model=config_data['llm']['model'],
        ollama_base_url=config_data['llm']['base_url'],
        ollama_temperature=config_data['llm']['temperature'],
        use_cache=config_data['evaluation']['use_cache'],
        fitness_metrics=config_data['evaluation']['fitness_metric'],
        experiment_name=experiment_name,
        output_dir=config_data['experiment']['output_dir'],
        seed=config_data['experiment']['seed'],
        checkpoint_interval=config_data['evaluation']['checkpoint_interval']
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
    print("\nLoading configuration from config.yaml...")
    config_data = load_config_file('config.yaml')
    print("[OK] Configuration loaded\n")
    
    # Check if Ollama is running
    print("Verifying Ollama connection...")
    try:
        from src.evaluation.llm_interface import OllamaInterface
        test_llm = OllamaInterface(
            base_url=config_data['llm']['base_url'],
            model=config_data['llm']['model']
        )
        print("[OK] Ollama is running and ready!\n")
    except Exception as e:
        print(f"\n[FAIL] Error: Cannot connect to Ollama")
        print(f"  {e}")
        print(f"\nPlease start Ollama:")
        print(f"  1. Run: ollama serve")
        print(f"  2. In another terminal, run: ollama pull {config_data['llm']['model']}")
        sys.exit(1)
    
    # Load configuration from file (no prompts)
    config = load_experiment_config(config_data)
    
    # Display configuration
    print("=" * 60)
    print("Configuration Summary")
    print("=" * 60)
    print(f"Dataset size:          {config.dataset_size}")
    print(f"Population size:       {config.population_size}")
    print(f"Max iterations:        {config.max_iterations}")
    print(f"LLM model:             {config.ollama_model}")
    print(f"Base URL:              {config.ollama_base_url}")
    print(f"Temperature:           {config.ollama_temperature}")
    print(f"Gc (gravity):          {config.Gc}")
    print(f"Pdp (predation):       {config.Pdp}")
    print(f"Early stopping:        {config.early_stopping_patience} iterations")
    print(f"Experiment name:       {config.experiment_name}")
    print(f"Output directory:      {config.output_dir}")
    print("=" * 60 + "\n")
    
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