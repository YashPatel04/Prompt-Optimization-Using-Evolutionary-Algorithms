import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import random
import sys
from scipy import stats

from src.ssa.ssa_optimizer import SSAOptimizer
from src.genome.genome import GenomeConfig
from src.genome.decoder import GenomeDecoder
from src.evaluation.llm_interface import OllamaInterface
from src.evaluation.fitness import FitnessCalculator
from src.evaluation.evaluator import Evaluator
from src.evaluation.cache import PromptCache
from src.utils.logger import get_logger
from src.utils.metrics import MetricsCalculator, StatisticsCalculator
from src.utils.visualizer import Visualizer
from src.utils.progress import ProgressBar, Spinner, IterationTracker, StatusMessage
from src.utils.analyze_results import ResultsAnalyzer
from src.utils.token_tracker import TokenTracker


@dataclass
class ExperimentConfig:
    """Configuration for SSA experiment"""
    
    # Dataset settings
    dataset_size: int = 500  # How many samples to use
    
    # SSA settings
    population_size: int = 20
    max_iterations: int = 50
    Gc: float = 1.9  # Gravitational coefficient
    Pdp: float = 0.1  # Predation probability
    early_stopping_patience: int = 10
    
    # LLM settings
    ollama_model: str = "gemma3:270m"
    ollama_base_url: str = "http://localhost:11434"
    ollama_temperature: float = 0.0
    
    # Evaluation settings
    use_cache: bool = True
    fitness_metrics: str = "combined"
    
    # Experiment settings
    experiment_name: str = "ssa_sentiment_optimization"
    seed: int = 42
    output_dir: str = "outputs"
    checkpoint_interval: int = 5
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return asdict(self)
    
    def save(self, filepath: str):
        """Save config to JSON"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ExperimentConfig':
        """Load config from dictionary"""
        return cls(**data)
    
    @classmethod
    def from_file(cls, filepath: str) -> 'ExperimentConfig':
        """Load config from JSON file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


class ExperimentRunner:
    """
    Full SSA optimization experiment pipeline.
    Orchestrates data loading, LLM evaluation, optimization, and analysis.
    """

    def __init__(self, config: ExperimentConfig):
        """
        Initialize experiment runner.
        
        Args:
            config: ExperimentConfig object
        """
        self.config = config
        
        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        
        # Create output directories first
        self.setup_output_dirs()
        
        # Initialize logger with correct log directory
        self.logger = get_logger(log_dir=str(self.dirs['logs']))
        
        # Initialize components
        self.dataset = None
        self.eval_data = None
        
        self.llm = None
        self.fitness_calculator = None
        self.cache = None
        self.evaluator = None
        
        self.genome_decoder = None
        self.optimizer = None
        self.visualizer = None
        
        self.token_tracker = TokenTracker()
        
        self.chkpt_cnt = 0
        
        self.results = {
            'config': config.to_dict(),
            'history': None,
            'best_squirrel': None,
            'best_prompt': None,
            'metrics': None,
            'statistics': None,
            'timestamp': datetime.now().isoformat()
        }
    
    def setup_output_dirs(self):
        """Create output directory structure"""
        base_dir = Path(self.config.output_dir)
        
        self.dirs = {
            'base': base_dir,
            'logs': base_dir / 'logs',
            'figures': base_dir / 'figures',
            'results': base_dir / 'results',
            'checkpoints': base_dir / 'checkpoints',
            'cache': base_dir / 'cache',
            'analysis': base_dir / 'analysis'
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self, iteration: int, best_fitness: float, population_data: Dict):
        """Save checkpoint during optimization"""
        chkpt_data = {
            'iteration': iteration,
            'best_fitness': float(best_fitness),
            'timestamp': datetime.now().isoformat(),
            'population': population_data,
            'config': self.config.to_dict()
        }
        
        chkpt_file = self.dirs['checkpoints'] / f"checkpoint_iter_{iteration:04d}.json"
        with open(chkpt_file, 'w') as f:
            json.dump(chkpt_data, f, indent=2)
        
        self.chkpt_cnt += 1
        return chkpt_file
    
    def save_best_checkpoint(self, best_squirrel, iteration: int):
        """Save checkpoint for best solution found"""
        best_prompt = self.genome_decoder.decode(best_squirrel.genome)
        
        best_chkpt = {
            'iteration': iteration,
            'fitness': float(best_squirrel.fitness),
            'genome': best_squirrel.genome.vector.tolist(),
            'prompt': best_prompt,
            'squirrel_type': best_squirrel.squirrel_type,
            'timestamp': datetime.now().isoformat()
        }
        
        chkpt_file = self.dirs['checkpoints'] / "best_solution.json"
        with open(chkpt_file, 'w') as f:
            json.dump(best_chkpt, f, indent=2)  
        self.chkpt_cnt += 1
        return chkpt_file

    def load_dataset(self, csv_path: str = 'data/processed/evaluation.csv'):
        """
        Load dataset for evaluation.
        
        Args:
            csv_path: Path to evaluation.csv file
        """
        print("\n" + "="*80)
        print("LOADING DATASET".center(80))
        print("="*80 + "\n")
        self.logger.info("")
        self.logger.section("LOADING DATASET")
        
        try:
            spinner = Spinner("Loading dataset")
            spinner.start()
            
            df = pd.read_csv(csv_path)
            spinner.stop(f"Loaded {len(df)} samples")
            self.logger.info(f"Loaded {len(df)} samples from {csv_path}")
            
            # Sample if needed
            if len(df) > self.config.dataset_size:
                spinner = Spinner(f"Sampling {self.config.dataset_size} examples")
                spinner.start()
                df = df.sample(n=self.config.dataset_size, random_state=self.config.seed)
                spinner.stop(f"Sampled {self.config.dataset_size} examples")
                self.logger.info(f"Sampled {self.config.dataset_size} examples")
            
            # Show label distribution
            self.logger.info("Label distribution:")
            for label in ['positive', 'negative', 'neutral']:
                count = (df['label'] == label).sum()
                pct = 100 * count / len(df) if len(df) > 0 else 0
                bar_len = int(20 * pct / 100)
                bar = '█' * bar_len + '░' * (20 - bar_len)
                self.logger.info(f"  {label:10s} {bar} {pct:5.1f}%")
            
            # Convert to list of dicts
            self.eval_data = df.to_dict('records')
            
            self.logger.info(f"Total evaluation samples: {len(self.eval_data)}")
            
        except FileNotFoundError:
            self.logger.error(f"Dataset file not found: {csv_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading dataset: {e}")
            raise
    
    def initialize_llm(self):
        """Initialize Ollama LLM interface"""
        print("\n" + "="*80)
        print("INITIALIZING LLM".center(80))
        print("="*80 + "\n")
        self.logger.section("INITIALIZING LLM")
        
        try:
            spinner = Spinner("Connecting to Ollama")
            spinner.start()
            
            self.llm = OllamaInterface(
                base_url=self.config.ollama_base_url,
                model=self.config.ollama_model,
                temperature=self.config.ollama_temperature
            )
            
            spinner.stop(f"Connected to {self.config.ollama_model}")
            self.logger.info(f"Connected to {self.config.ollama_model}")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to LLM: {e}")
            raise
    
    def initialize_evaluation(self):
        """Initialize fitness calculator and evaluator"""
        print("\n" + "="*80)
        print("INITIALIZING EVALUATION".center(80))
        print("="*80 + "\n")
        self.logger.section("INITIALIZING EVALUATION")
        
        spinner = Spinner("Setting up evaluation")
        spinner.start()
        
        self.fitness_calculator = FitnessCalculator(task='sentiment_classification')
        
        self.cache = PromptCache(
            cache_dir=str(self.dirs['cache']),
            use_disk=self.config.use_cache
        )
        
        self.evaluator = Evaluator(
            llm_interface=self.llm,
            fitness_calculator=self.fitness_calculator,
            cache=self.cache,
        )
        
        spinner.stop("Evaluation components ready")
        self.logger.info(f"Cache enabled: {self.config.use_cache}")
        self.logger.info(f"Fitness metric: {self.config.fitness_metrics}")
        
    def initialize_optimizer(self):
        """Initialize genome decoder and SSA optimizer"""
        print("\n" + "="*80)
        print("INITIALIZING OPTIMIZER".center(80))
        print("="*80 + "\n")
        self.logger.section("INITIALIZING OPTIMIZER")
        
        spinner = Spinner("Initializing SSA optimizer")
        spinner.start()
        
        self.genome_decoder = GenomeDecoder()
        
        genome_config = GenomeConfig(dimensions=10)
        
        self.optimizer = SSAOptimizer(
            population_size=self.config.population_size,
            max_iterations=self.config.max_iterations,
            Gc=self.config.Gc,
            Pdp=self.config.Pdp,
            genome_config=genome_config
        )
        
        spinner.stop("SSA optimizer ready")
        self.logger.info(f"Population size: {self.config.population_size}")
        self.logger.info(f"Max iterations: {self.config.max_iterations}")
        self.logger.info(f"Gc: {self.config.Gc}, Pdp: {self.config.Pdp}")
        self.logger.info(f"Checkpoint interval: {self.config.checkpoint_interval}")
    
    def initialize_visualization(self):
        """Initialize visualizer"""
        spinner = Spinner("Initializing visualizer")
        spinner.start()
        
        self.visualizer = Visualizer(output_dir=str(self.dirs['figures']))
        
        spinner.stop("Visualizer ready")
        self.logger.info("Visualizer ready")
    
    def create_fitness_function(self):
        """
        Create fitness function for SSA optimization.
        
        Returns:
            Callable fitness function
        """
        def fitness_fn(squirrel):
            """
            Evaluate squirrel and return fitness.
            Uses evaluation set for evaluation.
            """
            try:
                # Decode genome to prompt
                prompt = self.genome_decoder.decode(squirrel.genome)
                
                # Evaluate on eval set
                fitness = self.evaluator.evaluate_prompt(
                    prompt,
                    self.eval_data,
                    sample_size=min(len(self.eval_data), 100)
                )
                
                # Get tokens used in this evaluation
                token_stats = self.evaluator.get_token_stats()
                current_total = token_stats['total_input_tokens'] + token_stats['total_output_tokens']
                
                # Add tokens for this iteration
                if current_total > self._last_total_tokens:
                    new_tokens_input = token_stats['total_input_tokens'] - (self._last_evaluator_input or 0)
                    new_tokens_output = token_stats['total_output_tokens'] - (self._last_evaluator_output or 0)
                    self.token_tracker.add_tokens(new_tokens_input, new_tokens_output, self._curr_iter)
                    self._last_evaluator_input = token_stats['total_input_tokens']
                    self._last_evaluator_output = token_stats['total_output_tokens']
                
                return fitness
            
            except Exception as e:
                self.logger.warning(f"Error evaluating squirrel: {e}")
                return 1.0  # Return worst fitness on error
        
        return fitness_fn
    
    def _get_population_data(self) -> Dict:
        """Extract population data for checkpoint"""
        pop_data = {
            'squirrels': [],
            'stats': {}
        }
        
        if self.optimizer and self.optimizer.population:
            pop_data['stats'] = self.optimizer.population.get_fitness_stats()
            for sqrl in self.optimizer.population.squirrels:
                sqrl_data = {
                    'genome': sqrl.genome.vector.tolist(),
                    'fitness': float(sqrl.fitness) if sqrl.fitness is not None else None,
                    'type': sqrl.squirrel_type,
                    'evaluated': sqrl.evaluated
                }
                pop_data['squirrels'].append(sqrl_data)
        
        return pop_data
    
    def run_optimization(self):
        """Run SSA optimization using SSAOptimizer.optimize()"""
        print("\n" + "="*80)
        print("RUNNING SSA OPTIMIZATION".center(80))
        print("="*80 + "\n")
        self.logger.section("RUNNING SSA OPTIMIZATION")
        
        self.logger.info(f"Dataset: {len(self.eval_data)} evaluation samples")
        self.logger.info(f"Population: {self.config.population_size} squirrels")
        self.logger.info(f"Max iterations: {self.config.max_iterations}")
        self.logger.info(f"LLM: {self.config.ollama_model}")
        self.logger.info(f"Checkpoints: every {self.config.checkpoint_interval} iterations")
        
        fitness_fn = self.create_fitness_function()
        self._init_bar = None
        self._tracker = None
        self._last_best_fitness = None
        self._curr_iter = 0
        self._last_total_tokens = 0
        self._last_evaluator_input = 0
        self._last_evaluator_output = 0
        
        def progress_callback(iteration, best_fitness, improved, phase):
            """Handle progress updates from optimizer"""
            self._curr_iter = iteration
            
            if phase == 'init_start':
                spinner = Spinner("Initializing population")
                spinner.start()
                spinner.stop("Population initialized")
                print()
                self._init_bar = ProgressBar(
                    self.config.population_size, 
                    "Evaluating initial population"
                )
                self.token_tracker.set_iteration(-1)  # Use -1 for initialization
            
            elif phase == 'init_eval':
                self._init_bar.update()
            
            elif phase == 'init_complete':
                self._init_bar.finish()
                print(f"Initial best fitness: {best_fitness:.6f}\n")
                self._tracker = IterationTracker(self.config.max_iterations)
                self._last_best_fitness = best_fitness
                
                # Log initialization token usage
                init_tokens = self.token_tracker.get_iteration_tokens(-1)
                self.logger.info(f"Initialization tokens - Input: {init_tokens['input_tokens']}, Output: {init_tokens['output_tokens']}, Total: {init_tokens['total_tokens']}")
                
                pop_data = self._get_population_data()
                self.save_checkpoint(0, best_fitness, pop_data)
                
            elif phase == 'iter_start':
                self._tracker.start_iteration(iteration)
                self.token_tracker.set_iteration(iteration)
            
            elif phase == 'iter_complete':
                self._tracker.end_iteration(best_fitness, improved)
                
                # Log token usage for this iteration
                iter_tokens = self.token_tracker.get_iteration_tokens(iteration)
                self.logger.debug(f"Iteration {iteration} tokens - Input: {iter_tokens['input_tokens']}, Output: {iter_tokens['output_tokens']}, Total: {iter_tokens['total_tokens']}")
                
                if iteration % self.config.checkpoint_interval == 0:
                    pop_data = self._get_population_data()
                    self.save_checkpoint(iteration, best_fitness, pop_data)
                
                if improved and self.optimizer.best_squirrel:
                    self.save_best_checkpoint(self.optimizer.best_squirrel, iteration)
                
            elif phase == 'early_stop':
                print()
                print(f"\nEarly stopping triggered "
                      f"(no improvement for {self.config.early_stopping_patience} iterations)")
                
                pop_data = self._get_population_data()
                self.save_checkpoint(iteration, best_fitness, pop_data)
                
            elif phase == 'complete':
                self._tracker.finish()
                
                pop_data = self._get_population_data()
                self.save_checkpoint(iteration, best_fitness, pop_data)
        
        # Run optimization
        best_squirrel, history = self.optimizer.optimize(
            fitness_function=fitness_fn,
            early_stopping_patience=self.config.early_stopping_patience,
            progress_callback=progress_callback,
            verbose=False
        )
        
        # Store results
        self.save_best_checkpoint(best_squirrel, len(history['best_fitness']) - 1)
        
        self.results['best_squirrel'] = {
            'fitness': float(best_squirrel.fitness),
            'genome': best_squirrel.genome.vector.tolist()
        }
        self.results['history'] = history
        
        # Store token tracking information
        token_summary = self.token_tracker.get_summary()
        self.results['token_tracking'] = self.token_tracker.to_dict()
        
        self.logger.info(f"Optimization complete! Improvements: {history['improvements']}")
        self.logger.info(f"Best fitness: {best_squirrel.fitness:.6f}")
        self.logger.info(f"Checkpoints saved: {self.chkpt_cnt}")
        self.logger.info("")
        self.logger.info("TOKEN USAGE SUMMARY")
        self.logger.info(f"  Total input tokens: {token_summary['total']['total_input_tokens']}")
        self.logger.info(f"  Total output tokens: {token_summary['total']['total_output_tokens']}")
        self.logger.info(f"  Total tokens: {token_summary['total']['total_tokens']}")
        self.logger.info(f"  Total LLM calls: {token_summary['total_calls']}")
        self.logger.info(f"  Average tokens per call: {token_summary['avg_tokens_per_call']:.2f}")
        self.logger.info(f"  Average input per call: {token_summary['avg_input_per_call']:.2f}")
        self.logger.info(f"  Average output per call: {token_summary['avg_output_per_call']:.2f}")
        
        return best_squirrel

    def evaluate_best_prompt(self, best_squirrel):
        """
        Evaluate best prompt on full dataset.
        Get detailed metrics.
        """
        print("\n" + "="*80)
        print("EVALUATING BEST PROMPT".center(80))
        print("="*80 + "\n")
        self.logger.section("EVALUATING BEST PROMPT")
        
        # Decode best genome
        best_prompt = self.genome_decoder.decode(best_squirrel.genome)
        self.results['best_prompt'] = best_prompt
        
        self.logger.info("Best prompt found:")
        self.logger.info(f"{best_prompt}")
        
        # Evaluate on full dataset
        spinner = Spinner("Evaluating on full dataset")
        spinner.start()
        
        eval_fitness = self.evaluator.evaluate_prompt(
            best_prompt,
            self.eval_data,
            sample_size=len(self.eval_data)
        )
        
        spinner.stop()
        
        self.logger.info("Evaluation Results:")
        self.logger.info(f"  Fitness (error): {eval_fitness:.6f}")
        self.logger.info(f"  Accuracy:        {1 - eval_fitness:.4f}")
        
        self.results['metrics'] = {
            'eval_fitness': float(eval_fitness),
            'eval_accuracy': float(1 - eval_fitness)
        }
        
        return best_prompt, eval_fitness
    
    def analyze_results(self):
        """Analyze and summarize results with SSA effectiveness proof"""
        print("\n" + "="*80)
        print("ANALYZING RESULTS".center(80))
        print("="*80 + "\n")
        self.logger.section("ANALYZING RESULTS")
        
        history = self.results['history']
        
        # ============================================================
        # PROOF 1: Fitness Improvement (Primary Evidence)
        # ============================================================
        initial_fitness = history['best_fitness'][0]
        final_fitness = history['best_fitness'][-1]
        improvement = initial_fitness - final_fitness
        improvement_pct = (improvement / initial_fitness) * 100 if initial_fitness > 0 else 0
        
        self.logger.info("="*60)
        self.logger.info("PROOF 1: FITNESS IMPROVEMENT")
        self.logger.info("="*60)
        self.logger.info(f"  Initial fitness: {initial_fitness:.6f}")
        self.logger.info(f"  Final fitness:   {final_fitness:.6f}")
        self.logger.info(f"  Absolute improvement: {improvement:.6f}")
        self.logger.info(f"  Relative improvement: {improvement_pct:.2f}%")
        self.logger.info(f"  Number of improvements: {history['improvements']}")
        
        if improvement > 0:
            self.logger.info("  [OK] SSA IMPROVED fitness over iterations")
        else:
            self.logger.info("  [FAIL] No improvement detected")
        
        # ============================================================
        # PROOF 2: Monotonic Improvement (Best Never Gets Worse)
        # ============================================================
        best_fitness_series = history['best_fitness']
        is_monotonic = all(
            best_fitness_series[i] >= best_fitness_series[i+1] 
            for i in range(len(best_fitness_series)-1)
        )
        
        self.logger.info("="*60)
        self.logger.info("PROOF 2: MONOTONIC CONVERGENCE")
        self.logger.info("="*60)
        self.logger.info(f"  Best fitness never increased: {is_monotonic}")
        if is_monotonic:
            self.logger.info("  [OK] Algorithm correctly tracks best solution")
        
        # ============================================================
        # PROOF 3: Population Diversity Reduction (Convergence)
        # ============================================================
        initial_std = history['std_fitness'][0]
        final_std = history['std_fitness'][-1]
        diversity_reduction = ((initial_std - final_std) / initial_std * 100) if initial_std > 0 else 0
        
        self.logger.info("="*60)
        self.logger.info("PROOF 3: POPULATION CONVERGENCE")
        self.logger.info("="*60)
        self.logger.info(f"  Initial population std: {initial_std:.6f}")
        self.logger.info(f"  Final population std:   {final_std:.6f}")
        self.logger.info(f"  Diversity reduction:    {diversity_reduction:.2f}%")
        
        if diversity_reduction > 0:
            self.logger.info("  [OK] Population converged (squirrels found good region)")
        else:
            print("  [--] Population maintained diversity (may need more iterations)")
        print()
        
        # ============================================================
        # PROOF 4: Statistical Test - Better Than Random?
        # ============================================================
        print("="*60)
        print("PROOF 4: STATISTICAL SIGNIFICANCE")
        print("="*60)
        
        # Compare first half vs second half of optimization
        mid = len(best_fitness_series) // 2
        first_half = best_fitness_series[:mid]
        second_half = best_fitness_series[mid:]
        
        if len(first_half) > 1 and len(second_half) > 1:
            # Wilcoxon signed-rank test (non-parametric)
            try:
                # Mann-Whitney U test: is second half significantly better?
                statistic, p_value = stats.mannwhitneyu(
                    first_half, second_half, alternative='greater'
                )
                
                print(f"  Mann-Whitney U test (first half > second half):")
                print(f"    U-statistic: {statistic:.4f}")
                print(f"    p-value:     {p_value:.6f}")
                
                if p_value < 0.05:
                    print("  [OK] STATISTICALLY SIGNIFICANT improvement (p < 0.05)")
                    print("    SSA is NOT random - it systematically improves")
                else:
                    print("  [--] Not statistically significant (p >= 0.05)")
                    print("    May need more iterations or larger population")
            except Exception as e:
                print(f"  Could not perform statistical test: {e}")
        print()
        
        # ============================================================
        # PROOF 5: Convergence Rate Analysis
        # ============================================================
        print("="*60)
        print("PROOF 5: CONVERGENCE RATE")
        print("="*60)
        
        # Calculate when 50%, 75%, 90% of improvement was achieved
        total_improvement = initial_fitness - final_fitness
        
        if total_improvement > 0:
            milestones = {'50%': None, '75%': None, '90%': None}
            
            for i, fitness in enumerate(best_fitness_series):
                current_improvement = initial_fitness - fitness
                pct_complete = (current_improvement / total_improvement) * 100
                
                for milestone, iteration in milestones.items():
                    threshold = float(milestone.rstrip('%'))
                    if iteration is None and pct_complete >= threshold:
                        milestones[milestone] = i
            
            print("  Improvement milestones:")
            for milestone, iteration in milestones.items():
                if iteration is not None:
                    print(f"    {milestone} improvement at iteration {iteration}")
                else:
                    print(f"    {milestone} improvement not reached")
            
            # Early improvement indicates exploitation is working
            if milestones['50%'] is not None:
                early_pct = (milestones['50%'] / len(best_fitness_series)) * 100
                print(f"\n  50% improvement achieved in first {early_pct:.1f}% of iterations")
                if early_pct < 30:
                    print("  [OK] Fast initial convergence (exploration → exploitation)")
        print()
        
        # ============================================================
        # PROOF 6: Movement Type Effectiveness
        # ============================================================
        print("="*60)
        print("PROOF 6: EXPLORATION VS EXPLOITATION BALANCE")
        print("="*60)
        
        mean_fitness = history['mean_fitness']
        
        # Check if mean fitness also improved (not just best)
        mean_improvement = mean_fitness[0] - mean_fitness[-1]
        mean_improvement_pct = (mean_improvement / mean_fitness[0]) * 100 if mean_fitness[0] > 0 else 0
        
        print(f"  Mean population fitness improvement: {mean_improvement_pct:.2f}%")
        
        # Gap between best and mean (exploitation indicator)
        initial_gap = mean_fitness[0] - best_fitness_series[0]
        final_gap = mean_fitness[-1] - best_fitness_series[-1]
        
        print(f"  Initial best-mean gap: {initial_gap:.6f}")
        print(f"  Final best-mean gap:   {final_gap:.6f}")
        
        if final_gap < initial_gap:
            print("  [OK] Population following best solution (exploitation working)")
        print()
        
        # ============================================================
        # SUMMARY VERDICT
        # ============================================================
        print("="*60)
        print("SUMMARY: IS SSA WORKING?")
        print("="*60)
        
        evidence_count = 0
        total_evidence = 5
        
        # Evidence 1: Improvement
        if improvement > 0:
            evidence_count += 1
            print("  [OK] Fitness improved over iterations")
        else:
            print("  [FAIL] No fitness improvement")
        
        # Evidence 2: Monotonic
        if is_monotonic:
            evidence_count += 1
            print("  [OK] Best solution properly tracked")
        else:
            print("  [FAIL] Best tracking issue")
        
        # Evidence 3: Convergence
        if diversity_reduction > 10:
            evidence_count += 1
            print("  [OK] Population converged")
        else:
            print("  [--] Limited convergence")
        
        # Evidence 4: Multiple improvements
        if history['improvements'] >= 3:
            evidence_count += 1
            print(f"  [OK] Multiple improvements ({history['improvements']})")
        else:
            print(f"  [--] Few improvements ({history['improvements']})")
        
        # Evidence 5: Accuracy gain
        accuracy_gain = improvement_pct
        if accuracy_gain >= 5:
            evidence_count += 1
            print(f"  [OK] Meaningful accuracy gain ({accuracy_gain:.1f}%)")
        else:
            print(f"  [--] Small accuracy gain ({accuracy_gain:.1f}%)")
        
        print()
        print(f"  Evidence score: {evidence_count}/{total_evidence}")
        
        if evidence_count >= 4:
            print("  *** STRONG EVIDENCE: SSA is effectively optimizing prompts")
        elif evidence_count >= 2:
            print("  **- MODERATE EVIDENCE: SSA shows optimization behavior")
        else:
            print("  *-- WEAK EVIDENCE: Results inconclusive")
        
        print()
        
        # Store all analysis
        self.results['statistics'] = {
            'proof_fitness_improvement': {
                'initial': float(initial_fitness),
                'final': float(final_fitness),
                'absolute_improvement': float(improvement),
                'relative_improvement_pct': float(improvement_pct),
                'num_improvements': history['improvements']
            },
            'proof_monotonic': is_monotonic,
            'proof_convergence': {
                'initial_std': float(initial_std),
                'final_std': float(final_std),
                'diversity_reduction_pct': float(diversity_reduction)
            },
            'proof_mean_improvement': float(mean_improvement_pct),
            'evidence_score': f"{evidence_count}/{total_evidence}",
            'final_population_stats': {
                'best': float(history['best_fitness'][-1]),
                'worst': float(history['worst_fitness'][-1]),
                'mean': float(history['mean_fitness'][-1]),
                'std': float(history['std_fitness'][-1])
            }
        }
    
    def generate_visualizations(self):
        """Generate publication-quality plots"""
        print("\n" + "="*80)
        print("GENERATING VISUALIZATIONS".center(80))
        print("="*80 + "\n")
        self.logger.section("GENERATING VISUALIZATIONS")
        
        history = self.results['history']
        
        spinner = Spinner("Generating plots")
        spinner.start()
        
        plots_generated = []
        
        # 1. Basic convergence plot
        path = self.dirs['figures'] / 'convergence.png'
        self.visualizer.plot_convergence(history, save_path=str(path))
        plots_generated.append('convergence.png')
        
        # 2. Fitness distribution
        path = self.dirs['figures'] / 'fitness_distribution.png'
        self.visualizer.plot_fitness_distribution(
            history['best_fitness'], save_path=str(path)
        )
        plots_generated.append('fitness_distribution.png')
        
        # 3. SSA proof (2x2 grid)
        path = self.dirs['figures'] / 'ssa_proof.png'
        self.visualizer.plot_ssa_proof(history, save_path=str(path))
        plots_generated.append('ssa_proof.png')
        
        # 4. Exploration vs Exploitation
        path = self.dirs['figures'] / 'exploration_exploitation.png'
        self.visualizer.plot_exploration_exploitation(history, save_path=str(path))
        plots_generated.append('exploration_exploitation.png')
        
        # 5. Improvement heatmap
        path = self.dirs['figures'] / 'improvement_heatmap.png'
        self.visualizer.plot_improvement_heatmap(history, save_path=str(path))
        plots_generated.append('improvement_heatmap.png')
        
        # 6. Cumulative improvement
        path = self.dirs['figures'] / 'cumulative_improvement.png'
        self.visualizer.plot_cumulative_improvement(history, save_path=str(path))
        plots_generated.append('cumulative_improvement.png')
        
        # 7. Population statistics (2x2 grid)
        path = self.dirs['figures'] / 'population_statistics.png'
        self.visualizer.plot_population_statistics(history, save_path=str(path))
        plots_generated.append('population_statistics.png')
        
        # 8. Final summary (publication-ready)
        path = self.dirs['figures'] / 'final_summary.png'
        self.visualizer.plot_final_summary(self.results, save_path=str(path))
        plots_generated.append('final_summary.png')
        
        spinner.stop(f"Generated {len(plots_generated)} visualizations")
        
        for plot in plots_generated:
            self.logger.info(f"  [OK] {plot}")
        
        self.logger.info(f"Saved to: {self.dirs['figures']}")

    def run_post_analysis(self):
        """Run detailed post-experiment analysis"""
        print("="*80)
        print("RUNNING POST-EXPERIMENT ANALYSIS".center(80))
        print("="*80)
        print()
        
        try:
            from src.utils.analyze_results import ResultsAnalyzer
            
            spinner = Spinner("Generating detailed analysis")
            spinner.start()
            
            # Create experiment data dict
            exp_data = {
                'results': self.results,
                'config': self.config.to_dict(),
                'best_prompt': self.results['best_prompt'],
                'exp_name': self.config.experiment_name
            }
            
            spinner.stop("Analysis in progress")
            print()
            
            # Initialize analyzer
            analyser = ResultsAnalyzer()
            
            # Extract analysis data with error checking
            print("Extracting mutation impact...")
            mut_contrib = analyser.extract_mutation_impact(exp_data)
            print(f"  Got {len(mut_contrib)} mutation types")
            
            print("Calculating cost metrics...")
            cost_metrcs = analyser.calculate_cost_metrics(exp_data)
            print(f"  Total LLM calls: {cost_metrcs.get('total_llm_calls', 0)}")
            
            print("Creating prompt evolution table...")
            prompt_evol = analyser.create_prompt_evolution_table(exp_data)
            print(f"  Got {len(prompt_evol)} evolution points")
            
            print("Creating cost table...")
            cost_tbl = analyser.create_cost_table(cost_metrcs)
            print(f"  Got {len(cost_tbl)} cost metrics")
            print()
            
            anlys_dir = self.dirs['analysis']
            
            # Save CSVs
            print("Saving analysis CSVs...")
            mut_file = anlys_dir / f"{self.config.experiment_name}_mutation_impact.csv"
            mut_contrib.to_csv(mut_file, index=False)
            print(f"  Saved: {mut_file.name}")
            
            prompt_file = anlys_dir / f"{self.config.experiment_name}_prompt_evolution.csv"
            prompt_evol.to_csv(prompt_file, index=False)
            print(f"  Saved: {prompt_file.name}")
            
            cost_file = anlys_dir / f"{self.config.experiment_name}_cost_analysis.csv"
            cost_tbl.to_csv(cost_file, index=False)
            print(f"  Saved: {cost_file.name}")
            print()
            
            # Generate plots
            print("Generating visualization plots...")
            mut_plot = str(anlys_dir / f"{self.config.experiment_name}_mutations.png")
            analyser.plot_mutation_breakdown(mut_contrib, save_path=mut_plot)
            print(f"  Saved: {mut_plot.split('/')[-1]}")
            
            cost_plot = str(anlys_dir / f"{self.config.experiment_name}_costs.png")
            analyser.plot_cost_analysis(cost_metrcs, save_path=cost_plot)
            print(f"  Saved: {cost_plot.split('/')[-1]}")
            print()
            
            # Print summaries
            print("="*60)
            print("PROMPT EVOLUTION SUMMARY")
            print("="*60)
            if len(prompt_evol) > 0:
                print(prompt_evol.to_string(index=False))
            else:
                print("  [--] No prompt evolution data")
            print()
            
            print("="*60)
            print("MUTATION OPERATOR IMPACT")
            print("="*60)
            if len(mut_contrib) > 0:
                print(mut_contrib.to_string(index=False))
            else:
                print("  [--] No mutation data")
            print()
            
            print("="*60)
            print("COST ANALYSIS")
            print("="*60)
            if len(cost_tbl) > 0:
                print(cost_tbl.to_string(index=False))
            else:
                print("  [--] No cost data")
            print()
            
            # Store in results
            self.results['analysis'] = {
                'mutation_impact': mut_contrib.to_dict('records') if len(mut_contrib) > 0 else [],
                'prompt_evolution': prompt_evol.to_dict('records') if len(prompt_evol) > 0 else [],
                'cost_metrics': cost_metrcs
            }
            
            print(f"Analysis saved to: {anlys_dir}")
            print()
            
        except Exception as e:
            print(f"[ERROR] Post-analysis failed: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    def save_results(self):
        """Save all results to files"""
        print("\n" + "="*80)
        print("SAVING RESULTS".center(80))
        print("="*80 + "\n")
        self.logger.section("SAVING RESULTS")
        
        spinner = Spinner("Saving results")
        spinner.start()
        
        # Save results JSON
        results_file = self.dirs['results'] / f"{self.config.experiment_name}_results.json"
        results_to_save = self._make_json_serializable(self.results)
        
        with open(results_file, 'w') as f:
            json.dump(results_to_save, f, indent=2)
        
        # Save config
        config_file = self.dirs['results'] / f"{self.config.experiment_name}_config.json"
        self.config.save(str(config_file))
        
        # Save best prompt to text file
        prompt_file = self.dirs['results'] / f"{self.config.experiment_name}_best_prompt.txt"
        with open(prompt_file, 'w') as f:
            f.write(self.results['best_prompt'])
        
        spinner.stop()
        self.logger.info(f"Results: {results_file}")
        self.logger.info(f"Config:  {config_file}")
        self.logger.info(f"Prompt:  {prompt_file}")
        print(f"  Checkpoints: {self.chkpt_cnt} saved\n")
    
    def _make_json_serializable(self, obj):
        """Convert numpy types to Python types for JSON serialization"""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def run_full_experiment(self):
        """
        Run complete SSA optimization experiment.
        Orchestrates all steps.
        """
        print("\n")
        print("=" + "="*78 + "=")
        print("|" + "SSA PROMPT OPTIMIZATION EXPERIMENT".center(78) + "|")
        print("|" + f"Dataset size: {self.config.dataset_size} | Population: {self.config.population_size}".center(78) + "|")
        print("=" + "="*78 + "=")
        try:
            # Step 1: Load data
            self.load_dataset()
            
            # Step 2: Initialize LLM
            self.initialize_llm()
            
            # Step 3: Initialize evaluation
            self.initialize_evaluation()
            
            # Step 4: Initialize optimizer
            self.initialize_optimizer()
            
            # Step 5: Initialize visualization
            self.initialize_visualization()
            
            # Step 6: Run optimization
            best_squirrel = self.run_optimization()
            
            # Step 7: Evaluate best prompt
            self.evaluate_best_prompt(best_squirrel)
            
            # Step 8: Analyze results
            self.analyze_results()
            
            # Step 9: Generate visualizations
            self.generate_visualizations()
            
            # Step 10: Save results
            self.save_results()
            
            self.logger.section("[OK] EXPERIMENT COMPLETE")
            self.logger.info(f"Results saved to: {self.dirs['results']}")
            self.logger.info(f"Figures saved to: {self.dirs['figures']}")
            self.logger.info(f"Analysis saved to: {self.dirs['analysis']}")
            self.logger.info(f"Checkpoints saved to: {self.dirs['checkpoints']}")
            
            return self.results
        
        except Exception as e:
            self.logger.error(f"EXPERIMENT FAILED: {e}")
            raise


def validate_dataset_size(size: int) -> bool:
    """Validate dataset size is reasonable"""
    return 10 <= size <= 31000