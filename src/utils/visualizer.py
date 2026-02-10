import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


class Visualizer:
    """Generate publication-quality visualizations for SSA experiments."""
    
    def __init__(self, output_dir: str = 'outputs/figures'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure matplotlib backend for headless operation
        try:
            matplotlib.use('Agg')  # Use non-interactive backend
        except Exception as e:
            print(f"[WARNING] Could not set matplotlib backend: {e}")
        
        # Set publication style with fallback
        try:
            # Try newer seaborn style first
            plt.style.use('seaborn-v0_8-whitegrid')
        except (OSError, ValueError):
            try:
                # Fallback to older seaborn style
                plt.style.use('seaborn-whitegrid')
            except (OSError, ValueError):
                try:
                    # Final fallback to default style
                    plt.style.use('default')
                except (OSError, ValueError):
                    print("[WARNING] Could not load any matplotlib style - using defaults")
        
        # Configure matplotlib parameters
        try:
            plt.rcParams.update({
                'font.size': 12,
                'axes.labelsize': 14,
                'axes.titlesize': 16,
                'xtick.labelsize': 12,
                'ytick.labelsize': 12,
                'legend.fontsize': 11,
                'figure.titlesize': 18,
                'font.family': 'serif',
                'axes.grid': True,
                'grid.alpha': 0.3,
                'savefig.dpi': 300,
                'savefig.bbox': 'tight',
                'figure.autolayout': True
            })
        except Exception as e:
            print(f"[WARNING] Could not configure matplotlib parameters: {e}")
    
    def _save_figure(self, fig, save_path: str) -> bool:
        """
        Safely save figure to disk with error handling.
        
        Args:
            fig: matplotlib figure object
            save_path: path where to save the figure
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        try:
            # Ensure parent directory exists
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save figure
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            # Verify file was created
            if Path(save_path).exists():
                return True
            else:
                print(f"[ERROR] Figure saved but file not found: {save_path}")
                return False
                
        except PermissionError as e:
            print(f"[ERROR] Permission denied saving figure: {save_path} - {e}")
            plt.close(fig)
            return False
        except IOError as e:
            print(f"[ERROR] IO error saving figure: {save_path} - {e}")
            plt.close(fig)
            return False
        except Exception as e:
            print(f"[ERROR] Unexpected error saving figure: {save_path} - {type(e).__name__}: {e}")
            plt.close(fig)
            return False
    
    def plot_convergence(self, history: Dict, save_path: Optional[str] = None):
        """Plot fitness convergence over iterations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        iterations = range(len(history['best_fitness']))
        
        ax.plot(iterations, history['best_fitness'], 'b-', linewidth=2, label='Best Fitness')
        ax.plot(iterations, history['mean_fitness'], 'g--', linewidth=1.5, label='Mean Fitness')
        ax.plot(iterations, history['worst_fitness'], 'r:', linewidth=1, label='Worst Fitness')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fitness (Lower is Better)')
        ax.set_title('SSA Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_fitness_distribution(self, fitness_values: List[float], 
                                   save_path: Optional[str] = None):
        """Plot distribution of fitness values"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(fitness_values, bins=20, edgecolor='black', alpha=0.7, color='steelblue')
        ax.axvline(np.mean(fitness_values), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean: {np.mean(fitness_values):.4f}')
        ax.axvline(np.min(fitness_values), color='green', linestyle='-', 
                   linewidth=2, label=f'Best: {np.min(fitness_values):.4f}')
        
        ax.set_xlabel('Fitness')
        ax.set_ylabel('Frequency')
        ax.set_title('Fitness Distribution Over Optimization')
        ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_ssa_proof(self, history: Dict, save_path: Optional[str] = None):
        """Generate comprehensive proof visualization (2x2 grid)"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(history['best_fitness']))
        
        # Plot 1: Convergence with confidence band
        ax1 = axes[0, 0]
        ax1.plot(iterations, history['best_fitness'], 'b-', linewidth=2, label='Best')
        ax1.plot(iterations, history['mean_fitness'], 'g--', linewidth=1.5, label='Mean')
        ax1.fill_between(
            iterations,
            [m - s for m, s in zip(history['mean_fitness'], history['std_fitness'])],
            [m + s for m, s in zip(history['mean_fitness'], history['std_fitness'])],
            alpha=0.3, color='green', label='±1 std'
        )
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness (lower = better)')
        ax1.set_title('(a) Fitness Convergence with Population Spread')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Per-iteration improvement
        ax2 = axes[0, 1]
        improvements = [0] + [
            max(0, history['best_fitness'][i] - history['best_fitness'][i+1])
            for i in range(len(history['best_fitness'])-1)
        ]
        colors = ['green' if imp > 0 else 'lightgray' for imp in improvements]
        ax2.bar(iterations, improvements, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness Improvement')
        ax2.set_title('(b) Per-Iteration Improvements')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Population diversity over time
        ax3 = axes[1, 0]
        ax3.plot(iterations, history['std_fitness'], 'r-', linewidth=2)
        ax3.fill_between(iterations, 0, history['std_fitness'], alpha=0.3, color='red')
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Population Std Dev')
        ax3.set_title('(c) Population Diversity Reduction')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Best vs Worst gap (exploitation indicator)
        ax4 = axes[1, 1]
        gap = [w - b for w, b in zip(history['worst_fitness'], history['best_fitness'])]
        ax4.plot(iterations, gap, 'm-', linewidth=2)
        ax4.fill_between(iterations, 0, gap, alpha=0.3, color='purple')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best-Worst Gap')
        ax4.set_title('(d) Population Spread (Convergence Indicator)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_exploration_exploitation(self, history: Dict, save_path: Optional[str] = None):
        """Visualize exploration vs exploitation phases"""
        fig, ax = plt.subplots(figsize=(12, 6))
        
        iterations = np.array(range(len(history['best_fitness'])))
        best = np.array(history['best_fitness'])
        mean = np.array(history['mean_fitness'])
        std = np.array(history['std_fitness'])
        
        # Normalize for comparison
        best_norm = (best - best.min()) / (best.max() - best.min() + 1e-8)
        std_norm = (std - std.min()) / (std.max() - std.min() + 1e-8)
        
        ax.plot(iterations, best_norm, 'b-', linewidth=2, label='Best Fitness (normalized)')
        ax.plot(iterations, std_norm, 'r--', linewidth=2, label='Diversity (normalized)')
        
        # Identify phases
        mid = len(iterations) // 2
        ax.axvspan(0, mid, alpha=0.1, color='green', label='Exploration Phase')
        ax.axvspan(mid, len(iterations), alpha=0.1, color='blue', label='Exploitation Phase')
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Normalized Value')
        ax.set_title('Exploration vs Exploitation Balance')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_improvement_heatmap(self, history: Dict, window_size: int = 5,
                                  save_path: Optional[str] = None):
        """Heatmap showing improvement intensity over time"""
        fig, ax = plt.subplots(figsize=(12, 4))
        
        best = np.array(history['best_fitness'])
        improvements = np.array([0] + [
            max(0, best[i] - best[i+1]) for i in range(len(best)-1)
        ])
        
        # Create windowed improvement matrix
        n_windows = len(improvements) // window_size
        if n_windows == 0:
            n_windows = 1
            window_size = len(improvements)
        
        windowed = improvements[:n_windows * window_size].reshape(1, -1)
        
        sns.heatmap(windowed, ax=ax, cmap='Greens', cbar_kws={'label': 'Improvement'},
                    xticklabels=range(len(improvements)), yticklabels=False)
        ax.set_xlabel('Iteration')
        ax.set_title('Improvement Intensity Over Time')
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_cumulative_improvement(self, history: Dict, save_path: Optional[str] = None):
        """Plot cumulative improvement over iterations"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        best = np.array(history['best_fitness'])
        initial = best[0]
        
        # Cumulative improvement (as percentage)
        cumulative_improvement = [(initial - b) / initial * 100 for b in best]
        
        iterations = range(len(best))
        
        ax.fill_between(iterations, 0, cumulative_improvement, alpha=0.3, color='green')
        ax.plot(iterations, cumulative_improvement, 'g-', linewidth=2)
        
        # Mark key milestones
        final_improvement = cumulative_improvement[-1]
        for pct in [25, 50, 75]:
            target = final_improvement * pct / 100
            for i, ci in enumerate(cumulative_improvement):
                if ci >= target:
                    ax.axhline(y=target, color='gray', linestyle=':', alpha=0.5)
                    ax.annotate(f'{pct}% @ iter {i}', xy=(i, target),
                               xytext=(i+2, target+1), fontsize=10)
                    break
        
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Cumulative Improvement (%)')
        ax.set_title('Cumulative Fitness Improvement')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_population_statistics(self, history: Dict, save_path: Optional[str] = None):
        """Comprehensive population statistics over time"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        iterations = range(len(history['best_fitness']))
        
        # Plot 1: All fitness tracks
        ax1 = axes[0, 0]
        ax1.plot(iterations, history['best_fitness'], 'g-', linewidth=2, label='Best')
        ax1.plot(iterations, history['mean_fitness'], 'b-', linewidth=1.5, label='Mean')
        ax1.plot(iterations, history['worst_fitness'], 'r-', linewidth=1, label='Worst')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('(a) Population Fitness Trajectories')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Fitness range (box-plot style)
        ax2 = axes[0, 1]
        best = np.array(history['best_fitness'])
        worst = np.array(history['worst_fitness'])
        mean = np.array(history['mean_fitness'])
        
        ax2.fill_between(iterations, best, worst, alpha=0.3, color='blue', label='Range')
        ax2.plot(iterations, mean, 'b-', linewidth=2, label='Mean')
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Fitness')
        ax2.set_title('(b) Population Fitness Range')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Coefficient of Variation (CV)
        ax3 = axes[1, 0]
        cv = [s / m if m > 0 else 0 for s, m in zip(history['std_fitness'], history['mean_fitness'])]
        ax3.plot(iterations, cv, 'purple', linewidth=2)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Coefficient of Variation')
        ax3.set_title('(c) Population Homogeneity (CV)')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Rate of improvement
        ax4 = axes[1, 1]
        improvements = [0] + [
            (history['best_fitness'][i] - history['best_fitness'][i+1])
            for i in range(len(history['best_fitness'])-1)
        ]
        
        # Moving average
        window = 3
        smoothed = np.convolve(improvements, np.ones(window)/window, mode='valid')
        smoothed_x = range(window-1, len(improvements))
        
        ax4.bar(iterations, improvements, alpha=0.3, color='green', label='Per-iteration')
        ax4.plot(smoothed_x, smoothed, 'g-', linewidth=2, label=f'{window}-iter moving avg')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Improvement')
        ax4.set_title('(d) Rate of Improvement')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig
    
    def plot_final_summary(self, results: Dict, save_path: Optional[str] = None):
        """Create a publication-ready summary figure"""
        fig = plt.figure(figsize=(16, 12))
        
        history = results['history']
        stats = results.get('statistics', {})
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Plot 1: Main convergence (large)
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = range(len(history['best_fitness']))
        ax1.plot(iterations, history['best_fitness'], 'b-', linewidth=2.5, label='Best Fitness')
        ax1.fill_between(
            iterations,
            [m - s for m, s in zip(history['mean_fitness'], history['std_fitness'])],
            [m + s for m, s in zip(history['mean_fitness'], history['std_fitness'])],
            alpha=0.2, color='blue'
        )
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Fitness')
        ax1.set_title('SSA Optimization Convergence', fontsize=16, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Key metrics box
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        proof = stats.get('proof_fitness_improvement', {})
        metrics_text = (
            f"OPTIMIZATION SUMMARY\n"
            f"{'─' * 30}\n\n"
            f"Initial Fitness: {proof.get('initial', 'N/A'):.4f}\n"
            f"Final Fitness: {proof.get('final', 'N/A'):.4f}\n"
            f"Improvement: {proof.get('relative_improvement_pct', 0):.2f}%\n\n"
            f"Total Improvements: {proof.get('num_improvements', 'N/A')}\n"
            f"Evidence Score: {stats.get('evidence_score', 'N/A')}\n"
        )
        ax2.text(0.1, 0.9, metrics_text, transform=ax2.transAxes,
                fontsize=12, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
        
        # Plot 3: Improvement rate
        ax3 = fig.add_subplot(gs[1, 0])
        improvements = [0] + [
            max(0, history['best_fitness'][i] - history['best_fitness'][i+1])
            for i in range(len(history['best_fitness'])-1)
        ]
        colors = ['forestgreen' if imp > 0 else 'lightgray' for imp in improvements]
        ax3.bar(iterations, improvements, color=colors, edgecolor='black', linewidth=0.3)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Improvement')
        ax3.set_title('Per-Iteration Improvements')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Population diversity
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(iterations, history['std_fitness'], 'r-', linewidth=2)
        ax4.fill_between(iterations, 0, history['std_fitness'], alpha=0.3, color='red')
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Std Dev')
        ax4.set_title('Population Diversity')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Cumulative improvement
        ax5 = fig.add_subplot(gs[1, 2])
        best = np.array(history['best_fitness'])
        initial = best[0]
        cumulative = [(initial - b) / initial * 100 if initial > 0 else 0 for b in best]
        ax5.fill_between(iterations, 0, cumulative, alpha=0.3, color='green')
        ax5.plot(iterations, cumulative, 'g-', linewidth=2)
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('Cumulative Improvement (%)')
        ax5.set_title('Total Progress')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Final distribution
        ax6 = fig.add_subplot(gs[2, 0])
        ax6.hist(history['best_fitness'], bins=15, edgecolor='black', 
                alpha=0.7, color='steelblue')
        ax6.axvline(np.min(history['best_fitness']), color='green', 
                   linestyle='--', linewidth=2, label='Best')
        ax6.set_xlabel('Fitness')
        ax6.set_ylabel('Frequency')
        ax6.set_title('Fitness Distribution')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # Plot 7: Best-Worst gap
        ax7 = fig.add_subplot(gs[2, 1])
        gap = [w - b for w, b in zip(history['worst_fitness'], history['best_fitness'])]
        ax7.plot(iterations, gap, 'm-', linewidth=2)
        ax7.fill_between(iterations, 0, gap, alpha=0.3, color='purple')
        ax7.set_xlabel('Iteration')
        ax7.set_ylabel('Gap')
        ax7.set_title('Population Convergence Gap')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Algorithm verdict
        ax8 = fig.add_subplot(gs[2, 2])
        ax8.axis('off')
        
        evidence = stats.get('evidence_score', '0/5')
        score = int(evidence.split('/')[0]) if '/' in str(evidence) else 0
        
        if score >= 4:
            verdict = "✓ STRONG EVIDENCE\nSSA Effectively Optimizing"
            color = 'green'
        elif score >= 2:
            verdict = "○ MODERATE EVIDENCE\nSSA Shows Progress"
            color = 'orange'
        else:
            verdict = "✗ WEAK EVIDENCE\nResults Inconclusive"
            color = 'red'
        
        ax8.text(0.5, 0.5, verdict, transform=ax8.transAxes,
                fontsize=14, ha='center', va='center', fontweight='bold',
                color=color,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                         edgecolor=color, linewidth=2))
        
        plt.suptitle('SSA Prompt Optimization - Experimental Results', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        if save_path:
            self._save_figure(fig, save_path)
        else:
            plt.show()
        
        return fig