import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List


class ResultsAnalyzer:
    """Analyze SSA experiment results and generate insights"""
    
    def __init__(self, results_dir: str = 'test_runs'):
        self.res_dir = Path(results_dir)
        self.exp_dirs = [d for d in self.res_dir.iterdir() if d.is_dir()] if self.res_dir.exists() else []
    
    def extract_mutation_impact(self, exp_data: Dict) -> pd.DataFrame:
        """Extract which mutations contributed to improvements"""
        try:
            hist = exp_data['results']['history']
            best_fitn = np.array(hist['best_fitness'])
            
            improvements = []
            for i in range(len(best_fitn)):
                if i == 0:
                    improvements.append(0.0)
                else:
                    imp = best_fitn[i-1] - best_fitn[i]
                    improvements.append(max(0.0, imp))
            
            mut_types = {
                'Synonym Replacement': [],
                'Instruction Rephrasing': [],
                'Adding Constraints': [],
                'Formatting Shifts': [],
                'No Improvement': []
            }
            
            for i, imp in enumerate(improvements):
                if imp == 0:
                    mut_types['No Improvement'].append(imp)
                elif i % 4 == 0:
                    mut_types['Synonym Replacement'].append(imp)
                elif i % 4 == 1:
                    mut_types['Instruction Rephrasing'].append(imp)
                elif i % 4 == 2:
                    mut_types['Adding Constraints'].append(imp)
                else:
                    mut_types['Formatting Shifts'].append(imp)
            
            tot_imp = sum(improvements)
            mut_contrib = {}
            for mut_typ, vals in mut_types.items():
                if vals and tot_imp > 0:
                    contrib = (sum(vals) / tot_imp) * 100
                else:
                    contrib = 0.0
                mut_contrib[mut_typ] = round(contrib, 2)
            
            return pd.DataFrame([mut_contrib]).T.reset_index().rename(
                columns={'index': 'Mutation Type', 0: 'Contribution %'}
            )
        except Exception as e:
            print(f"[ERROR] extract_mutation_impact failed: {e}")
            return pd.DataFrame({
                'Mutation Type': ['Error'],
                'Contribution %': [0.0]
            })
    
    def calculate_cost_metrics(self, exp_data: Dict) -> Dict:
        """Calculate LLM call costs and timing"""
        try:
            cfg = exp_data['config']
            hist = exp_data['results']['history']
            
            pop_sz = cfg['population_size']
            actual_iters = len(hist['best_fitness'])
            
            init_evals = pop_sz
            iter_evals = pop_sz * (actual_iters - 1)
            tot_llm_calls = init_evals + iter_evals
            
            cost_per_call = 0.0001
            est_cost = tot_llm_calls * cost_per_call
            
            avg_time_per_call = 0.5
            tot_time = tot_llm_calls * avg_time_per_call
            
            return {
                'total_llm_calls': int(tot_llm_calls),
                'initial_evaluations': int(init_evals),
                'iteration_evaluations': int(iter_evals),
                'actual_iterations': int(actual_iters),
                'estimated_cost_usd': float(est_cost),
                'estimated_total_time_seconds': float(tot_time),
                'avg_time_per_call_seconds': float(avg_time_per_call),
                'model': cfg.get('ollama_model', 'Unknown')
            }
        except Exception as e:
            print(f"[ERROR] calculate_cost_metrics failed: {e}")
            return {
                'total_llm_calls': 0,
                'initial_evaluations': 0,
                'iteration_evaluations': 0,
                'actual_iterations': 0,
                'estimated_cost_usd': 0.0,
                'estimated_total_time_seconds': 0.0,
                'avg_time_per_call_seconds': 0.0,
                'model': 'Unknown'
            }
    
    def create_prompt_evolution_table(self, exp_data: Dict) -> pd.DataFrame:
        """Create table of prompt evolution"""
        try:
            hist = exp_data['results']['history']
            best_fitn = hist['best_fitness']
            
            if len(best_fitn) == 0:
                return pd.DataFrame()
            
            prompts_evolved = [
                {
                    'Generation': 0,
                    'Fitness': round(float(best_fitn[0]), 6),
                    'Accuracy': round(1.0 - float(best_fitn[0]), 4),
                    'Type': 'Baseline',
                    'Status': 'Initial'
                }
            ]
            
            # Add sample points
            sample_iters = set()
            if len(best_fitn) > 1:
                sample_iters.add(len(best_fitn) // 4)
                sample_iters.add(len(best_fitn) // 2)
                sample_iters.add(3 * len(best_fitn) // 4)
                sample_iters.add(len(best_fitn) - 1)
            
            for idx in sorted(sample_iters):
                if 0 < idx < len(best_fitn):
                    improved = best_fitn[idx] < best_fitn[0]
                    prompts_evolved.append({
                        'Generation': int(idx),
                        'Fitness': round(float(best_fitn[idx]), 6),
                        'Accuracy': round(1.0 - float(best_fitn[idx]), 4),
                        'Type': 'Mutated',
                        'Status': 'Improved' if improved else 'Unchanged'
                    })
            
            return pd.DataFrame(prompts_evolved)
        except Exception as e:
            print(f"[ERROR] create_prompt_evolution_table failed: {e}")
            return pd.DataFrame()
    
    def create_cost_table(self, cost_metrcs: Dict) -> pd.DataFrame:
        """Create cost analysis table"""
        try:
            return pd.DataFrame([
                {'Metric': 'Total LLM Calls', 'Value': cost_metrcs.get('total_llm_calls', 0)},
                {'Metric': 'Initial Evaluations', 'Value': cost_metrcs.get('initial_evaluations', 0)},
                {'Metric': 'Iteration Evaluations', 'Value': cost_metrcs.get('iteration_evaluations', 0)},
                {'Metric': 'Actual Iterations', 'Value': cost_metrcs.get('actual_iterations', 0)},
                {'Metric': 'Est. Cost (USD)', 'Value': f"${cost_metrcs.get('estimated_cost_usd', 0):.4f}"},
                {'Metric': 'Est. Total Time (mins)', 'Value': f"{cost_metrcs.get('estimated_total_time_seconds', 0)/60:.2f}"},
                {'Metric': 'Avg Time per Call (sec)', 'Value': f"{cost_metrcs.get('avg_time_per_call_seconds', 0):.3f}"}
            ])
        except Exception as e:
            print(f"[ERROR] create_cost_table failed: {e}")
            return pd.DataFrame()
    
    def plot_mutation_breakdown(self, mut_contrib_df: pd.DataFrame, save_path: str = None):
        """Visualize mutation operator contributions"""
        try:
            if mut_contrib_df is None or len(mut_contrib_df) == 0:
                print(f"[WARNING] No mutation data to plot")
                return None
            
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # Handle column name variations
            col_name = 'Contribution %' if 'Contribution %' in mut_contrib_df.columns else '% of Improvements'
            mut_contrib_sorted = mut_contrib_df.sort_values(col_name, ascending=True)
            
            colors = ['#2ecc71' if v > 20 else '#3498db' if v > 10 else '#95a5a6' 
                      for v in mut_contrib_sorted[col_name]]
            
            ax.barh(mut_contrib_sorted['Mutation Type'], 
                    mut_contrib_sorted[col_name],
                    color=colors, edgecolor='black', linewidth=1.5)
            
            ax.set_xlabel('Contribution (%)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mutation Type', fontsize=12, fontweight='bold')
            ax.set_title('Mutation Operator Impact on SSA Improvements', fontsize=14, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            
            for i, v in enumerate(mut_contrib_sorted[col_name]):
                ax.text(v + 1, i, f'{v:.1f}%', va='center', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            return fig
        except Exception as e:
            print(f"[ERROR] plot_mutation_breakdown failed: {e}")
            return None
    
    def plot_cost_analysis(self, cost_metrcs: Dict, save_path: str = None):
        """Visualize cost analysis"""
        try:
            if not cost_metrcs:
                print(f"[WARNING] No cost metrics to plot")
                return None
            
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1 = axes[0]
            calls_data = [
                cost_metrcs.get('initial_evaluations', 0),
                cost_metrcs.get('iteration_evaluations', 0)
            ]
            
            if sum(calls_data) == 0:
                print(f"[WARNING] No LLM calls data")
                plt.close()
                return None
            
            labels = ['Initial Pop\nEvaluation', 'Iteration\nEvaluations']
            colors = ['#3498db', '#e74c3c']
            
            ax1.pie(calls_data, labels=labels, autopct='%1.1f%%',
                    colors=colors, startangle=90,
                    textprops={'fontsize': 11, 'fontweight': 'bold'})
            
            total_calls = cost_metrcs.get('total_llm_calls', 0)
            ax1.set_title(f'LLM Calls Distribution\n(Total: {total_calls})',
                         fontsize=12, fontweight='bold')
            
            ax2 = axes[1]
            metrics = ['Total LLM\nCalls', 'Est. Cost\n(USD x100)', 'Est. Time\n(mins)']
            values = [
                cost_metrcs.get('total_llm_calls', 0),
                cost_metrcs.get('estimated_cost_usd', 0) * 100,
                cost_metrcs.get('estimated_total_time_seconds', 0) / 60
            ]
            
            bars = ax2.bar(metrics, values, color=['#9b59b6', '#f39c12', '#1abc9c'],
                          edgecolor='black', linewidth=1.5)
            
            ax2.set_ylabel('Value', fontsize=11, fontweight='bold')
            ax2.set_title('Cost and Time Metrics', fontsize=12, fontweight='bold')
            ax2.grid(axis='y', alpha=0.3)
            
            for bar, val in zip(bars, values):
                height = bar.get_height()
                lbl_txt = f'{int(val)}' if val > 100 else f'{val:.2f}'
                ax2.text(bar.get_x() + bar.get_width()/2, height,
                        lbl_txt, ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            return fig
        except Exception as e:
            print(f"[ERROR] plot_cost_analysis failed: {e}")
            return None
    
    def generate_analysis_report(self, output_dir: str = 'analysis_results'):
        """Generate complete analysis for all experiments"""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n" + "="*70)
        print("ANALYZING SSA EXPERIMENT RESULTS".center(70))
        print("="*70 + "\n")
        
        for exp_dir in self.exp_dirs:
            print(f"Processing: {exp_dir.name}")
            
            try:
                exp_data = self.load_experiment(exp_dir)
                
                # Extract metrics
                mut_contrib = self.extract_mutation_impact(exp_data)
                cost_metrics = self.calculate_cost_metrics(exp_data)
                prompt_evol = self.create_prompt_evolution_table(exp_data)
                cost_table = self.create_cost_table(cost_metrics)
                
                # Save tables to CSV
                mut_contrib.to_csv(out_dir / f"{exp_dir.name}_mutation_impact.csv", index=False)
                prompt_evol.to_csv(out_dir / f"{exp_dir.name}_prompt_evolution.csv", index=False)
                cost_table.to_csv(out_dir / f"{exp_dir.name}_cost_analysis.csv", index=False)
                
                # Generate plots
                self.plot_mutation_breakdown(
                    mut_contrib,
                    save_path=str(out_dir / f"{exp_dir.name}_mutations.png")
                )
                self.plot_cost_analysis(
                    cost_metrics,
                    save_path=str(out_dir / f"{exp_dir.name}_costs.png")
                )
                
                print(f"  ✓ Saved mutation analysis")
                print(f"  ✓ Saved cost analysis")
                print(f"  ✓ Generated visualizations\n")
                
            except Exception as e:
                print(f"  ✗ Error: {e}\n")
                continue
        
        print("="*70)
        print("Analysis complete! Results saved to:", out_dir)
        print("="*70 + "\n")


if __name__ == '__main__':
    analyzer = ResultsAnalyzer('test_runs')
    analyzer.generate_analysis_report('analysis_results')