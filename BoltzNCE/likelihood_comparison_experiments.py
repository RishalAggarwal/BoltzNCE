#!/usr/bin/env python3
"""
Likelihood Comparison Experiments

This script performs two key experiments:
1. Compare EBM likelihood (from potential forward pass) vs true likelihood
2. Compare flow matching likelihood vs Hutchinson's approximator (reverse integral)

The script is designed to be flexible with configurable file paths for:
- Potential model weights
- Samples data  
- True likelihoods data
"""

import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn import metrics
import wandb
import yaml
from pathlib import Path

from models.interpolant import Interpolant
from models.ebm import GVP_EBM
from utils.utils import load_models
from dataset.ad2_dataset import get_alanine_atom_types, get_alanine_implicit_dataset, get_alanine_features


def parse_arguments():
    """Parse command line arguments for likelihood comparison experiments."""
    parser = argparse.ArgumentParser(description="Likelihood Comparison Experiments")
    
    # Model and data paths (to be provided later)
    parser.add_argument('--potential_model_path', type=str, default=None,
                       help='Path to potential model weights (.pt file)')
    parser.add_argument('--vector_model_path', type=str, default=None,
                       help='Path to vector field model weights (.pt file)')
    parser.add_argument('--config_path', type=str, default=None,
                       help='Path to model config (.yaml file)')
    parser.add_argument('--samples_path', type=str, default=None,
                       help='Path to samples data (.npy file)')
    parser.add_argument('--true_ebm_likelihood_path', type=str, default=None,
                       help='Path to true EBM likelihoods (.npy file)')
    parser.add_argument('--true_flow_likelihood_path', type=str, default=None,
                       help='Path to true flow matching likelihoods (.npy file)')
    
    # Experiment settings
    parser.add_argument('--experiment', type=str, choices=['ebm', 'flow', 'both'], default='both',
                       help='Which experiment to run')
    parser.add_argument('--batch_size', type=int, default=1000,
                       help='Batch size for processing samples')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to run on (cuda/cpu)')
    
    # ODE solver settings for Hutchinson approximator
    parser.add_argument('--rtol', type=float, default=1e-5,
                       help='Relative tolerance for ODE solver')
    parser.add_argument('--atol', type=float, default=1e-5,
                       help='Absolute tolerance for ODE solver')
    parser.add_argument('--tmin', type=float, default=1e-3,
                       help='Minimum time for integration')
    
    # Output settings
    parser.add_argument('--output_dir', type=str, default='./likelihood_comparison_results/',
                       help='Directory to save results')
    parser.add_argument('--save_plots', action='store_true', default=True,
                       help='Save comparison plots')
    parser.add_argument('--wandb_project', type=str, default='likelihood_comparisons',
                       help='Wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default=None,
                       help='Wandb run name')
    
    return parser.parse_args()


def load_model_and_config(args):
    """Load models and configuration."""
    if args.config_path is None:
        raise ValueError("Config path must be provided")
    
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set up model arguments from config
    model_args = {
        'config': args.config_path,
        'rtol': args.rtol,
        'atol': args.atol, 
        'tmin': args.tmin
    }
    
    # Get dataset info for model initialization
    h_initial = get_alanine_implicit_dataset(n_samples=1, nframes_training=1)
    
    # Load models using existing utility
    potential_model, vector_field, interpolant_obj = load_models(model_args, h_initial)
    
    if potential_model is not None:
        potential_model.eval()
    if vector_field is not None:
        vector_field.eval()
        
    return potential_model, vector_field, interpolant_obj, config


def get_ebm_likelihood_from_potential(interpolant_obj, samples, batch_size=1000):
    """
    Compute EBM likelihood using potential model forward pass.
    Based on get_potential_logp() from infer_ad2.py:131
    """
    print("Computing EBM likelihoods from potential forward pass...")
    
    dlogf_all = []
    samples_torch = torch.from_numpy(samples).float().to('cuda')
    samples_torch = samples_torch / interpolant_obj.scaling
    
    # Process in batches
    for i in range(0, len(samples_torch), batch_size):
        end_idx = min(i + batch_size, len(samples_torch))
        samples_batch = samples_torch[i:end_idx]
        
        with torch.no_grad():
            dlogf = interpolant_obj.log_prob_forward(samples_batch)
            dlogf_all.append(dlogf.cpu().detach())
    
    # Concatenate and normalize
    dlogf_all = torch.cat(dlogf_all, dim=0)
    # Note: Removing normalization that was in original code for fair comparison
    # dlogf_all = dlogf_all - torch.logsumexp(dlogf_all, dim=(0,1))
    
    return dlogf_all.numpy()


def get_hutchinson_likelihood_reverse(interpolant_obj, samples, batch_size=500):
    """
    Compute likelihood using Hutchinson's estimator via reverse integration.
    Based on NLL_integral() from interpolant.py:276
    """
    print("Computing likelihoods using Hutchinson's approximator (reverse integral)...")
    
    nll_all = []
    samples_torch = torch.from_numpy(samples).float().to('cuda')
    
    # Process in batches 
    for i in range(0, len(samples_torch), batch_size):
        end_idx = min(i + batch_size, len(samples_torch))
        samples_batch = samples_torch[i:end_idx]
        
        print(f'Processing batch: {i} to {end_idx}')
        
        with torch.no_grad():
            # This performs reverse integration with Hutchinson's estimator
            nll = interpolant_obj.NLL_integral(samples_batch)
            nll_all.append(nll.cpu().detach())
    
    # Concatenate results
    nll_all = torch.cat(nll_all, dim=0)
    
    return nll_all.numpy()


def compute_comparison_metrics(predicted_ll, true_ll, experiment_name):
    """Compute various comparison metrics between predicted and true likelihoods."""
    
    # Flatten arrays
    pred_flat = predicted_ll.flatten()
    true_flat = true_ll.flatten()
    
    # Basic statistics
    mse = np.mean((pred_flat - true_flat) ** 2)
    mae = np.mean(np.abs(pred_flat - true_flat))
    
    # Correlation
    pearson_r, pearson_p = stats.pearsonr(pred_flat, true_flat)
    spearman_r, spearman_p = stats.spearmanr(pred_flat, true_flat)
    
    # R-squared
    r2 = metrics.r2_score(true_flat, pred_flat)
    
    metrics_dict = {
        f'{experiment_name}_mse': mse,
        f'{experiment_name}_mae': mae,
        f'{experiment_name}_pearson_r': pearson_r,
        f'{experiment_name}_pearson_p': pearson_p,
        f'{experiment_name}_spearman_r': spearman_r,
        f'{experiment_name}_spearman_p': spearman_p,
        f'{experiment_name}_r2': r2,
        f'{experiment_name}_pred_mean': np.mean(pred_flat),
        f'{experiment_name}_pred_std': np.std(pred_flat),
        f'{experiment_name}_true_mean': np.mean(true_flat),
        f'{experiment_name}_true_std': np.std(true_flat)
    }
    
    return metrics_dict


def create_comparison_plots(predicted_ll, true_ll, experiment_name, output_dir):
    """Create comparison plots between predicted and true likelihoods."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Flatten arrays
    pred_flat = predicted_ll.flatten()
    true_flat = true_ll.flatten()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{experiment_name} Likelihood Comparison', fontsize=16)
    
    # Scatter plot
    axes[0, 0].scatter(true_flat, pred_flat, alpha=0.6, s=1)
    axes[0, 0].plot([true_flat.min(), true_flat.max()], 
                    [true_flat.min(), true_flat.max()], 'r--', alpha=0.8)
    axes[0, 0].set_xlabel('True Likelihood')
    axes[0, 0].set_ylabel('Predicted Likelihood')
    axes[0, 0].set_title('Predicted vs True')
    
    # Residuals plot
    residuals = pred_flat - true_flat
    axes[0, 1].scatter(true_flat, residuals, alpha=0.6, s=1)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', alpha=0.8)
    axes[0, 1].set_xlabel('True Likelihood')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs True')
    
    # Distribution comparison
    axes[1, 0].hist(true_flat, bins=50, alpha=0.7, label='True', density=True)
    axes[1, 0].hist(pred_flat, bins=50, alpha=0.7, label='Predicted', density=True)
    axes[1, 0].set_xlabel('Likelihood')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Distribution Comparison')
    axes[1, 0].legend()
    
    # Q-Q plot
    stats.probplot(residuals, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot of Residuals')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'{experiment_name.lower()}_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to: {plot_path}")


def run_ebm_experiment(args, interpolant_obj, samples, true_ebm_ll):
    """Run EBM likelihood comparison experiment."""
    
    print("\n" + "="*50)
    print("EXPERIMENT 1: EBM vs True Likelihood")  
    print("="*50)
    
    # Get EBM likelihoods from potential forward pass
    predicted_ebm_ll = get_ebm_likelihood_from_potential(
        interpolant_obj, samples, args.batch_size
    )
    
    # Compute metrics
    metrics = compute_comparison_metrics(predicted_ebm_ll, true_ebm_ll, 'ebm')
    
    # Log metrics
    if wandb.run is not None:
        wandb.log(metrics)
    
    # Print key metrics
    print(f"EBM Comparison Metrics:")
    print(f"  MSE: {metrics['ebm_mse']:.6f}")
    print(f"  MAE: {metrics['ebm_mae']:.6f}")
    print(f"  Pearson R: {metrics['ebm_pearson_r']:.4f} (p={metrics['ebm_pearson_p']:.2e})")
    print(f"  R²: {metrics['ebm_r2']:.4f}")
    
    # Create plots
    if args.save_plots:
        create_comparison_plots(predicted_ebm_ll, true_ebm_ll, 'EBM', args.output_dir)
    
    return predicted_ebm_ll, metrics


def run_flow_experiment(args, interpolant_obj, samples, true_flow_ll):
    """Run flow matching likelihood comparison experiment."""
    
    print("\n" + "="*50)
    print("EXPERIMENT 2: Flow Matching vs Hutchinson's Approximator")
    print("="*50)
    
    # Get likelihoods using Hutchinson's approximator (reverse integral)
    predicted_flow_ll = get_hutchinson_likelihood_reverse(
        interpolant_obj, samples, args.batch_size
    )
    
    # Compute metrics
    metrics = compute_comparison_metrics(predicted_flow_ll, true_flow_ll, 'flow')
    
    # Log metrics
    if wandb.run is not None:
        wandb.log(metrics)
    
    # Print key metrics
    print(f"Flow Matching Comparison Metrics:")
    print(f"  MSE: {metrics['flow_mse']:.6f}")
    print(f"  MAE: {metrics['flow_mae']:.6f}")
    print(f"  Pearson R: {metrics['flow_pearson_r']:.4f} (p={metrics['flow_pearson_p']:.2e})")
    print(f"  R²: {metrics['flow_r2']:.4f}")
    
    # Create plots
    if args.save_plots:
        create_comparison_plots(predicted_flow_ll, true_flow_ll, 'Flow_Matching', args.output_dir)
    
    return predicted_flow_ll, metrics


def main():
    """Main execution function."""
    
    args = parse_arguments()
    
    # Initialize wandb if requested
    if args.wandb_project:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    print("Likelihood Comparison Experiments")
    print("="*50)
    print(f"Device: {args.device}")
    print(f"Experiment: {args.experiment}")
    
    # Load models and config
    try:
        potential_model, vector_field, interpolant_obj, config = load_model_and_config(args)
        print("Models loaded successfully")
    except Exception as e:
        print(f"Error loading models: {e}")
        return 1
    
    # Load samples data
    if args.samples_path is None:
        print("Error: samples_path must be provided")
        return 1
    
    try:
        samples = np.load(args.samples_path)
        print(f"Loaded samples: {samples.shape}")
    except Exception as e:
        print(f"Error loading samples: {e}")
        return 1
    
    # Run experiments based on arguments
    all_metrics = {}
    
    if args.experiment in ['ebm', 'both']:
        if args.true_ebm_likelihood_path is None:
            print("Warning: true_ebm_likelihood_path not provided, skipping EBM experiment")
        else:
            try:
                true_ebm_ll = np.load(args.true_ebm_likelihood_path)
                predicted_ebm_ll, ebm_metrics = run_ebm_experiment(
                    args, interpolant_obj, samples, true_ebm_ll
                )
                all_metrics.update(ebm_metrics)
                
                # Save predictions
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / 'predicted_ebm_likelihoods.npy', predicted_ebm_ll)
                
            except Exception as e:
                print(f"Error in EBM experiment: {e}")
    
    if args.experiment in ['flow', 'both']:
        if args.true_flow_likelihood_path is None:
            print("Warning: true_flow_likelihood_path not provided, skipping flow experiment")
        else:
            try:
                true_flow_ll = np.load(args.true_flow_likelihood_path)
                predicted_flow_ll, flow_metrics = run_flow_experiment(
                    args, interpolant_obj, samples, true_flow_ll
                )
                all_metrics.update(flow_metrics)
                
                # Save predictions
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                np.save(output_dir / 'predicted_flow_likelihoods.npy', predicted_flow_ll)
                
            except Exception as e:
                print(f"Error in flow experiment: {e}")
    
    # Save all metrics
    if all_metrics:
        import json
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'comparison_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
    
    if wandb.run is not None:
        wandb.finish()
    
    print("\nExperiments completed!")
    return 0


if __name__ == "__main__":
    exit(main())