#!/usr/bin/env python3
"""
Script to calculate W2 (Wasserstein-2) distances for energy and torsion metrics.
Based on the W2 calculation methods from infer_aa2.py.

Usage:
    python calculate_w2_metrics.py --help
"""

import argparse
import numpy as np
import ot
import mdtraj as md
from typing import Tuple, Optional, List


def calc_energy_w2(gen_energies: np.ndarray, ref_energies: np.ndarray) -> float:
    """Calculate W2 distance between energy distributions using optimal transport.
    
    Args:
        gen_energies: Generated sample energies (1D array)
        ref_energies: Reference sample energies (1D array)
        
    Returns:
        W2 distance (square root of EMD with squared Euclidean metric)
    """
    gen_energies = np.asarray(gen_energies).ravel()
    ref_energies = np.asarray(ref_energies).ravel()
    
    # Use 1D EMD with squared Euclidean metric
    loss = ot.emd2_1d(gen_energies, ref_energies, metric="sqeuclidean")
    return np.sqrt(loss)


def calc_torsion_w2(gen_angles: np.ndarray, ref_angles: np.ndarray) -> float:
    """Calculate W2 distance between torsion angle distributions.
    
    Args:
        gen_angles: Generated torsion angles (N x 2 array for phi/psi pairs)
        ref_angles: Reference torsion angles (N x 2 array for phi/psi pairs)
        
    Returns:
        W2 distance for torsion angles on circular manifold
    """
    # Compute pairwise distances on circular manifold
    dist = np.expand_dims(gen_angles, 0) - np.expand_dims(ref_angles, 1)
    dist = np.sum((dist % np.pi)**2, axis=-1)
    
    # Create uniform distributions
    a = ot.unif(gen_angles.shape[0])
    b = ot.unif(ref_angles.shape[0])
    
    # Compute EMD
    W, log = ot.emd2(a, b, dist, log=True, numItermax=int(1e9))
    return np.sqrt(W)



def extract_torsion_angles_from_samples(samples_np):
    """Extract torsion angles from molecular samples (alanine dipeptide specific).
    
    Args:
        samples_np: Molecular samples (numpy array, shape: [n_samples, 66])
        
    Returns:
        numpy array: Torsion angles (shape: [n_samples, 2]) for phi and psi angles
    """
    # Import required functions (assuming they're available in the environment)
    try:
        from dataset.ad2_dataset import get_alanine_atom_types, get_alanine_implicit_dataset
    except ImportError:
        raise ImportError("Required modules for alanine dipeptide analysis not found. Please ensure the environment has access to the dataset modules.")
    
    def determine_chirality_batch(cartesian_coords_batch):
        """Determine chirality for batch of coordinates."""
        coords_batch = np.array(cartesian_coords_batch)
        
        if coords_batch.shape[-2:] != (4, 3):
            raise ValueError("Input should be a batch of four 3D Cartesian coordinates")
        
        # Calculate vectors from chirality centers to connected atoms
        vectors_batch = coords_batch - coords_batch[:, 0:1, :]
        
        # Calculate normal vectors of planes
        normal_vectors_batch = np.cross(vectors_batch[:, 1, :], vectors_batch[:, 2, :])
        
        # Calculate dot products
        dot_products_batch = np.einsum('...i,...i->...', normal_vectors_batch, vectors_batch[:, 3, :])
        
        # Determine chirality labels
        chirality_labels_batch = np.where(dot_products_batch > 0.000, 'L', 'D')
        
        return chirality_labels_batch
    
    # Get atom types and identify carbon positions
    atom_types = get_alanine_atom_types()
    atom_types[[4, 6, 8, 14, 16]] = np.arange(4, 9)
    carbon_pos = np.where(atom_types == 1)[0]
    
    # Reshape samples to [n_samples, 22, 3]
    carbon_samples_np = samples_np.reshape(-1, 22, 3)[:, carbon_pos]
    carbon_distances = np.linalg.norm(samples_np.reshape(-1, 22, 3)[:, [8]] - carbon_samples_np, axis=-1)
    
    # Find C-beta atom index
    cb_idx = np.where(carbon_distances == carbon_distances.min(1, keepdims=True))
    
    # Get backbone and C-beta samples
    back_bone_samples = samples_np.reshape(-1, 22, 3)[:, np.array([8, 6, 14])]
    cb_samples = samples_np.reshape(-1, 22, 3)[cb_idx[0], carbon_pos[cb_idx[1]]][:, None, :]
    
    # Determine chirality and apply mapping
    chirality = determine_chirality_batch(np.concatenate([back_bone_samples, cb_samples], axis=1))
    samples_np_mapped = samples_np.copy()
    samples_np_mapped[chirality == "D"] *= -1
    
    # Create trajectory and compute dihedral angles
    dataset = get_alanine_implicit_dataset()
    traj_samples = md.Trajectory(samples_np_mapped.reshape(-1, 22, 3), topology=dataset.system.mdtraj_topology)
    
    # Define phi and psi dihedral indices
    phi_indices, psi_indices = [4, 6, 8, 14], [6, 8, 14, 16]
    angles = md.compute_dihedrals(traj_samples, [phi_indices, psi_indices])
    
    return angles


def calculate_w2_metrics(
    gen_data_path: str,
    ref_samples: np.ndarray,
    ref_energies: np.ndarray,
    n_bootstrap: int = 5,
    subsample_size: int = 10000
) -> dict:
    """Calculate W2 metrics for energy and torsion angles with optional reweighting.
    
    Args:
        gen_data_path: Path to .npz file containing generated data (with keys: samples, energies, log_w)
        ref_samples: Reference samples (n_ref, 22, 3) for alanine dipeptide  
        ref_energies: Reference energies (n_ref,)
        n_bootstrap: Number of bootstrap iterations for error estimation
        subsample_size: Number of samples to use in each bootstrap iteration
        
    Returns:
        Dictionary containing W2 metrics and statistics
    """
    # Load generated data
    gen_data = np.load(gen_data_path)
    samples = gen_data['samples']  # Expected shape: (n_samples, 66)
    energies = gen_data['energies']  # Expected shape: (n_samples, 1) or (n_samples,)
    
    # Handle weights if available
    weights = None
    if 'log_w' in gen_data.files:
        log_w = gen_data['log_w']
        # Convert log weights to probabilities
        log_w_flat = log_w.ravel()
        log_w_stable = log_w_flat - np.max(log_w_flat)
        weights = np.exp(log_w_stable)
        weights = weights / np.sum(weights)
    
    # Flatten energies if needed
    energies = energies.ravel()
    ref_energies = ref_energies.ravel()
    
    # Reshape reference samples for torsion angle calculation (flatten to match expected format)  
    ref_samples_flat = ref_samples.reshape(-1, 66)  # Convert (n, 22, 3) to (n, 66)
    
    # Extract torsion angles using the alanine-specific function
    sample_angles = extract_torsion_angles_from_samples(samples)
    ref_angles = extract_torsion_angles_from_samples(ref_samples_flat)
    
    results = {
        'energy_w2_unweighted': [],
        'torsion_w2_unweighted': [],
        'energy_w2_reweighted': [],
        'torsion_w2_reweighted': []
    }
    
    # Bootstrap sampling for error estimation
    for i in range(n_bootstrap):
        # Subsample reference data
        if len(ref_energies) > subsample_size:
            ref_idx = np.random.choice(len(ref_energies), subsample_size, replace=False)
            ref_energies_sub = ref_energies[ref_idx]
            ref_angles_sub = ref_angles[ref_idx]
        else:
            ref_energies_sub = ref_energies
            ref_angles_sub = ref_angles
        
        # Unweighted (proposal) sampling
        if len(energies) > subsample_size:
            sample_idx = np.random.choice(len(energies), subsample_size, replace=False)
            energies_proposal = energies[sample_idx]
            angles_proposal = sample_angles[sample_idx]
        else:
            energies_proposal = energies
            angles_proposal = sample_angles
        
        # Calculate unweighted W2 distances
        energy_w2 = calc_energy_w2(energies_proposal, ref_energies_sub)
        torsion_w2 = calc_torsion_w2(angles_proposal, ref_angles_sub)
        
        results['energy_w2_unweighted'].append(energy_w2)
        results['torsion_w2_unweighted'].append(torsion_w2)
        
        # Reweighted sampling (if weights provided)
        if weights is not None:
            if len(energies) > subsample_size:
                reweighted_idx = np.random.choice(
                    len(energies), 
                    subsample_size, 
                    replace=False, 
                    p=weights
                )
                energies_reweighted = energies[reweighted_idx]
                angles_reweighted = sample_angles[reweighted_idx]
            else:
                energies_reweighted = energies
                angles_reweighted = sample_angles
            
            # Calculate reweighted W2 distances
            energy_w2_rw = calc_energy_w2(energies_reweighted, ref_energies_sub)
            torsion_w2_rw = calc_torsion_w2(angles_reweighted, ref_angles_sub)
            
            results['energy_w2_reweighted'].append(energy_w2_rw)
            results['torsion_w2_reweighted'].append(torsion_w2_rw)
    
    # Calculate statistics
    stats = {}
    for key, values in results.items():
        if values:  # Only calculate stats if we have values
            stats[f'{key}_mean'] = np.mean(values)
            stats[f'{key}_std'] = np.std(values)
            stats[f'{key}_values'] = values
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Calculate W2 metrics for molecular samples')
    parser.add_argument('--generated_data', required=True,
                       help='Path to .npz file containing generated data (with keys: samples, energies, log_w)')
    parser.add_argument('--reference_samples', required=True,
                       help='Path to numpy file containing reference samples (shape: n_samples, 22, 3)')
    parser.add_argument('--reference_energies', required=True,
                       help='Path to numpy file containing reference energies')
    parser.add_argument('--n_bootstrap', type=int, default=5,
                       help='Number of bootstrap iterations (default: 5)')
    parser.add_argument('--subsample_size', type=int, default=10000,
                       help='Subsample size for each bootstrap (default: 10000)')
    parser.add_argument('--output', default='w2_results.npz',
                       help='Output file for results (default: w2_results.npz)')
    
    args = parser.parse_args()
    
    # Load reference data
    print("Loading reference data...")
    reference_samples = np.load(args.reference_samples) 
    reference_energies = np.load(args.reference_energies)
    
    print(f"Reference samples shape: {reference_samples.shape}")
    print(f"Reference energies shape: {reference_energies.shape}")
    
    # Calculate W2 metrics
    print("Calculating W2 metrics...")
    results = calculate_w2_metrics(
        gen_data_path=args.generated_data,
        ref_samples=reference_samples,
        ref_energies=reference_energies,
        n_bootstrap=args.n_bootstrap,
        subsample_size=args.subsample_size
    )
    
    # Print results
    print("\nResults:")
    print("-" * 50)
    for key, value in results.items():
        if key.endswith('_values'):
            continue  # Skip printing the raw values
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    
    # Save results  
    np.savez(args.output, **results)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()