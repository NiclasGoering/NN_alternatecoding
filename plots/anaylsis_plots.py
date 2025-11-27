#!/usr/bin/env python3
"""
Analysis plots for experiment results.

This script creates comprehensive plots from experiment output directories.
It expects a directory structure like:
  experiment_dir/
    n128_lam0/
      algorithm_name/
        training_history.csv
        final_metrics.json
    n256_lam1e-3/
      ...
"""

import os
import json
import re
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from pathlib import Path
import ast
import torch
import copy

# Add src directory to path for model imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.data.models.ffnn import GatedMLP

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (12, 8)


def parse_dir_name(dirname):
    """Parse n_train and lambda from directory name like 'n128_lam0' or 'n1024_lam1e-3'"""
    match = re.match(r'n(\d+)_lam(.+)', dirname)
    if match:
        n_train = int(match.group(1))
        lam_str = match.group(2)
        # Convert lambda string to float
        if lam_str == '0':
            lam = 0.0
        else:
            lam = float(lam_str)
        return n_train, lam
    return None, None


def parse_list_string(s):
    """Parse a string representation of a list like '[1.0, 2.0, 3.0]'"""
    if pd.isna(s) or s == '' or s == '[]' or s == 'None' or str(s).strip() == 'None':
        return None
    try:
        # Try to parse as Python literal
        parsed = ast.literal_eval(str(s))
        # Handle None values in the list
        if parsed is None:
            return None
        # Filter out None values from the list and convert to float
        if isinstance(parsed, list):
            parsed = [float(v) for v in parsed if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if len(parsed) == 0:
                return None
        return parsed
    except:
        # If that fails, try to parse manually
        s_str = str(s).strip('[]')
        if not s_str or s_str == 'None':
            return None
        try:
            return [float(x.strip()) for x in s_str.split(',') if x.strip() != 'None' and x.strip() != '']
        except:
            return None


def load_experiment_data(experiment_dir):
    """Load all training histories and final metrics from experiment directory."""
    data = defaultdict(lambda: defaultdict(dict))  # algo -> n_train -> lambda -> {'history': df, 'final': dict}
    
    # Get all combination directories
    combo_dirs = [d for d in os.listdir(experiment_dir) 
                  if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith('n')]
    
    algorithms = set()
    
    for combo_dir in combo_dirs:
        n_train, lam = parse_dir_name(combo_dir)
        if n_train is None:
            continue
        
        combo_path = os.path.join(experiment_dir, combo_dir)
        
        # Find all algorithm subdirectories
        for item in os.listdir(combo_path):
            algo_path = os.path.join(combo_path, item)
            if not os.path.isdir(algo_path):
                continue
            
            algo = item
            algorithms.add(algo)
            
            # Load training history
            history_path = os.path.join(algo_path, "training_history.csv")
            final_path = os.path.join(algo_path, "final_metrics.json")
            
            history_df = None
            if os.path.exists(history_path):
                try:
                    history_df = pd.read_csv(history_path)
                except Exception as e:
                    print(f"Error loading {history_path}: {e}")
            
            final_metrics = None
            if os.path.exists(final_path):
                try:
                    with open(final_path, 'r') as f:
                        final_metrics = json.load(f)
                except Exception as e:
                    print(f"Error loading {final_path}: {e}")
            
            if history_df is not None or final_metrics is not None:
                data[algo][n_train][lam] = {
                    'history': history_df,
                    'final': final_metrics
                }
    
    return data, sorted(list(algorithms))


def get_epoch_column(df):
    """Get the epoch/cycle column name from dataframe."""
    if 'epoch' in df.columns:
        return 'epoch'
    elif 'cycle' in df.columns:
        return 'cycle'
    else:
        # Use index if no epoch column
        return None


def plot_training_error(data, algorithms, output_dir):
    """Plot 1: Grid with algorithms on x-axis, dataset sizes on y-axis, training error vs epochs color-coded by lambda."""
    all_n_trains = set()
    all_lambdas = set()
    for algo in algorithms:
        for n_train in data[algo].keys():
            all_n_trains.add(n_train)
            all_lambdas.update(data[algo][n_train].keys())
    all_n_trains = sorted(all_n_trains)
    all_lambdas = sorted(all_lambdas)
    
    # Create colormap for lambdas
    n_colors = len(all_lambdas)
    cmap = plt.cm.get_cmap('viridis', n_colors)
    lambda_to_color = {lam: cmap(i) for i, lam in enumerate(all_lambdas)}
    
    n_rows = len(all_n_trains)
    n_cols = len(algorithms)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # First pass: collect all y-values to determine shared y-axis limits
    all_y_values = []
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, algo in enumerate(algorithms):
            for lam in all_lambdas:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                
                if 'train_loss' not in history.columns:
                    continue
                
                y = history['train_loss'].values
                # Clamp values at 3
                y = np.clip(y, None, 3.0)
                all_y_values.extend(y[y > 0])  # Only positive values for log scale
    
    # Determine shared y-axis limits
    if len(all_y_values) > 0:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        # Add small padding
        y_min = y_min * 0.8
        y_max = y_max * 1.2
    else:
        y_min, y_max = 1e-5, 1.0
    
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, algo in enumerate(algorithms):
            ax = axes[row_idx][col_idx]
            
            for lam in all_lambdas:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                
                epoch_col = get_epoch_column(history)
                if epoch_col is None:
                    x = np.arange(len(history))
                else:
                    x = history[epoch_col].values
                
                if 'train_loss' not in history.columns:
                    continue
                
                y = history['train_loss'].values
                # Clamp values at 3
                y = np.clip(y, None, 3.0)
                
                # Format lambda for label
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.plot(x, y, 'o-', color=lambda_to_color[lam], 
                       label=f'λ={lam_label}', 
                       linewidth=1.5, markersize=3, alpha=0.7)
            
            ax.set_xlabel('Epoch/Cycle', fontsize=10)
            ax.set_ylabel('Training Error', fontsize=10)
            ax.set_title(f'{algo}\nn={n_train}', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc='best', ncol=1)
    
    plt.suptitle('Training Error vs Epochs', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot1_training_error.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_generalization_error(data, algorithms, output_dir):
    """Plot 2: Grid with algorithms on x-axis, dataset sizes on y-axis, generalization error vs epochs color-coded by lambda."""
    all_n_trains = set()
    all_lambdas = set()
    for algo in algorithms:
        for n_train in data[algo].keys():
            all_n_trains.add(n_train)
            all_lambdas.update(data[algo][n_train].keys())
    all_n_trains = sorted(all_n_trains)
    all_lambdas = sorted(all_lambdas)
    
    # Create colormap for lambdas
    n_colors = len(all_lambdas)
    cmap = plt.cm.get_cmap('viridis', n_colors)
    lambda_to_color = {lam: cmap(i) for i, lam in enumerate(all_lambdas)}
    
    n_rows = len(all_n_trains)
    n_cols = len(algorithms)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # First pass: collect all y-values to determine shared y-axis limits
    all_y_values = []
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, algo in enumerate(algorithms):
            for lam in all_lambdas:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                
                # Use test_loss if available, otherwise val_loss
                if 'test_loss' in history.columns:
                    y = history['test_loss'].values
                elif 'val_loss' in history.columns:
                    y = history['val_loss'].values
                else:
                    continue
                
                # Clamp values at 3
                y = np.clip(y, None, 3.0)
                all_y_values.extend(y[y > 0])  # Only positive values for log scale
    
    # Determine shared y-axis limits
    if len(all_y_values) > 0:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        # Add small padding
        y_min = y_min * 0.8
        y_max = y_max * 1.2
    else:
        y_min, y_max = 1e-5, 1.0
    
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, algo in enumerate(algorithms):
            ax = axes[row_idx][col_idx]
            
            for lam in all_lambdas:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                
                epoch_col = get_epoch_column(history)
                if epoch_col is None:
                    x = np.arange(len(history))
                else:
                    x = history[epoch_col].values
                
                # Use test_loss if available, otherwise val_loss
                if 'test_loss' in history.columns:
                    y = history['test_loss'].values
                elif 'val_loss' in history.columns:
                    y = history['val_loss'].values
                else:
                    continue
                
                # Clamp values at 3
                y = np.clip(y, None, 3.0)
                
                # Format lambda for label
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.plot(x, y, 'o-', color=lambda_to_color[lam], 
                       label=f'λ={lam_label}', 
                       linewidth=1.5, markersize=3, alpha=0.7)
            
            ax.set_xlabel('Epoch/Cycle', fontsize=10)
            ax.set_ylabel('Generalization Error', fontsize=10)
            ax.set_title(f'{algo}\nn={n_train}', fontsize=11, fontweight='bold')
            ax.set_yscale('log')
            ax.set_ylim(y_min, y_max)
            ax.grid(True, alpha=0.3)
            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=7, loc='best', ncol=1)
    
    plt.suptitle('Generalization Error vs Epochs', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot2_generalization_error.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_gen_error_vs_train_size(data, algorithms, output_dir):
    """Plot: Generalization error vs training set size, separate subplot for each algorithm, color-coded by lambda."""
    all_n_trains = set()
    all_lambdas = set()
    for algo in algorithms:
        for n_train in data[algo].keys():
            all_n_trains.add(n_train)
            all_lambdas.update(data[algo][n_train].keys())
    all_n_trains = sorted(all_n_trains)
    all_lambdas = sorted(all_lambdas)
    
    # Create colormap for lambdas
    n_colors = len(all_lambdas)
    cmap = plt.cm.get_cmap('viridis', n_colors)
    lambda_to_color = {lam: cmap(i) for i, lam in enumerate(all_lambdas)}
    
    n_algorithms = len(algorithms)
    fig, axes = plt.subplots(1, n_algorithms, figsize=(6*n_algorithms, 6))
    if n_algorithms == 1:
        axes = [axes]
    
    for algo_idx, algo in enumerate(algorithms):
        ax = axes[algo_idx]
        
        for lam in all_lambdas:
            n_trains = []
            gen_errors = []
            
            for n_train in all_n_trains:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                # Get final metrics (preferred) or last value from history
                final_metrics = data[algo][n_train][lam]['final']
                history = data[algo][n_train][lam]['history']
                
                gen_error = None
                if final_metrics is not None:
                    # Prefer test_loss, then val_loss
                    if 'test_loss' in final_metrics:
                        gen_error = final_metrics['test_loss']
                    elif 'val_loss' in final_metrics:
                        gen_error = final_metrics['val_loss']
                elif history is not None and len(history) > 0:
                    # Use last value from history
                    if 'test_loss' in history.columns:
                        gen_error = history['test_loss'].iloc[-1]
                    elif 'val_loss' in history.columns:
                        gen_error = history['val_loss'].iloc[-1]
                
                if gen_error is not None:
                    # Clamp value at 3
                    gen_error = min(gen_error, 3.0)
                    n_trains.append(n_train)
                    gen_errors.append(gen_error)
            
            if len(n_trains) > 0:
                # Format lambda for label
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.plot(n_trains, gen_errors, 'o-',
                       color=lambda_to_color[lam], 
                       label=f'λ={lam_label}',
                       linewidth=2, markersize=8, alpha=0.8)
        
        ax.set_xlabel('Training Set Size', fontsize=12)
        ax.set_ylabel('Generalization Error', fontsize=12)
        ax.set_title(f'{algo}', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        if algo_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9, ncol=1)
    
    plt.suptitle('Generalization Error vs Training Set Size', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot3_gen_error_vs_train_size.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_grid(data, algo, metric_name, metric_key, output_dir, is_layerwise=False):
    """Plot a grid: x-axis = lambda, y-axis = dataset size, showing metric vs epochs."""
    all_n_trains = sorted(data[algo].keys())
    all_lambdas = set()
    for n_train in all_n_trains:
        all_lambdas.update(data[algo][n_train].keys())
    all_lambdas = sorted(all_lambdas)
    
    n_rows = len(all_n_trains)
    n_cols = len(all_lambdas)
    
    if n_rows == 0 or n_cols == 0:
        return
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]
    
    # Determine number of layers if layerwise
    n_layers = None
    if is_layerwise:
        for n_train in all_n_trains:
            for lam in all_lambdas:
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                if metric_key in history.columns:
                    # Try to find first non-None value to determine n_layers
                    for val in history[metric_key]:
                        if pd.isna(val) or val == '' or str(val).strip() == 'None':
                            continue
                        layer_list = parse_list_string(val)
                        if layer_list is not None and isinstance(layer_list, list) and len(layer_list) > 0:
                            n_layers = len(layer_list)
                            break
                    if n_layers is not None:
                        break
            if n_layers is not None:
                break
        
        if n_layers is None:
            is_layerwise = False
    
    # Create colormap for layers
    if is_layerwise and n_layers:
        layer_cmap = plt.cm.get_cmap('tab10', n_layers)
        layer_colors = [layer_cmap(i) for i in range(n_layers)]
    
    # First pass: collect all y-values to determine shared y-axis limits
    all_y_values = []
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, lam in enumerate(all_lambdas):
            if n_train not in data[algo] or lam not in data[algo][n_train]:
                continue
            
            history = data[algo][n_train][lam]['history']
            if history is None or len(history) == 0 or metric_key not in history.columns:
                continue
            
            if is_layerwise and n_layers:
                # Collect layer values
                for layer_idx in range(n_layers):
                    layer_values = []
                    for val_str in history[metric_key]:
                        # Handle None, NaN, or empty strings
                        if pd.isna(val_str) or val_str == '' or str(val_str).strip() == 'None':
                            layer_values.append(np.nan)
                            continue
                        layer_list = parse_list_string(val_str)
                        if layer_list is not None and isinstance(layer_list, list) and layer_idx < len(layer_list):
                            val = layer_list[layer_idx]
                            # Check if value is valid (not None, not NaN)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                layer_values.append(float(val))
                            else:
                                layer_values.append(np.nan)
                        else:
                            layer_values.append(np.nan)
                    
                    valid_values = [v for v in layer_values if not np.isnan(v) and v is not None]
                    if len(valid_values) > 0:
                        all_y_values.extend(valid_values)
            else:
                # Collect single values
                values = history[metric_key].values
                try:
                    values = pd.to_numeric(values, errors='coerce')
                except:
                    pass
                
                # Filter out None, NaN, and string 'None'
                valid_values = [v for v in values if not pd.isna(v) and v != 'None' and v is not None]
                if len(valid_values) > 0:
                    all_y_values.extend(valid_values)
    
    # Determine shared y-axis limits
    if len(all_y_values) > 0:
        y_min = min(all_y_values)
        y_max = max(all_y_values)
        # Add padding (10% on each side)
        y_range = y_max - y_min
        if y_range > 0:
            y_min = y_min - 0.1 * y_range
            y_max = y_max + 0.1 * y_range
        else:
            y_min = y_min * 0.9 if y_min > 0 else y_min - 0.1
            y_max = y_max * 1.1 if y_max > 0 else y_max + 0.1
    else:
        y_min, y_max = None, None
    
    for row_idx, n_train in enumerate(all_n_trains):
        for col_idx, lam in enumerate(all_lambdas):
            ax = axes[row_idx][col_idx]
            
            if n_train not in data[algo] or lam not in data[algo][n_train]:
                ax.axis('off')
                continue
            
            history = data[algo][n_train][lam]['history']
            if history is None or len(history) == 0 or metric_key not in history.columns:
                ax.axis('off')
                continue
            
            epoch_col = get_epoch_column(history)
            if epoch_col is None:
                x = np.arange(len(history))
            else:
                x = history[epoch_col].values
            
            if is_layerwise and n_layers:
                # Plot each layer separately
                for layer_idx in range(n_layers):
                    layer_values = []
                    for val_str in history[metric_key]:
                        # Handle None, NaN, or empty strings
                        if pd.isna(val_str) or val_str == '' or str(val_str).strip() == 'None':
                            layer_values.append(np.nan)
                            continue
                        layer_list = parse_list_string(val_str)
                        if layer_list is not None and isinstance(layer_list, list) and layer_idx < len(layer_list):
                            val = layer_list[layer_idx]
                            # Check if value is valid (not None, not NaN)
                            if val is not None and not (isinstance(val, float) and np.isnan(val)):
                                layer_values.append(float(val))
                            else:
                                layer_values.append(np.nan)
                        else:
                            layer_values.append(np.nan)
                    
                    # Filter out NaN values for plotting
                    valid_mask = ~np.isnan(layer_values)
                    if np.any(valid_mask) and np.sum(valid_mask) > 0:
                        x_valid = x[valid_mask]
                        values_valid = np.array(layer_values)[valid_mask]
                        # Only plot if we have at least one valid point
                        if len(values_valid) > 0:
                            ax.plot(x_valid, values_valid, 'o-', color=layer_colors[layer_idx],
                                   label=f'Layer {layer_idx}', linewidth=1.5, markersize=2, alpha=0.7)
            else:
                # Plot single value
                values = history[metric_key].values
                # Try to convert to numeric if needed
                try:
                    values = pd.to_numeric(values, errors='coerce')
                except:
                    pass
                
                # Filter out None, NaN, and invalid values
                valid_mask = ~pd.isna(values)
                if np.any(valid_mask):
                    x_valid = x[valid_mask]
                    values_valid = values[valid_mask]
                    ax.plot(x_valid, values_valid, 'o-', color='blue', linewidth=1.5, markersize=2, alpha=0.7)
            
            # Format lambda for title
            if lam == 0:
                lam_label = '0'
            elif lam < 0.001:
                lam_label = f'{lam:.0e}'
            else:
                lam_label = f'{lam:.3f}'
            
            ax.set_title(f'n={n_train}, λ={lam_label}', fontsize=9)
            ax.set_xlabel('Epoch/Cycle', fontsize=8)
            ax.set_ylabel(metric_name, fontsize=8)
            if is_layerwise and n_layers:
                ax.legend(fontsize=6, loc='best')
            ax.grid(True, alpha=0.3)
            if metric_name in ['Training Error', 'Generalization Error']:
                ax.set_yscale('log')
                if y_min is not None and y_max is not None and y_min > 0:
                    ax.set_ylim(y_min, y_max)
            elif y_min is not None and y_max is not None:
                ax.set_ylim(y_min, y_max)
            
            # Special handling for path-centric metrics
            if 'H_path' in metric_name or 'H_gain' in metric_name:
                # Entropy metrics: ensure y-axis starts at 0
                if y_min is not None and y_min < 0:
                    ax.set_ylim(bottom=0)
            elif 'NRI' in metric_name or 'Neural-Race Index' in metric_name:
                # NRI is typically between 0 and 1
                if y_min is not None and y_max is not None:
                    ax.set_ylim(max(0, y_min), min(1.1, y_max) if y_max > 1 else y_max)
    
    plt.suptitle(f'{algo} - {metric_name}', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    safe_name = metric_name.lower().replace(' ', '_').replace('/', '_')
    plt.savefig(os.path.join(output_dir, f'{algo}_{safe_name}_grid.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_path_snr_components(data, algo, output_dir):
    """Plot Path SNR components (c_gamma, N_gamma, SNR_gamma) with mean/median and std ribbon."""
    all_n_trains = sorted(data[algo].keys())
    all_lambdas = set()
    for n_train in all_n_trains:
        all_lambdas.update(data[algo][n_train].keys())
    all_lambdas = sorted(all_lambdas)
    
    n_rows = len(all_n_trains)
    n_cols = len(all_lambdas)
    
    if n_rows == 0 or n_cols == 0:
        return
    
    # Three components to plot
    components = [
        ('Label Correlation (c_gamma)', 'path_snr_c_gamma', 'blue'),
        ('Sample Support (N_gamma)', 'path_snr_N_gamma', 'green'),
        ('Signal-to-Noise (SNR_gamma)', 'path_snr_SNR_gamma', 'red'),
    ]
    
    for comp_name, comp_prefix, comp_color in components:
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # Collect all y-values for shared y-axis
        all_y_values = []
        for row_idx, n_train in enumerate(all_n_trains):
            for col_idx, lam in enumerate(all_lambdas):
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    continue
                
                mean_key = f"{comp_prefix}_mean"
                median_key = f"{comp_prefix}_median"
                std_key = f"{comp_prefix}_std"
                
                if mean_key in history.columns:
                    mean_vals = pd.to_numeric(history[mean_key], errors='coerce')
                    valid_mean = [v for v in mean_vals if not pd.isna(v)]
                    if len(valid_mean) > 0:
                        all_y_values.extend(valid_mean)
                
                if median_key in history.columns:
                    median_vals = pd.to_numeric(history[median_key], errors='coerce')
                    valid_median = [v for v in median_vals if not pd.isna(v)]
                    if len(valid_median) > 0:
                        all_y_values.extend(valid_median)
        
        # Determine shared y-axis limits
        if len(all_y_values) > 0:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            y_range = y_max - y_min
            if y_range > 0:
                y_min = y_min - 0.1 * y_range
                y_max = y_max + 0.1 * y_range
            else:
                y_min = y_min * 0.9 if y_min > 0 else y_min - 0.1
                y_max = y_max * 1.1 if y_max > 0 else y_max + 0.1
        else:
            y_min, y_max = None, None
        
        # Plot each subplot
        for row_idx, n_train in enumerate(all_n_trains):
            for col_idx, lam in enumerate(all_lambdas):
                ax = axes[row_idx][col_idx]
                
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    ax.axis('off')
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0:
                    ax.axis('off')
                    continue
                
                mean_key = f"{comp_prefix}_mean"
                median_key = f"{comp_prefix}_median"
                std_key = f"{comp_prefix}_std"
                
                if mean_key not in history.columns and median_key not in history.columns:
                    ax.axis('off')
                    continue
                
                epoch_col = get_epoch_column(history)
                if epoch_col is None:
                    x = np.arange(len(history))
                else:
                    x = history[epoch_col].values
                
                # Plot mean with std ribbon
                if mean_key in history.columns:
                    mean_vals = pd.to_numeric(history[mean_key], errors='coerce').values
                    std_vals = pd.to_numeric(history[std_key], errors='coerce').values if std_key in history.columns else np.zeros_like(mean_vals)
                    
                    valid_mask = ~pd.isna(mean_vals)
                    if np.any(valid_mask):
                        x_valid = x[valid_mask]
                        mean_valid = mean_vals[valid_mask]
                        std_valid = std_vals[valid_mask]
                        
                        # Plot mean line
                        ax.plot(x_valid, mean_valid, 'o-', color=comp_color, 
                               label='Mean', linewidth=1.5, markersize=2, alpha=0.8)
                        
                        # Plot std ribbon
                        ax.fill_between(x_valid, 
                                       mean_valid - std_valid, 
                                       mean_valid + std_valid,
                                       color=comp_color, alpha=0.2, label='±1 std')
                
                # Plot median (optional, as dashed line)
                if median_key in history.columns:
                    median_vals = pd.to_numeric(history[median_key], errors='coerce').values
                    valid_mask = ~pd.isna(median_vals)
                    if np.any(valid_mask):
                        x_valid = x[valid_mask]
                        median_valid = median_vals[valid_mask]
                        ax.plot(x_valid, median_valid, '--', color=comp_color, 
                               label='Median', linewidth=1.5, alpha=0.6)
                
                # Format
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.set_title(f'n={n_train}, λ={lam_label}', fontsize=9)
                ax.set_xlabel('Epoch/Cycle', fontsize=8)
                ax.set_ylabel(comp_name, fontsize=8)
                ax.legend(fontsize=6, loc='best')
                ax.grid(True, alpha=0.3)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
        
        plt.suptitle(f'{algo} - Path SNR: {comp_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        safe_name = comp_name.lower().replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
        plt.savefig(os.path.join(output_dir, f'{algo}_path_snr_{safe_name}_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()


def plot_gate_stats_grid(data, algo, output_dir):
    """Special handling for gate_stats which is a dictionary."""
    all_n_trains = sorted(data[algo].keys())
    all_lambdas = set()
    for n_train in all_n_trains:
        all_lambdas.update(data[algo][n_train].keys())
    all_lambdas = sorted(all_lambdas)
    
    gate_stat_types = ['linear', 'relu', 'dead', 'other']
    
    for stat_type in gate_stat_types:
        metric_name = f'Gate Stats: {stat_type}'
        metric_key = 'gate_stats'
        
        n_rows = len(all_n_trains)
        n_cols = len(all_lambdas)
        
        if n_rows == 0 or n_cols == 0:
            continue
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # First pass: collect all y-values to determine shared y-axis limits
        all_y_values = []
        for row_idx, n_train in enumerate(all_n_trains):
            for col_idx, lam in enumerate(all_lambdas):
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0 or metric_key not in history.columns:
                    continue
                
                # Extract the specific stat type from gate_stats dictionaries
                values = []
                for val_str in history[metric_key]:
                    if pd.isna(val_str) or val_str == '':
                        values.append(np.nan)
                    else:
                        try:
                            gate_dict = ast.literal_eval(str(val_str))
                            if isinstance(gate_dict, dict) and stat_type in gate_dict:
                                values.append(gate_dict[stat_type])
                            else:
                                values.append(np.nan)
                        except:
                            values.append(np.nan)
                
                valid_values = [v for v in values if not np.isnan(v)]
                if len(valid_values) > 0:
                    all_y_values.extend(valid_values)
        
        # Determine shared y-axis limits
        if len(all_y_values) > 0:
            y_min = min(all_y_values)
            y_max = max(all_y_values)
            # Add padding (10% on each side)
            y_range = y_max - y_min
            if y_range > 0:
                y_min = y_min - 0.1 * y_range
                y_max = y_max + 0.1 * y_range
            else:
                y_min = y_min * 0.9 if y_min > 0 else y_min - 0.1
                y_max = y_max * 1.1 if y_max > 0 else y_max + 0.1
        else:
            y_min, y_max = None, None
        
        for row_idx, n_train in enumerate(all_n_trains):
            for col_idx, lam in enumerate(all_lambdas):
                ax = axes[row_idx][col_idx]
                
                if n_train not in data[algo] or lam not in data[algo][n_train]:
                    ax.axis('off')
                    continue
                
                history = data[algo][n_train][lam]['history']
                if history is None or len(history) == 0 or metric_key not in history.columns:
                    ax.axis('off')
                    continue
                
                epoch_col = get_epoch_column(history)
                if epoch_col is None:
                    x = np.arange(len(history))
                else:
                    x = history[epoch_col].values
                
                # Extract the specific stat type from gate_stats dictionaries
                values = []
                for val_str in history[metric_key]:
                    if pd.isna(val_str) or val_str == '':
                        values.append(np.nan)
                    else:
                        try:
                            gate_dict = ast.literal_eval(str(val_str))
                            if isinstance(gate_dict, dict) and stat_type in gate_dict:
                                values.append(gate_dict[stat_type])
                            else:
                                values.append(np.nan)
                        except:
                            values.append(np.nan)
                
                if not all(np.isnan(v) for v in values):
                    ax.plot(x, values, 'o-', color='blue', linewidth=1.5, markersize=2, alpha=0.7)
                
                # Format lambda for title
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.set_title(f'n={n_train}, λ={lam_label}', fontsize=9)
                ax.set_xlabel('Epoch/Cycle', fontsize=8)
                ax.set_ylabel(metric_name, fontsize=8)
                ax.grid(True, alpha=0.3)
                if y_min is not None and y_max is not None:
                    ax.set_ylim(y_min, y_max)
        
        plt.suptitle(f'{algo} - {metric_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        safe_name = f'gate_stats_{stat_type}'
        plt.savefig(os.path.join(output_dir, f'{algo}_{safe_name}_grid.png'), dpi=300, bbox_inches='tight')
        plt.close()


def load_model_a_values(model_path, input_dim, widths, use_gates=True):
    """Load a model and extract a_plus and a_minus values for each layer."""
    try:
        model = GatedMLP(input_dim, widths, bias=False, use_gates=use_gates)
        state_dict = torch.load(model_path, map_location='cpu')
        model.load_state_dict(state_dict)
        model.eval()
        
        a_values_per_layer = []
        if use_gates and model.gates is not None:
            for gate in model.gates:
                a_plus = gate.a_plus.detach().cpu().numpy()
                a_minus = gate.a_minus.detach().cpu().numpy()
                # Combine a_plus and a_minus into one array
                a_combined = np.concatenate([a_plus, a_minus])
                a_values_per_layer.append(a_combined)
        
        if len(a_values_per_layer) == 0:
            print(f"    Warning: No a values extracted from {model_path}")
            return None
        
        return a_values_per_layer
    except Exception as e:
        print(f"    Error loading model from {model_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def plot_a_values_histogram_grid(experiment_dir, algorithms, output_dir):
    """Plot grid with train set size x-axis, lambda y-axis, showing histograms of a values.
    
    For each cell, plots histograms for each layer (final state only).
    """
    # Load config to get model architecture
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: config.json not found at {config_path}, skipping a values histogram plot")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model architecture
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    
    # Determine input dimension from dataset
    if dataset_cfg.get("name") == "hierarchical_xor":
        # For hierarchical_xor, input dim = (m^L) * s * 2 + distractor_dims
        # Each synonym is a pair (2 features), and there are s synonyms per group
        L = dataset_cfg.get("L", 3)
        m = dataset_cfg.get("m", 2)
        s = dataset_cfg.get("s", 4)
        distractor_dims = dataset_cfg.get("distractor_dims", 128)
        n_groups = m ** L
        d_leaf = n_groups * s * 2  # 2 features per synonym
        input_dim = d_leaf + distractor_dims
    else:
        # Default: try to get d from dataset config
        input_dim = dataset_cfg.get("d", 35)
    
    widths = model_cfg.get("widths", [256, 256])
    use_gates = model_cfg.get("use_gates", True)
    n_layers = len(widths)
    
    # Get all n_train and lambda combinations, and map them to directory names
    all_n_trains = set()
    all_lambdas = set()
    dir_name_map = {}  # (n_train, lam) -> directory_name
    
    combo_dirs = [d for d in os.listdir(experiment_dir) 
                  if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith('n')]
    
    for combo_dir in combo_dirs:
        n_train, lam = parse_dir_name(combo_dir)
        if n_train is not None:
            all_n_trains.add(n_train)
            all_lambdas.add(lam)
            dir_name_map[(n_train, lam)] = combo_dir
    
    all_n_trains = sorted(all_n_trains)
    all_lambdas = sorted(all_lambdas)
    
    if len(all_n_trains) == 0 or len(all_lambdas) == 0:
        print("Warning: No valid n_train/lambda combinations found, skipping a values histogram plot")
        return
    
    # Create separate plot for each algorithm
    for algo in algorithms:
        print(f"  Creating a values histogram grid for {algo}...")
        
        # Check if this algorithm uses gates (skip sgd_relu which doesn't use gates)
        if not use_gates or algo == "sgd_relu":
            print(f"    Skipping {algo} (doesn't use gates)")
            continue
        
        n_rows = len(all_lambdas)
        n_cols = len(all_n_trains)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        
        # Define colors for each layer
        layer_colors = plt.cm.tab10(np.linspace(0, 1, n_layers))
        final_alpha = 0.8
        final_linestyle = '-'
        
        # First pass: Collect all final a values to determine shared x-axis limits
        all_a_values = []
        data_cache = {}  # Cache loaded data
        
        for lam in all_lambdas:
            for n_train in all_n_trains:
                combo_dir = dir_name_map.get((n_train, lam))
                if combo_dir is None:
                    continue
                algo_path = os.path.join(experiment_dir, combo_dir, algo)
                
                if not os.path.exists(algo_path):
                    continue
                
                # Load final model only
                final_path = os.path.join(algo_path, "model_final.pt")
                
                a_final = None
                
                if os.path.exists(final_path):
                    a_final = load_model_a_values(final_path, input_dim, widths, use_gates)
                    if a_final:
                        for layer_vals in a_final:
                            all_a_values.extend(layer_vals)
                        data_cache[(lam, n_train, 'final')] = a_final
        
        # Determine shared x-axis limits
        if len(all_a_values) > 0:
            x_min = min(all_a_values)
            x_max = max(all_a_values)
            x_range = x_max - x_min
            if x_range == 0:
                x_range = 1.0
            x_min = x_min - 0.1 * x_range
            x_max = x_max + 0.1 * x_range
            print(f"    Found {len(all_a_values)} a values, range: [{x_min:.3f}, {x_max:.3f}]")
        else:
            print(f"    Warning: No a values found for {algo}, skipping plot")
            plt.close(fig)
            continue
        
        # Second pass: Plot histograms
        for row_idx, lam in enumerate(all_lambdas):
            for col_idx, n_train in enumerate(all_n_trains):
                ax = axes[row_idx][col_idx]
                
                combo_dir = dir_name_map.get((n_train, lam))
                if combo_dir is None:
                    ax.axis('off')
                    continue
                algo_path = os.path.join(experiment_dir, combo_dir, algo)
                
                if not os.path.exists(algo_path):
                    ax.axis('off')
                    continue
                
                # Get cached data
                a_final = data_cache.get((lam, n_train, 'final'))
                
                if a_final is None:
                    ax.axis('off')
                    continue
                
                # Plot histograms for each layer (final only)
                for layer_idx in range(n_layers):
                    color = layer_colors[layer_idx]
                    
                    # Plot final histogram
                    if layer_idx < len(a_final):
                        ax.hist(a_final[layer_idx], bins=50, alpha=final_alpha, 
                               color=color, linestyle=final_linestyle, 
                               label=f'L{layer_idx}', density=True, 
                               histtype='step', linewidth=2.5, edgecolor=color)
                
                # Format lambda for title
                if lam == 0:
                    lam_label = '0'
                elif lam < 0.001:
                    lam_label = f'{lam:.0e}'
                else:
                    lam_label = f'{lam:.3f}'
                
                ax.set_title(f'n={n_train}, λ={lam_label}', fontsize=8)
                ax.set_xlabel('a values', fontsize=7)
                ax.set_ylabel('Density', fontsize=7)
                ax.set_xlim(x_min, x_max)
                ax.set_yscale('log')
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=6)
                
                # Add legend only to first subplot
                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=5, loc='best', ncol=2, framealpha=0.9)
        
        plt.suptitle(f'{algo} - A Values Histogram (Final)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{algo}_a_values_histogram_grid.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def run_path_analysis_on_final_models(experiment_dir, algorithms, output_dir):
    """Run path analysis on final models and save plots to path_analysis subfolder."""
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from src.analysis.path_analysis import run_full_analysis_at_checkpoint
    from src.data.models.ffnn import GatedMLP
    from torch.utils.data import DataLoader
    import json
    
    # Load config to get model architecture
    config_path = os.path.join(experiment_dir, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: config.json not found at {config_path}, skipping path analysis")
        return
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Get model architecture
    dataset_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    
    # Determine input dimension
    if dataset_cfg.get("name") == "hierarchical_xor":
        L = dataset_cfg.get("L", 3)
        m = dataset_cfg.get("m", 2)
        s = dataset_cfg.get("s", 4)
        distractor_dims = dataset_cfg.get("distractor_dims", 128)
        n_groups = m ** L
        d_leaf = n_groups * s * 2
        input_dim = d_leaf + distractor_dims
    else:
        input_dim = dataset_cfg.get("d", 35)
    
    widths = model_cfg.get("widths", [256, 256])
    use_gates = model_cfg.get("use_gates", True)
    
    # Create path_analysis subdirectory in plots
    path_analysis_dir = os.path.join(output_dir, "path_analysis")
    os.makedirs(path_analysis_dir, exist_ok=True)
    
    # Get all combination directories
    combo_dirs = [d for d in os.listdir(experiment_dir) 
                  if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith('n')]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for combo_dir in combo_dirs:
        n_train, lam = parse_dir_name(combo_dir)
        if n_train is None:
            continue
        
        combo_path = os.path.join(experiment_dir, combo_dir)
        
        for algo in algorithms:
            algo_path = os.path.join(combo_path, algo)
            if not os.path.isdir(algo_path):
                continue
            
            final_model_path = os.path.join(algo_path, "model_final.pt")
            if not os.path.exists(final_model_path):
                print(f"  Skipping {algo} in {combo_dir}: model_final.pt not found")
                continue
            
            print(f"  Running path analysis for {algo} in {combo_dir}...")
            
            try:
                # Load model
                model = GatedMLP(input_dim, widths, bias=False, use_gates=use_gates)
                state_dict = torch.load(final_model_path, map_location=device)
                model.load_state_dict(state_dict)
                model.to(device).eval()
                
                # Recreate dataloader (we need val_loader for path analysis)
                # Try to load from dataset config
                dataset_name = dataset_cfg.get("name", "ksparse_parity").lower()
                
                if dataset_name == "hierarchical_xor":
                    from src.data.hierarchical_xor import build_hierarchical_xor_datasets, HierarchicalXORDataset
                    temp_cfg = copy.deepcopy(config)
                    temp_cfg["dataset"]["n_train"] = n_train
                    Xtr, ytr, Xva, yva, Xte, yte, meta, groups_tr, groups_va, groups_te = build_hierarchical_xor_datasets(temp_cfg)
                    n_groups = meta.get("n_groups")
                    ds_va = HierarchicalXORDataset(Xva, yva, groups=groups_va, n_groups=n_groups)
                    val_loader = DataLoader(ds_va, batch_size=len(ds_va), shuffle=False)
                elif dataset_name == "synonym_tree":
                    from src.data.hierarchical_synonyms import build_synonym_tree_datasets, SynonymTreeDataset
                    temp_cfg = copy.deepcopy(config)
                    temp_cfg["dataset"]["n_train"] = n_train
                    Xtr, ytr, Xva, yva, Xte, yte, meta = build_synonym_tree_datasets(temp_cfg)
                    ds_va = SynonymTreeDataset(Xva, yva)
                    val_loader = DataLoader(ds_va, batch_size=len(ds_va), shuffle=False)
                else:
                    # Default: ksparse_parity
                    from src.data.ksparse_parity import gen_ksparse_parity, ParityDataset
                    d, k = dataset_cfg["d"], dataset_cfg["k"]
                    nva = dataset_cfg["n_val"]
                    x_dist = dataset_cfg.get("x_dist", "pm1")
                    noise = dataset_cfg.get("label_noise", 0.0)
                    seed = config["seed"]
                    Xva, yva, _ = gen_ksparse_parity(d, k, nva, x_dist=x_dist, label_noise=noise, seed=seed+1)
                    ds_va = ParityDataset(Xva, yva)
                    val_loader = DataLoader(ds_va, batch_size=len(ds_va), shuffle=False)
                
                # Run path analysis
                step_tag = f"{algo}_n{n_train}_lam{lam}_final"
                run_full_analysis_at_checkpoint(
                    model=model,
                    val_loader=val_loader,
                    out_dir=path_analysis_dir,
                    step_tag=step_tag,
                    kernel_k=48,
                    kernel_mode="routing_gain",
                    include_input_in_kernel=True,
                    block_size=1024,
                    max_samples_kernel=5000,
                    max_samples_embed=5000,
                )
                print(f"    ✓ Completed path analysis for {algo} in {combo_dir}")
            except Exception as e:
                print(f"    ✗ Error in path analysis for {algo} in {combo_dir}: {e}")
                import traceback
                traceback.print_exc()


def main():
    # Set the experiment directory here
    experiment_dir = "/home/goring/NN_alternatecoding/outputs/24_11/hierarchical_xor_run_3_20251124_191204"
    
    if not os.path.exists(experiment_dir):
        raise ValueError(f"Experiment directory does not exist: {experiment_dir}\n"
                        f"Please update the path in the main() function.")
    
    # Create output directory for plots
    output_dir = os.path.join(experiment_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading data from: {experiment_dir}")
    data, algorithms = load_experiment_data(experiment_dir)
    
    print(f"Found {len(algorithms)} algorithms: {algorithms}")
    for algo in algorithms:
        n_train_counts = len(data[algo])
        print(f"  {algo}: {n_train_counts} training sizes")
    
    # Plot 1: Training error
    print("\nCreating Plot 1: Training Error...")
    plot_training_error(data, algorithms, output_dir)
    
    # Plot 2: Generalization error
    print("Creating Plot 2: Generalization Error...")
    plot_generalization_error(data, algorithms, output_dir)
    
    # Plot 3: Generalization error vs training set size
    print("Creating Plot 3: Generalization Error vs Training Set Size...")
    plot_gen_error_vs_train_size(data, algorithms, output_dir)
    
    # Plot 4+: Metric grids for each algorithm
    print("\nCreating metric grid plots for each algorithm...")
    
    # Define metrics to plot
    metrics_to_plot = [
        # Slope-based metrics
        ('Slope Budget Total', 'slope_budget_total', False),
        ('Slope Budget Per Layer', 'slope_budget_layers', True),
        ('Slope Entropy Total', 'slope_entropy_total', False),
        ('Slope Entropy Per Layer', 'slope_entropy_layers', True),
        ('Slope Deviation Per Layer', 'slope_deviation_layers', True),
        ('Mask Churn Per Layer', 'mask_churn_layers', True),
        ('Effective Rank Per Layer', 'effective_rank_layers', True),
        # Path metrics - the 4 key metrics (old)
        ('Path Sparsity & Concentration (Who is winning the race?)', 'active_path_complexity', True),
        ('Path Alignment to Teacher (Are they the right paths?)', 'sei_layers', True),
        ('Path Churn (Do gates stabilize when winners emerge?)', 'churn_active_layers', True),
        ('Path Capacity vs Data (Path-wise neural-race SNR)', 'snr_max_layers', True),
        # Additional path metrics (old)
        ('Path Pressure Per Layer', 'path_pressure_layers', True),
        ('Path Entropy Per Layer', 'path_entropy_layers', True),
        ('Path SNR P95 Per Layer', 'snr_p95_layers', True),
        # NEW PATH-CENTRIC METRICS
        ('Path Support Entropy (H_path)', 'H_path', False),
        ('Path Gain Concentration (H_gain)', 'H_gain', False),
        ('Path-to-group Mutual Information (I_layers)', 'I_layers', True),
        ('Confident Path Churn (rho_conf)', 'confident_churn_layers', True),
        ('Path SNR Count Above Threshold', 'path_snr_count_above_threshold', False),
        ('Path SNR: Number of Distinct Paths', 'path_snr_num_paths', False),
        ('Path SNR Threshold', 'path_snr_threshold', False),
        ('Path SNR N_gamma Total', 'path_snr_N_gamma_total', False),
        ('Neural-Race Index (NRI)', 'nri', False),
        # Path SNR component statistics (scalar metrics)
        ('Path SNR: Label Correlation Mean (c_gamma)', 'path_snr_c_gamma_mean', False),
        ('Path SNR: Label Correlation Median (c_gamma)', 'path_snr_c_gamma_median', False),
        ('Path SNR: Label Correlation Std (c_gamma)', 'path_snr_c_gamma_std', False),
        ('Path SNR: Sample Support Mean (N_gamma)', 'path_snr_N_gamma_mean', False),
        ('Path SNR: Sample Support Median (N_gamma)', 'path_snr_N_gamma_median', False),
        ('Path SNR: Sample Support Std (N_gamma)', 'path_snr_N_gamma_std', False),
        ('Path SNR: Signal-to-Noise Mean (SNR_gamma)', 'path_snr_SNR_gamma_mean', False),
        ('Path SNR: Signal-to-Noise Median (SNR_gamma)', 'path_snr_SNR_gamma_median', False),
        ('Path SNR: Signal-to-Noise Std (SNR_gamma)', 'path_snr_SNR_gamma_std', False),
    ]
    
    for algo in algorithms:
        print(f"  Processing {algo}...")
        
        # Plot regular metrics
        for metric_name, metric_key, is_layerwise in metrics_to_plot:
            # Check if this metric exists in the data
            has_metric = False
            for n_train in data[algo].keys():
                for lam in data[algo][n_train].keys():
                    history = data[algo][n_train][lam]['history']
                    if history is not None and metric_key in history.columns:
                        has_metric = True
                        break
                if has_metric:
                    break
            
            if has_metric:
                print(f"    Plotting {metric_name}...")
                plot_metric_grid(data, algo, metric_name, metric_key, output_dir, is_layerwise)
        
        # Plot gate stats separately (it's a dictionary)
        has_gate_stats = False
        for n_train in data[algo].keys():
            for lam in data[algo][n_train].keys():
                history = data[algo][n_train][lam]['history']
                if history is not None and 'gate_stats' in history.columns:
                    has_gate_stats = True
                    break
            if has_gate_stats:
                break
        
        if has_gate_stats:
            print(f"    Plotting Gate Stats...")
            plot_gate_stats_grid(data, algo, output_dir)
        
        # Plot Path SNR components with mean/median and std ribbon
        print(f"    Plotting Path SNR Components (with std ribbon)...")
        plot_path_snr_components(data, algo, output_dir)
    
    # Plot a values histograms
    print("\nCreating a values histogram grids...")
    plot_a_values_histogram_grid(experiment_dir, algorithms, output_dir)
    
    # Run path analysis on final models
    print("\nRunning path analysis on final models...")
    run_path_analysis_on_final_models(experiment_dir, algorithms, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

