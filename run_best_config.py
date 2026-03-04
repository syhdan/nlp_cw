"""
Run best hyperparameter configuration from search
Based on trial results with R-Drop
"""
import subprocess
import sys

if __name__ == '__main__':
    print("="*80)
    print("Running Best Hyperparameter Configuration")
    print("="*80)
    print()
    print("Configuration:")
    print("  lr: 3e-05")
    print("  batch_size: 16")
    print("  warmup_ratio: 0.2")
    print("  weight_decay: 0.001")
    print("  freeze_layers: 0")
    print("  use_rdrop: True")
    print("  rdrop_alpha: 0.5")
    print("  epochs: 10 (increased from search's 5)")
    print("="*80)
    print()
    
    cmd = [
        sys.executable,
        'task3_roberta_enhanced.py',
        '--epochs', '10',
        '--batch_size', '16',
        '--lr', '3e-5',
        '--warmup', '0.2',
        '--weight_decay', '0.001',
        '--freeze_layers', '0',
        '--use_rdrop',
        '--rdrop_alpha', '0.5'
    ]
    
    subprocess.run(cmd)
