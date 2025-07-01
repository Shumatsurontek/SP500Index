#!/usr/bin/env python3
"""
Performance Benchmark Script
Measures and compares performance of original vs optimized code
"""

import time
import sys
import gc
import psutil
import os
from pathlib import Path
from typing import Dict, Any, Callable

def measure_memory_usage() -> float:
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

def benchmark_function(func: Callable, name: str, *args, **kwargs) -> Dict[str, Any]:
    """
    Benchmark a function and return performance metrics
    
    Args:
        func: Function to benchmark
        name: Human-readable name for the function
        *args, **kwargs: Arguments to pass to the function
        
    Returns:
        Dictionary with performance metrics
    """
    print(f"\nBenchmarking: {name}")
    print("-" * 50)
    
    # Clear memory before benchmark
    gc.collect()
    
    # Measure initial memory
    memory_before = measure_memory_usage()
    
    # Measure execution time
    start_time = time.time()
    try:
        result = func(*args, **kwargs)
        success = True
        error = None
    except Exception as e:
        result = None
        success = False
        error = str(e)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Measure final memory
    memory_after = measure_memory_usage()
    memory_used = memory_after - memory_before
    
    metrics = {
        'name': name,
        'execution_time': execution_time,
        'memory_before_mb': memory_before,
        'memory_after_mb': memory_after,
        'memory_used_mb': memory_used,
        'success': success,
        'error': error,
        'result_size': len(str(result)) if result is not None else 0
    }
    
    # Print results
    print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Memory before: {memory_before:.2f} MB")
    print(f"Memory after: {memory_after:.2f} MB")
    print(f"Memory used: {memory_used:.2f} MB")
    print(f"Success: {success}")
    if error:
        print(f"Error: {error}")
    
    return metrics

def benchmark_data_loading():
    """Benchmark data loading performance"""
    print("\n" + "="*60)
    print("DATA LOADING BENCHMARKS")
    print("="*60)
    
    results = []
    
    # Test original data loading (if available)
    try:
        def load_original_data():
            import pandas as pd
            data = pd.read_csv('S&P500/sp500_index.csv')
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            data['SMA_50'] = data['Index'].rolling(window=50).mean()
            data['SMA_200'] = data['Index'].rolling(window=200).mean()
            return data
        
        result = benchmark_function(load_original_data, "Original Data Loading")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark original data loading: {e}")
    
    # Test optimized data loading
    try:
        def load_optimized_data():
            from optimized_data_manager import load_optimized_sp500_data
            return load_optimized_sp500_data()
        
        result = benchmark_function(load_optimized_data, "Optimized Data Loading")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark optimized data loading: {e}")
    
    # Test repeated loading (cache effectiveness)
    try:
        def load_optimized_data_repeated():
            from optimized_data_manager import load_optimized_sp500_data
            data1 = load_optimized_sp500_data()
            data2 = load_optimized_sp500_data()  # Should hit cache
            data3 = load_optimized_sp500_data()  # Should hit cache
            return data1
        
        result = benchmark_function(load_optimized_data_repeated, "Optimized Data Loading (3x - Cache Test)")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark repeated data loading: {e}")
    
    return results

def benchmark_ml_performance():
    """Benchmark machine learning performance"""
    print("\n" + "="*60)
    print("MACHINE LEARNING BENCHMARKS")
    print("="*60)
    
    results = []
    
    # Test original ML approach
    try:
        def train_original_model():
            from sklearn.datasets import load_iris
            from sklearn.model_selection import train_test_split, GridSearchCV
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score
            
            iris = load_iris()
            X, y = iris.data, iris.target
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            
            # Original extensive grid search
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            }
            
            model = RandomForestClassifier(random_state=42)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            
            return grid_search.best_estimator_
        
        result = benchmark_function(train_original_model, "Original ML Training")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark original ML training: {e}")
    
    # Test optimized ML approach
    try:
        def train_optimized_model():
            from optimized_iris_classification import get_or_train_model
            model, info = get_or_train_model(force_retrain=True)
            return model
        
        result = benchmark_function(train_optimized_model, "Optimized ML Training")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark optimized ML training: {e}")
    
    # Test cached model loading
    try:
        def load_cached_model():
            from optimized_iris_classification import get_or_train_model
            model, info = get_or_train_model(force_retrain=False)  # Should use cache
            return model
        
        result = benchmark_function(load_cached_model, "Cached Model Loading")
        results.append(result)
    except Exception as e:
        print(f"Could not benchmark cached model loading: {e}")
    
    return results

def benchmark_import_times():
    """Benchmark import performance"""
    print("\n" + "="*60)
    print("IMPORT BENCHMARKS")
    print("="*60)
    
    results = []
    
    # Test heavy imports
    import importlib
    import sys
    
    heavy_imports = [
        ('pandas', 'import pandas as pd'),
        ('matplotlib.pyplot', 'import matplotlib.pyplot as plt'),
        ('sklearn', 'import sklearn'),
        ('yfinance', 'import yfinance as yf'),
        ('statsmodels', 'import statsmodels'),
    ]
    
    for module_name, import_statement in heavy_imports:
        # Remove module if already imported
        modules_to_remove = [name for name in sys.modules if name.startswith(module_name)]
        for mod in modules_to_remove:
            if mod in sys.modules:
                del sys.modules[mod]
        
        def import_module():
            exec(import_statement)
            return True
        
        try:
            result = benchmark_function(import_module, f"Import {module_name}")
            results.append(result)
        except Exception as e:
            print(f"Could not benchmark import of {module_name}: {e}")
    
    return results

def generate_performance_report(all_results: Dict[str, list]):
    """Generate a comprehensive performance report"""
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY REPORT")
    print("="*80)
    
    for category, results in all_results.items():
        if not results:
            continue
            
        print(f"\n{category.upper()}:")
        print("-" * 40)
        
        for result in results:
            print(f"  {result['name']}:")
            print(f"    Time: {result['execution_time']:.3f}s")
            print(f"    Memory: {result['memory_used_mb']:+.2f}MB")
            print(f"    Success: {result['success']}")
            if result['error']:
                print(f"    Error: {result['error']}")
    
    # Calculate improvements
    print(f"\nOPTIMIZATION ANALYSIS:")
    print("-" * 40)
    
    # Data loading improvements
    data_results = all_results.get('data_loading', [])
    if len(data_results) >= 2:
        original = next((r for r in data_results if 'Original' in r['name']), None)
        optimized = next((r for r in data_results if 'Optimized' in r['name'] and 'Cache' not in r['name']), None)
        
        if original and optimized and original['success'] and optimized['success']:
            time_improvement = ((original['execution_time'] - optimized['execution_time']) / original['execution_time']) * 100
            memory_improvement = original['memory_used_mb'] - optimized['memory_used_mb']
            
            print(f"  Data Loading:")
            print(f"    Time improvement: {time_improvement:.1f}%")
            print(f"    Memory improvement: {memory_improvement:+.2f}MB")
    
    # ML improvements
    ml_results = all_results.get('machine_learning', [])
    if len(ml_results) >= 2:
        original = next((r for r in ml_results if 'Original' in r['name']), None)
        optimized = next((r for r in ml_results if 'Optimized' in r['name']), None)
        
        if original and optimized and original['success'] and optimized['success']:
            time_improvement = ((original['execution_time'] - optimized['execution_time']) / original['execution_time']) * 100
            memory_improvement = original['memory_used_mb'] - optimized['memory_used_mb']
            
            print(f"  ML Training:")
            print(f"    Time improvement: {time_improvement:.1f}%")
            print(f"    Memory improvement: {memory_improvement:+.2f}MB")

def main():
    """Run all benchmarks"""
    print("Performance Benchmark Suite")
    print("="*80)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print(f"Available memory: {psutil.virtual_memory().total / 1024 / 1024 / 1024:.1f} GB")
    print(f"CPU cores: {psutil.cpu_count()}")
    
    all_results = {}
    
    # Run benchmarks
    try:
        all_results['data_loading'] = benchmark_data_loading()
    except Exception as e:
        print(f"Error in data loading benchmarks: {e}")
        all_results['data_loading'] = []
    
    try:
        all_results['machine_learning'] = benchmark_ml_performance()
    except Exception as e:
        print(f"Error in ML benchmarks: {e}")
        all_results['machine_learning'] = []
    
    try:
        all_results['imports'] = benchmark_import_times()
    except Exception as e:
        print(f"Error in import benchmarks: {e}")
        all_results['imports'] = []
    
    # Generate report
    generate_performance_report(all_results)
    
    print(f"\nBenchmark completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()