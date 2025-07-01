import joblib
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

def load_iris_data():
    """Load iris data with lazy import"""
    from sklearn.datasets import load_iris
    return load_iris()

def get_or_train_model(force_retrain: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """
    Get cached model or train a new one with optimized parameters
    
    Args:
        force_retrain: Force retraining even if cached model exists
        
    Returns:
        Tuple of (model, training_info)
    """
    model_file = Path('cache/iris_model.pkl')
    info_file = Path('cache/iris_model_info.pkl')
    
    # Create cache directory if it doesn't exist
    model_file.parent.mkdir(exist_ok=True)
    
    # Load cached model if available and not forcing retrain
    if not force_retrain and model_file.exists() and info_file.exists():
        try:
            print("Loading cached model...")
            model = joblib.load(model_file)
            info = joblib.load(info_file)
            print(f"Cached model loaded. Training accuracy: {info.get('accuracy', 'N/A'):.4f}")
            return model, info
        except Exception as e:
            print(f"Error loading cached model: {e}")
    
    # Train new model
    print("Training new model...")
    start_time = time.time()
    
    # Lazy imports
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    
    # Load data
    iris = load_iris_data()
    X, y = iris.data, iris.target
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Optimized parameter grid (reduced for faster training)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    
    # Create base model
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Grid search with reduced CV for faster training
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=3,  # Reduced from 5 for faster training
        scoring='accuracy',
        n_jobs=-1,  # Use all available cores
        verbose=1
    )
    
    # Train model
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Evaluate model
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    # Training information
    training_time = time.time() - start_time
    training_info = {
        'accuracy': accuracy,
        'best_params': grid_search.best_params_,
        'training_time': training_time,
        'cv_score': grid_search.best_score_,
        'feature_names': iris.feature_names,
        'target_names': iris.target_names.tolist()
    }
    
    # Save model and info
    try:
        joblib.dump(best_model, model_file)
        joblib.dump(training_info, info_file)
        print(f"Model saved to {model_file}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Print results
    print(f"\nTraining completed in {training_time:.2f} seconds")
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Cross-validation score: {grid_search.best_score_:.4f}")
    print(f"Test accuracy: {accuracy:.4f}")
    
    return best_model, training_info

def predict_iris(features: list, model: Optional[Any] = None) -> Dict[str, Any]:
    """
    Predict iris species with confidence scores
    
    Args:
        features: List of 4 features [sepal_length, sepal_width, petal_length, petal_width]
        model: Optional pre-loaded model
        
    Returns:
        Dictionary with prediction results
    """
    if model is None:
        model, _ = get_or_train_model()
    
    # Load iris data for target names
    iris = load_iris_data()
    
    # Make prediction
    prediction = model.predict([features])[0]
    probabilities = model.predict_proba([features])[0]
    
    return {
        'predicted_class': int(prediction),
        'predicted_species': iris.target_names[prediction],
        'probabilities': {
            iris.target_names[i]: float(prob) 
            for i, prob in enumerate(probabilities)
        },
        'confidence': float(max(probabilities))
    }

def benchmark_model_performance(n_runs: int = 5) -> Dict[str, float]:
    """
    Benchmark model performance
    
    Args:
        n_runs: Number of benchmark runs
        
    Returns:
        Performance metrics
    """
    print(f"Benchmarking model performance over {n_runs} runs...")
    
    model, info = get_or_train_model()
    iris = load_iris_data()
    
    from sklearn.model_selection import cross_val_score
    import numpy as np
    
    # Benchmark prediction time
    sample_features = iris.data[0]
    
    prediction_times = []
    for _ in range(n_runs * 100):  # Many predictions for timing
        start = time.time()
        predict_iris(sample_features.tolist(), model)
        prediction_times.append(time.time() - start)
    
    # Benchmark cross-validation
    cv_times = []
    cv_scores = []
    for _ in range(n_runs):
        start = time.time()
        scores = cross_val_score(model, iris.data, iris.target, cv=3)
        cv_times.append(time.time() - start)
        cv_scores.extend(scores)
    
    return {
        'avg_prediction_time_ms': np.mean(prediction_times) * 1000,
        'avg_cv_time_s': np.mean(cv_times),
        'avg_cv_accuracy': np.mean(cv_scores),
        'std_cv_accuracy': np.std(cv_scores),
        'training_time_s': info.get('training_time', 0),
        'model_accuracy': info.get('accuracy', 0)
    }

def clear_model_cache():
    """Clear cached models"""
    cache_dir = Path('cache')
    model_files = ['iris_model.pkl', 'iris_model_info.pkl']
    
    for file in model_files:
        file_path = cache_dir / file
        if file_path.exists():
            file_path.unlink()
            print(f"Removed {file_path}")

if __name__ == "__main__":
    print("Optimized Iris Classification Demo")
    print("=" * 40)
    
    # Train or load model
    model, info = get_or_train_model()
    
    # Example predictions
    test_samples = [
        [5.1, 3.5, 1.4, 0.2],  # Setosa
        [7.0, 3.2, 4.7, 1.4],  # Versicolor
        [6.3, 3.3, 6.0, 2.5],  # Virginica
    ]
    
    print("\nExample Predictions:")
    print("-" * 20)
    for i, features in enumerate(test_samples, 1):
        result = predict_iris(features, model)
        print(f"Sample {i}: {features}")
        print(f"  Predicted: {result['predicted_species']} (confidence: {result['confidence']:.3f})")
        print(f"  Probabilities: {result['probabilities']}")
        print()
    
    # Benchmark performance
    print("Performance Benchmark:")
    print("-" * 20)
    benchmark_results = benchmark_model_performance()
    for metric, value in benchmark_results.items():
        if 'time' in metric.lower():
            if 'ms' in metric:
                print(f"{metric}: {value:.3f} ms")
            else:
                print(f"{metric}: {value:.3f} s")
        else:
            print(f"{metric}: {value:.4f}")