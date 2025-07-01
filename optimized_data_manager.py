import pandas as pd
import functools
import pickle
from pathlib import Path
from typing import Optional, Dict, Any
import time

class OptimizedDataManager:
    """Optimized data manager with caching and memory optimization"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self._memory_cache: Dict[str, Any] = {}
    
    def _get_cache_path(self, filename: str) -> Path:
        """Get cache file path for a given filename"""
        cache_name = Path(filename).stem + "_optimized.pkl"
        return self.cache_dir / cache_name
    
    def _optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize DataFrame memory usage by downcasting numeric types"""
        for col in df.columns:
            if df[col].dtype == 'float64':
                df[col] = pd.to_numeric(df[col], downcast='float')
            elif df[col].dtype == 'int64':
                df[col] = pd.to_numeric(df[col], downcast='integer')
        return df
    
    @functools.lru_cache(maxsize=32)
    def load_sp500_data(self, csv_path: str = "S&P500/sp500_index.csv") -> Optional[pd.DataFrame]:
        """
        Load S&P 500 data with caching and optimization
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            Optimized DataFrame or None if error
        """
        cache_path = self._get_cache_path(csv_path)
        csv_file = Path(csv_path)
        
        # Check if cache exists and is newer than source file
        if (cache_path.exists() and 
            csv_file.exists() and 
            cache_path.stat().st_mtime > csv_file.stat().st_mtime):
            
            try:
                print(f"Loading cached data from {cache_path}")
                return pd.read_pickle(cache_path)
            except Exception as e:
                print(f"Error loading cache: {e}")
        
        # Load and process original data
        if not csv_file.exists():
            print(f"CSV file not found: {csv_path}")
            return None
        
        try:
            print(f"Loading and processing data from {csv_path}")
            start_time = time.time()
            
            # Load data with optimized dtypes
            data = pd.read_csv(csv_path, dtype={'Index': 'float32'})
            
            # Convert date column
            data['Date'] = pd.to_datetime(data['Date'])
            data.set_index('Date', inplace=True)
            
            # Sort by date
            data.sort_index(inplace=True)
            
            # Optimize memory usage
            data = self._optimize_dataframe(data)
            
            # Add computed features
            data['SMA_50'] = data['Index'].rolling(window=50, min_periods=1).mean().astype('float32')
            data['SMA_200'] = data['Index'].rolling(window=200, min_periods=1).mean().astype('float32')
            
            # Cache the processed data
            data.to_pickle(cache_path)
            
            load_time = time.time() - start_time
            print(f"Data loaded and processed in {load_time:.3f} seconds")
            print(f"Memory usage: {data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            
            return data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def get_data_subset(self, csv_path: str, start_date: str = None, 
                       end_date: str = None, columns: list = None) -> Optional[pd.DataFrame]:
        """
        Get a subset of data for better performance
        
        Args:
            csv_path: Path to the CSV file
            start_date: Start date for filtering
            end_date: End date for filtering
            columns: List of columns to include
            
        Returns:
            Filtered DataFrame
        """
        data = self.load_sp500_data(csv_path)
        
        if data is None:
            return None
        
        # Apply date filtering
        if start_date:
            data = data[data.index >= start_date]
        if end_date:
            data = data[data.index <= end_date]
        
        # Apply column filtering
        if columns:
            available_columns = [col for col in columns if col in data.columns]
            data = data[available_columns]
        
        return data
    
    def clear_cache(self):
        """Clear all cached data"""
        try:
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
            self._memory_cache.clear()
            self.load_sp500_data.cache_clear()
            print("Cache cleared successfully")
        except Exception as e:
            print(f"Error clearing cache: {e}")
    
    def get_cache_info(self):
        """Get information about cached data"""
        cache_files = list(self.cache_dir.glob("*.pkl"))
        total_size = sum(f.stat().st_size for f in cache_files)
        
        print(f"Cache directory: {self.cache_dir}")
        print(f"Cached files: {len(cache_files)}")
        print(f"Total cache size: {total_size / 1024 / 1024:.2f} MB")
        print(f"LRU cache info: {self.load_sp500_data.cache_info()}")

# Global instance for easy access
data_manager = OptimizedDataManager()

# Convenience functions for backward compatibility
def load_optimized_sp500_data(csv_path: str = "S&P500/sp500_index.csv") -> Optional[pd.DataFrame]:
    """Load optimized S&P 500 data"""
    return data_manager.load_sp500_data(csv_path)

def get_sp500_subset(start_date: str = None, end_date: str = None, 
                    columns: list = None) -> Optional[pd.DataFrame]:
    """Get S&P 500 data subset"""
    return data_manager.get_data_subset("S&P500/sp500_index.csv", start_date, end_date, columns)

if __name__ == "__main__":
    # Test the optimized data manager
    print("Testing Optimized Data Manager...")
    
    # Load data
    data = load_optimized_sp500_data()
    if data is not None:
        print(f"Loaded {len(data)} rows of data")
        print(f"Date range: {data.index.min()} to {data.index.max()}")
        print(f"Columns: {list(data.columns)}")
        
        # Test subset functionality
        recent_data = get_sp500_subset(start_date="2020-01-01", columns=['Index', 'SMA_50'])
        if recent_data is not None:
            print(f"Recent data subset: {len(recent_data)} rows")
    
    # Show cache info
    data_manager.get_cache_info()