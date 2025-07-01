import tkinter as tk
from tkinter import ttk
import functools
import pickle
from pathlib import Path
from datetime import datetime, timedelta

# Global cache for API responses
_api_cache = {}
_cache_expiry = {}

def get_cached_etf_data(ticker, period="1y"):
    """Get ETF data with caching to avoid repeated API calls"""
    cache_key = f"{ticker}_{period}"
    current_time = datetime.now()
    
    # Check if we have cached data that's still valid (1 hour expiry)
    if (cache_key in _api_cache and 
        cache_key in _cache_expiry and 
        current_time < _cache_expiry[cache_key]):
        return _api_cache[cache_key]
    
    # Lazy import only when needed
    import yfinance as yf
    
    try:
        etf = yf.Ticker(ticker)
        hist = etf.history(period=period)
        
        # Cache the data with expiry time
        _api_cache[cache_key] = hist
        _cache_expiry[cache_key] = current_time + timedelta(hours=1)
        
        return hist
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return None

def get_etf_info():
    """Optimized ETF info retrieval with caching and error handling"""
    ticker = entry.get().upper().strip()
    
    if not ticker:
        label_name.config(text="Please enter a ticker symbol")
        return
    
    # Lazy import
    import yfinance as yf
    
    try:
        etf = yf.Ticker(ticker)
        info = etf.info
        
        # Get name with fallback
        name = info.get('longName', info.get('shortName', 'Name not available'))
        label_name.config(text=f"Name: {name}")
        
        # Display chart
        plot_etf_chart(ticker)
        
    except Exception as e:
        label_name.config(text=f"Error: Could not fetch data for {ticker}")
        print(f"Error: {e}")

def downsample_data(data, max_points=500):
    """Downsample data for better plotting performance"""
    if len(data) <= max_points:
        return data
    
    step = max(1, len(data) // max_points)
    return data[::step]

def plot_etf_chart(ticker):
    """Optimized chart plotting with downsampling and caching"""
    hist = get_cached_etf_data(ticker)
    
    if hist is None or hist.empty:
        return
    
    # Lazy imports
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    
    # Clear previous plot
    for widget in root.winfo_children():
        if isinstance(widget, tk.Widget) and hasattr(widget, 'get_tk_widget'):
            widget.destroy()
    
    # Downsample data for better performance
    hist_downsampled = downsample_data(hist)
    
    # Create optimized plot
    plt.style.use('default')  # Use default style for faster rendering
    fig, ax = plt.subplots(figsize=(8, 4), dpi=80)  # Lower DPI for faster rendering
    
    ax.plot(hist_downsampled.index, hist_downsampled['Close'], 
            label=ticker, linewidth=1.5, alpha=0.8)
    ax.set_title(f"Prix de clôture de {ticker} sur 1 an", fontsize=12)
    ax.set_xlabel("Date", fontsize=10)
    ax.set_ylabel("Prix de clôture (USD)", fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Optimize layout
    fig.tight_layout()
    
    # Integrate plot into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)

# Performance monitoring decorator
def monitor_performance(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.3f} seconds")
        return result
    return wrapper

# Apply performance monitoring to key functions
get_etf_info = monitor_performance(get_etf_info)
plot_etf_chart = monitor_performance(plot_etf_chart)

# Configuration de la fenêtre principale
root = tk.Tk()
root.title("Affichage des ETFs (Optimized)")
root.geometry("900x700")

# Create main frame
main_frame = tk.Frame(root)
main_frame.pack(pady=10, padx=10, fill='both', expand=True)

# Zone d'entrée pour le ticker de l'ETF
label = tk.Label(main_frame, text="Entrez le ticker de l'ETF :", font=('Arial', 12))
label.pack(pady=10)

entry = tk.Entry(main_frame, font=('Arial', 12), width=20)
entry.pack(pady=5)

# Bind Enter key to function
entry.bind('<Return>', lambda event: get_etf_info())

# Bouton pour récupérer les informations
button = tk.Button(main_frame, text="Afficher les informations", 
                   command=get_etf_info, font=('Arial', 12))
button.pack(pady=10)

# Label pour afficher le nom de l'ETF
label_name = tk.Label(main_frame, text="", font=('Arial', 11), wraplength=800)
label_name.pack(pady=5)

# Status label
status_label = tk.Label(main_frame, text="Ready", font=('Arial', 10), fg='green')
status_label.pack(pady=5)

# Lancer l'application
if __name__ == "__main__":
    print("Starting optimized ETF application...")
    root.mainloop()