import yfinance as yf
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# Fonction pour récupérer et afficher les informations sur un ETF
def get_etf_info():
    ticker = entry.get()
    etf = yf.Ticker(ticker)

    # Récupérer le nom complet de l'ETF
    name = etf.info.get('longName', 'Nom non disponible')
    label_name.config(text=f"Nom : {name}")

    # Afficher le graphique de l'ETF
    plot_etf_chart(ticker)


# Fonction pour afficher le graphique de l'ETF
def plot_etf_chart(ticker):
    etf = yf.Ticker(ticker)
    hist = etf.history(period="1y")  # Récupérer les données sur 1 an

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(hist.index, hist['Close'], label=ticker)
    ax.set_title(f"Prix de clôture de {ticker} sur 1 an")
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix de clôture (USD)")
    ax.legend()

    # Intégrer le graphique dans Tkinter
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)


# Configuration de la fenêtre principale
root = tk.Tk()
root.title("Affichage des ETFs")

# Zone d'entrée pour le ticker de l'ETF
label = tk.Label(root, text="Entrez le ticker de l'ETF :")
label.pack(pady=10)

entry = tk.Entry(root)
entry.pack(pady=5)

# Bouton pour récupérer les informations
button = tk.Button(root, text="Afficher les informations", command=get_etf_info)
button.pack(pady=10)

# Label pour afficher le nom de l'ETF
label_name = tk.Label(root, text="")
label_name.pack(pady=5)

# Lancer l'application
root.mainloop()
