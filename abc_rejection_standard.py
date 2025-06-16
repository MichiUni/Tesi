import torch
import numpy as np
import sqlite3
import csv
from viModel import StandardEmulator

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
y_obs = torch.tensor([32.6263, 96.7327, 208.6077], dtype=torch.float32).to(device)

# Inizializza modello standard
model = StandardEmulator()
model.load_state_dict(torch.load("modello_standard.pt", map_location=device))
model.to(device)
model.eval()

def init_db(db_file="risultati_standard_abc.db"):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS risultati
                 (x REAL, theta REAL, distanza REAL)''')
    conn.commit()
    conn.close()

def insert_result(x, theta, distanza, db_file="risultati_standard_abc.db"):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("INSERT INTO risultati (x, theta, distanza) VALUES (?, ?, ?)", (x, theta, distanza))
    conn.commit()
    conn.close()

def run_abc_to_db(S=1_000_000, db_file="risultati_standard_abc.db"):
    init_db(db_file)

    for s in range(S):
        x = np.random.exponential(1.5) * 100
        theta = np.random.uniform(0, 5000)
        inputs = torch.tensor([[x, theta]], dtype=torch.float32).to(device)

        with torch.no_grad():
            output = model(inputs)[0]
            dist = torch.norm(output - y_obs).item()

        insert_result(x, theta, dist, db_file=db_file)

        if (s + 1) % 10000 == 0:
            print(f"Simulazione {s+1}/{S} salvata.")

def estrai_top_percentuale(quantile=0.005, db_file="risultati_standard_abc.db", output_file="posterior_standard_top0_5perc.csv"):
    conn = sqlite3.connect(db_file)
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM risultati")
    total = c.fetchone()[0]
    top_n = int(total * quantile)

    c.execute("SELECT x, theta, distanza FROM risultati ORDER BY distanza ASC LIMIT ?", (top_n,))
    top_results = c.fetchall()
    conn.close()

    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'theta', 'distanza'])
        writer.writerows(top_results)

    print(f"Salvati {top_n} risultati in '{output_file}'")

if __name__ == "__main__":
    run_abc_to_db(S=1_000_000, db_file="risultati_standard_abc.db")
    estrai_top_percentuale(quantile=0.005, db_file="risultati_standard_abc.db", output_file="posterior_standard_top0_5perc.csv")
