import pandas as pd

def salva_posterior(posterior_list, filename="posterior_bayesiano.csv"):
    df = pd.DataFrame(posterior_list, columns=["x", "theta", "accettazioni"])
    df.to_csv(filename, index=False)
    print(f"✅ Posterior salvato in '{filename}'")
