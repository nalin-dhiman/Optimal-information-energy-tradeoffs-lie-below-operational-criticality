import pandas as pd
from pathlib import Path

BASE_DIR = Path()
IN_CSV = BASE_DIR "plateau_boundary_audit.csv"
OUT_CSV = BASE_DIR  "plateau_width_classification.csv"

def classify_plateaus():
    if not IN_CSV.exists():
        print(f"Error: {IN_CSV} not found.")
        return
        
    df = pd.read_csv(IN_CSV)
    
    classifications = []
    
    for _, row in df.iterrows():
        lower = row['touches_lower_boundary']
        upper = row['touches_upper_boundary']
        
        if lower and upper:
            cls = "not suitable for quantitative width claim"
            reason = "Plateau spans entire evaluated grid; true width unknown."
        elif lower or upper:
            cls = "lower bound"
            reason = "Plateau touches one grid boundary; true width is strictly larger."
        else:
            cls = "trusted"
            reason = "Plateau fully contained within evaluated grid."
            
        classifications.append({
            'N': int(row['N']),
            'beta_E': row['beta_E'],
            'reported_width': row['width_reported'],
            'classification': cls,
            'reason': reason
        })
        
    out_df = pd.DataFrame(classifications)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")

if __name__ == "__main__":
    classify_plateaus()
