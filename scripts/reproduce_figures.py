"""Regenerate a compact validation table and figure from immutable block scores."""
from pathlib import Path
import sys,argparse,json,hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',default='local_outputs/summary');a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];data=root/'results_summary/outer_test_blocks.csv';d=pd.read_csv(data,float_precision='round_trip')
    rows=[]
    for (J,no),g in d.groupby(['J','no_recurrence']):
        rows.append(dict(J=J,no_recurrence=no,blocks=len(g),normalized_MSE=g.normalized_MSE.mean(),R2=g.R2.mean(),mean_rate_Hz=g.mean_rate_Hz.mean()))
    table=pd.DataFrame(rows);table.to_csv(out/'outer_test_summary.csv',index=False)
    rec=table[~table.no_recurrence];zero=table[table.no_recurrence].iloc[0]
    fig,ax=plt.subplots(figsize=(6.2,3.7));ax.plot(rec.J,rec.normalized_MSE,'o-',ms=3,label='Parameters retuned at each J');ax.axhline(zero.normalized_MSE,color='black',ls='--',label='True no recurrence (J=0, theta_A=0)')
    ax.set(xlabel='J (model voltage / per-neuron Hz)',ylabel='Outer normalized MSE');ax.legend(fontsize=8);fig.tight_layout()
    for ext in ['pdf','svg']:fig.savefig(out/f'outer_summary_validation.{ext}')
    (out/'manifest.json').write_text(json.dumps(dict(source='results_summary/outer_test_blocks.csv',source_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),figure='Compact code-validation summary; measured scores, no shifted/scaled curves'),indent=2))
    print(json.dumps(dict(rows=len(table),source_rows=len(d),out=str(out))))
if __name__=='__main__':main()
