"""Small deterministic example by default; optional fixed Stage3 network replay."""
from pathlib import Path
import sys,argparse,json,hashlib,datetime
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.analysis.evaluation import evaluate_fixed

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',default='configs/smoke.json');p.add_argument('--out',default='local_outputs/smoke');p.add_argument('--save-raw',action='store_true');a=p.parse_args()
    config=json.loads(Path(a.config).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True);rows=[];candidates=[];failures=[]
    for network in config['networks']:
        for b in range(config['blocks']):
            try:
                row,raw,cs=evaluate_fixed(config,network,b);row['status']='ok';rows.append(row)
                candidates.extend(dict(network=network['name'],block=b,alpha=alpha,**s) for alpha,s in zip(config['ridge_alpha'],cs))
                if a.save_raw:
                    d=out/'raw';d.mkdir(exist_ok=True)
                    for split,arrays in raw.items():np.savez_compressed(d/f'{network["name"]}_b{b}_{split}.npz',**arrays)
            except Exception as ex:failures.append(dict(network=network['name'],block=b,error=repr(ex)))
    pd.DataFrame(rows).to_csv(out/'results.csv',index=False);pd.DataFrame(candidates).to_csv(out/'reader_candidates.csv',index=False)
    pd.DataFrame(failures,columns=['network','block','error']).to_csv(out/'failed_runs.csv',index=False)
    (out/'manifest.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),config=config,config_sha256=hashlib.sha256(Path(a.config).read_bytes()).hexdigest(),result_sha256=hashlib.sha256((out/'results.csv').read_bytes()).hexdigest(),scope='Fixed networks with independent train/validation/outer trajectories; smoke output is a code check, not a new scientific result'),indent=2))
    print(json.dumps(dict(completed=len(rows),failed=len(failures),output=str(out))))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
