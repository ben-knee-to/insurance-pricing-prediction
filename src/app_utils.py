"""Reusable, UI-independent functions for the insurance Streamlit dashboard."""
import math
import numpy as np
import pandas as pd
from prepare_features import FEATURES

CATEGORIES={"sex":{"male","female"},"smoker":{"yes","no"},"region":{"northeast","northwest","southeast","southwest"}}

def validate_customer(row):
    if set(FEATURES)-set(row):
        raise ValueError(f"Missing fields: {sorted(set(FEATURES)-set(row))}")
    for key, allowed in CATEGORIES.items():
        if row[key] not in allowed: raise ValueError(f"Invalid {key}")
    for key, lo, hi in [("age",18,64),("bmi",15,55),("children",0,5)]:
        v=row[key]
        if not isinstance(v,(int,float,np.integer,np.floating)) or not math.isfinite(v) or not lo<=v<=hi:
            raise ValueError(f"{key} must be between {lo} and {hi}")
    if int(row['age'])!=row['age'] or int(row['children'])!=row['children']:
        raise ValueError("Age and children must be integers")
    return pd.DataFrame([{k:row[k] for k in FEATURES}])

def predict_charge(model,row):
    result=float(model.predict(validate_customer(row))[0])
    if not math.isfinite(result): raise ValueError("Non-finite model prediction")
    return max(0.0,result)

def get_concentration(charges):
    x=np.sort(np.asarray(charges,dtype=float))
    if not len(x) or not np.all(np.isfinite(x)) or np.any(x<0) or x.sum()<=0:
        raise ValueError("Charges must be finite, nonnegative and have a positive sum")
    n=len(x); total=x.sum()
    cumulative=np.r_[0,np.cumsum(x)/total]
    population=np.linspace(0,1,n+1)
    gini=float((n+1-2*np.cumsum(x).sum()/total)/n)
    return {"top_10_share":float(x[-math.ceil(n*.1):].sum()/total),"top_20_share":float(x[-math.ceil(n*.2):].sum()/total),"gini":gini,"population":population,"cumulative":cumulative}
