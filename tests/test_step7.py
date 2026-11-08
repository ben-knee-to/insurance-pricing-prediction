import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from explain_model import summarize_segments
import pandas as pd

def test_segments():
    frame=pd.DataFrame({'age':[20,40],'bmi':[22,32],'sex':['male','female'],'smoker':['no','yes'],'region':['northwest','southeast'],'charges':[100,200]})
    result=summarize_segments(frame,[110,190])
    assert set(result.dimension)=={'smoker','sex','region','age_band','bmi_band'}
    assert (result.mae_usd==10).all()
