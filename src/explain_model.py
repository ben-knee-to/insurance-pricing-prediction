"""Step 7: explain a frozen model, using training data for importance and calibration."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.metrics import mean_absolute_error

ROOT=Path(__file__).resolve().parents[1]
FEATURES=['age','sex','bmi','children','smoker','region']

def summarize_segments(frame, pred):
    f=frame.copy(); f['prediction']=pred; f['error']=f['charges']-pred
    f['abs_error']=f.error.abs()
    f['age_band']=pd.cut(f.age,[0,30,45,60,np.inf],labels=['under_30','30_44','45_59','60_plus'])
    f['bmi_band']=pd.cut(f.bmi,[0,18.5,25,30,np.inf],labels=['under_18_5','18_5_24_9','25_29_9','30_plus'])
    rows=[]
    for col in ['smoker','sex','region','age_band','bmi_band']:
        for label,g in f.groupby(col,observed=True):
            rows.append({'dimension':col,'segment':str(label),'n':len(g),'mae_usd':g.abs_error.mean(),'bias_actual_minus_predicted_usd':g.error.mean()})
    return pd.DataFrame(rows)

def run(root=ROOT):
    root=Path(root); out=root/'reports/ml/step7'; figs=out/'figures'; figs.mkdir(parents=True,exist_ok=True)
    train=pd.read_csv(root/'data/processed/ml/train.csv'); test=pd.read_csv(root/'data/processed/ml/test.csv')
    model=joblib.load(root/'models/final_insurance_model.joblib')
    if set(train.source_row_id)&set(test.source_row_id): raise ValueError('Train/test overlap')
    X=train[FEATURES]; y=train.charges
    # This is a diagnostic, not a new model-selection exercise. The final model is not changed.
    imp=permutation_importance(model,X,y,scoring='neg_mean_absolute_error',n_repeats=8,random_state=42,n_jobs=1)
    importance=pd.DataFrame({'feature':FEATURES,'mae_increase_usd':imp.importances_mean,'std_usd':imp.importances_std}).sort_values('mae_increase_usd',ascending=False)
    importance.to_csv(out/'permutation_importance_training.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4)); ax.barh(importance.feature[::-1],importance.mae_increase_usd[::-1]); ax.set_xlabel('Increase in training MAE when permuted (USD)'); fig.tight_layout(); fig.savefig(figs/'permutation_importance.png',dpi=150); plt.close(fig)
    # Use out-of-fold predictions from the prior step, not in-sample residuals, to estimate error quantiles.
    oof=pd.read_csv(root/'reports/ml/step6/training_oof_diagnostics.csv')
    q=float(np.quantile(oof.absolute_error,0.9,method='higher'))
    # Empirical error band, NOT a calibrated prediction interval or guaranteed coverage.
    pred=model.predict(test[FEATURES]); segment=summarize_segments(test,pred)
    segment.to_csv(out/'holdout_segment_diagnostics.csv',index=False)
    case=test[['source_row_id']+FEATURES+['charges']].copy(); case['prediction_usd']=pred
    case['error_usd']=case.charges-pred; case['absolute_error_usd']=case.error_usd.abs()
    case['illustrative_lower_usd']=np.maximum(0,pred-q); case['illustrative_upper_usd']=pred+q
    case.to_csv(out/'holdout_explanations.csv',index=False)
    fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(pred,case.error_usd,s=15,alpha=.6); ax.axhline(0,color='black',ls='--'); ax.set(xlabel='Predicted charges (USD)',ylabel='Actual - predicted (USD)'); fig.tight_layout(); fig.savefig(figs/'residual_vs_prediction.png',dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); top=segment[segment.dimension.isin(['smoker','age_band','bmi_band'])]; ax.barh(top.dimension+': '+top.segment,top.mae_usd); ax.set_xlabel('Holdout MAE (USD)'); fig.tight_layout(); fig.savefig(figs/'segment_errors.png',dpi=150); plt.close(fig)
    # Tree SHAP on the fitted forest, explaining the transformed features. No refitting.
    shap_status='not_available'; shap_file=None
    try:
        import shap
        pre=model.named_steps['features']; forest=model.named_steps['regressor']
        sample=train[FEATURES].sample(n=min(60,len(train)),random_state=42)
        transformed=pre.transform(sample)
        names=list(pre.named_steps['preprocess'].get_feature_names_out())
        explainer=shap.TreeExplainer(forest)
        vals=explainer.shap_values(transformed)
        vals=np.asarray(vals)
        if vals.ndim!=2 or vals.shape[1]!=len(names): raise ValueError('Unexpected SHAP output shape')
        sv=pd.DataFrame({'transformed_feature':names,'mean_absolute_shap_usd':np.abs(vals).mean(axis=0)}).sort_values('mean_absolute_shap_usd',ascending=False)
        sv.to_csv(out/'shap_global_importance.csv',index=False)
        fig,ax=plt.subplots(figsize=(9,5)); top=sv.head(12).iloc[::-1]; ax.barh(top.transformed_feature,top.mean_absolute_shap_usd); ax.set_xlabel('Mean |SHAP value| (USD)'); fig.tight_layout(); fig.savefig(figs/'shap_global_importance.png',dpi=150); plt.close(fig)
        # Additive local breakdown for one training example (no use of test target).
        local=pd.DataFrame({'feature':names,'contribution_usd':vals[0]})
        local.to_csv(out/'shap_local_training_example.csv',index=False)
        shap_status='completed'; shap_file='shap_global_importance.csv'
    except (ImportError,AttributeError,ValueError,TypeError) as exc:
        shap_status=f'unavailable: {type(exc).__name__}: {exc}'
    result={'model_frozen':True,'train_rows':len(train),'holdout_rows':len(test),'training_permutation_importance':True,'shap_status':shap_status,'shap_global_file':shap_file,'oof_absolute_error_90th_percentile_usd':q,'holdout_mae_usd':float(mean_absolute_error(test.charges,pred)),'interval_warning':'Illustrative symmetric band based on OOF absolute errors, not calibrated conditional coverage. Do not use as clinical or underwriting guidance.','holdout_warning':'Holdout already evaluated in Step 6; diagnostics only. Do not select or tune models based on these results.'}
    (out/'explainability_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2)); return result

if __name__=='__main__': run()
