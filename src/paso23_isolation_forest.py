"""Detección exploratoria de anomalías estructurales con Isolation Forest."""
from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def assign_group(price: float) -> str:
    if price <= 100000: return "grupo1"
    if price <= 500000: return "grupo2"
    return "grupo3"

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data = pd.read_csv(root / "data" / "housing_train.csv")
    variables = pd.read_csv(root / "outputs" / "tables" / "paso19_variables_clustering.csv")["variable"].tolist()
    if len(variables) != 36 or {"Id", "SalePrice", "PriceGroup"} & set(variables): raise RuntimeError("Variables inválidas")
    X_raw = data[variables]
    imputer = SimpleImputer(strategy="median"); X = imputer.fit_transform(X_raw)
    if X.shape != (1460, 36) or not np.isfinite(X).all(): raise RuntimeError("Matriz imputada inválida")
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1); model.fit(X)
    prediction = model.predict(X); score_samples = model.score_samples(X); decision = model.decision_function(X); anomaly_score = -score_samples
    if set(np.unique(prediction)) != {-1, 1} or not np.isfinite(np.r_[anomaly_score, decision]).all(): raise RuntimeError("Scoring inválido")
    is_anomaly = prediction == -1; n_anomaly = int(is_anomaly.sum()); n_normal = int((~is_anomaly).sum())
    tables_dir, metrics_dir, figures_dir = [root / "outputs" / d for d in ("tables", "metrics", "figures")]
    for directory in (tables_dir, metrics_dir, figures_dir): directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"variable": variables, "dtype": [str(data[c].dtype) for c in variables], "missing_count": [int(data[c].isna().sum()) for c in variables], "preprocessing": "SimpleImputer(strategy='median'); sin escalado en Isolation Forest"}).to_csv(tables_dir / "paso23_variables_isolation_forest.csv", index=False)
    results = pd.DataFrame({"indice_original": np.arange(len(data)), "is_anomaly": is_anomaly, "prediction": prediction, "anomaly_score": anomaly_score, "decision_function": decision, "SalePrice": data.SalePrice})
    results.to_csv(tables_dir / "paso23_anomalias.csv", index=False)
    pd.DataFrame([{ "n_rows":len(data), "n_features":len(variables), "n_estimators":300, "contamination":"auto", "n_normal":n_normal, "n_anomaly":n_anomaly, "pct_anomaly":n_anomaly/len(data)*100, "random_state":42 }]).to_csv(metrics_dir / "paso23_resumen.csv", index=False)
    profiles = []
    for aggregation, filename in (("mean", "paso23_perfil_normal_vs_anomalo_media.csv"), ("median", "paso23_perfil_normal_vs_anomalo_mediana.csv")):
        table = X_raw.assign(status=np.where(is_anomaly, "anomalo", "normal")).groupby("status")[variables].agg(aggregation).T.reset_index().rename(columns={"index":"variable"}); table.to_csv(tables_dir / filename, index=False); profiles.append(table)
    means = X_raw.assign(status=np.where(is_anomaly, "anomalo", "normal")).groupby("status")[variables].mean().T
    stds = X_raw[variables].std().replace(0, np.nan)
    diff = pd.DataFrame({"variable":variables, "normal_mean":means["normal"].to_numpy(), "anomaly_mean":means["anomalo"].to_numpy()}); diff["difference"] = diff.anomaly_mean - diff.normal_mean; diff["relative_difference"] = diff.difference / diff.normal_mean.replace(0, np.nan); diff["abs_standardized_difference"] = diff.difference.abs() / stds.to_numpy(); diff = diff.replace([np.inf, -np.inf], np.nan).fillna(0).sort_values("abs_standardized_difference", ascending=False); diff.head(15).to_csv(tables_dir / "paso23_variables_diferenciadoras.csv", index=False)
    sale = data.assign(status=np.where(is_anomaly, "anomalo", "normal")).groupby("status").SalePrice.agg(count="count", mean="mean", median="median", min="min", max="max", Q1=lambda x:x.quantile(.25), Q3=lambda x:x.quantile(.75)).reset_index(); sale.to_csv(tables_dir / "paso23_saleprice_normal_vs_anomalo.csv", index=False)
    top_vars = diff.head(8).variable.tolist(); top = results.sort_values("anomaly_score", ascending=False).head(20).merge(data[top_vars].reset_index().rename(columns={"index":"indice_original"}), on="indice_original"); top.to_csv(tables_dir / "paso23_top_anomalias.csv", index=False)
    data["PriceGroup"] = data.SalePrice.map(assign_group); status_series = pd.Series(np.where(is_anomaly, "anomalo", "normal"), name="status"); counts = pd.crosstab(status_series, data.PriceGroup).reindex(index=["normal","anomalo"], columns=["grupo1","grupo2","grupo3"], fill_value=0); cross = counts.reset_index().melt(id_vars="status", var_name="price_group", value_name="n"); cross["pct_dentro_status"] = cross.apply(lambda r: r.n / counts.loc[r.status].sum() * 100, axis=1); cross.to_csv(tables_dir / "paso23_anomalias_vs_pricegroup.csv", index=False)
    plt.figure(figsize=(8,5)); plt.hist(anomaly_score, bins=30); plt.xlabel("Anomaly score (-score_samples)"); plt.ylabel("Frecuencia"); plt.title("Distribución del anomaly score"); plt.tight_layout(); plt.savefig(figures_dir / "paso23_distribucion_anomaly_score.png", dpi=150); plt.close()
    plt.figure(figsize=(7,5)); plt.boxplot([data.loc[~is_anomaly, "SalePrice"], data.loc[is_anomaly, "SalePrice"]], tick_labels=["normal","anómalo"]); plt.ylabel("SalePrice"); plt.title("SalePrice: normal vs anómalo"); plt.tight_layout(); plt.savefig(figures_dir / "paso23_saleprice_normal_vs_anomalo.png", dpi=150); plt.close()
    top_diff = diff.head(10).sort_values("abs_standardized_difference"); plt.figure(figsize=(9,6)); plt.barh(top_diff.variable, top_diff.abs_standardized_difference); plt.xlabel("Diferencia estandarizada absoluta"); plt.ylabel("Variable"); plt.title("Variables diferenciadoras"); plt.tight_layout(); plt.savefig(figures_dir / "paso23_variables_diferenciadoras.png", dpi=150); plt.close()
    pca = PCA(n_components=2); points = pca.fit_transform(StandardScaler().fit_transform(X)); plt.figure(figsize=(8,6)); plt.scatter(points[~is_anomaly,0], points[~is_anomaly,1], label="normal"); plt.scatter(points[is_anomaly,0], points[is_anomaly,1], label="anómalo"); plt.xlabel("PC1"); plt.ylabel("PC2"); plt.title("Anomalías proyectadas en PCA"); plt.legend(); plt.tight_layout(); plt.savefig(figures_dir / "paso23_anomalias_pca.png", dpi=150); plt.close()
    print(f"Anomalías: {n_anomaly}/{len(data)} ({n_anomaly/len(data)*100:.2f}%); normales: {n_normal}"); print(diff.head(10).to_string(index=False)); print(sale.to_string(index=False)); print(top[["indice_original","SalePrice","anomaly_score","decision_function"]].head(5).to_string(index=False)); print(cross.to_string(index=False))

if __name__ == "__main__": main()
