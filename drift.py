import h2o
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import pandas as pd
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection, RegressionPerformanceProfileSection

# ----------------------------
# CONFIG
# ----------------------------
features = ["FG_pct","3P_pct","2P_pct","eFG_pct","FT_pct"]
target = "PTS"

# Data paths
reference_csv = "train_shooting.csv"   # older data
production_csv = "new_shooting.csv"    # newer data

# ----------------------------
# LOAD DATA
# ----------------------------
reference_df = pd.read_csv(reference_csv)
production_df = pd.read_csv(production_csv)

reference_df = reference_df.rename(columns=lambda x: x.replace('%','_pct'))
production_df = production_df.rename(columns=lambda x: x.replace('%','_pct'))

# ----------------------------
# H2O INIT AND TRAIN MODEL
# ----------------------------
h2o.init(max_mem_size="4G")
train_hf = h2o.H2OFrame(reference_df)
prod_hf  = h2o.H2OFrame(production_df)

model = H2OGeneralizedLinearEstimator(
    lambda_search=True,
    max_iterations=50,
    alpha=0.5
)

model.train(x=features, y=target, training_frame=train_hf)

# ----------------------------
# PREDICTIONS ON NEW DATA
# ----------------------------
production_df["prediction"] = model.predict(prod_hf).as_data_frame().values.flatten()

# ----------------------------
# DRIFT AND PERFORMANCE ANALYSIS
# ----------------------------
profile = Profile(sections=[DataDriftProfileSection(), RegressionPerformanceProfileSection()])
profile.calculate(reference_df, production_df, column_mapping={"target": target, "prediction": "prediction"})
report = profile.get_metrics()

# ----------------------------
# SIMPLE SCREEN/TABLE SUMMARY
# ----------------------------
print("\n=== DATA DRIFT SUMMARY ===")
for feature, drift in report['data_drift']['metrics_by_feature'].items():
    print(f"{feature}: Drift={drift['drift_score']:.2f}, Num Bins={drift['n_bins']}")

print("\n=== PERFORMANCE SUMMARY ===")
perf_metrics = report['regression_performance']['metrics']
for metric, value in perf_metrics.items():
    print(f"{metric}: {value:.3f}")

# ----------------------------
# SHUTDOWN H2O
# ----------------------------
h2o.cluster().shutdown(prompt=False)
