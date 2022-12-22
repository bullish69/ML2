!pip install shap 



import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_breast_cancer 
import xgboost as xgb 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 




data = load_breast_cancer() 
df = pd.DataFrame(data.data, columns=data.feature_names) 
df.head()



df['target'] = data.target 
df.head()



Y = df['target'].to_frame() 
X = df[df.columns.difference(['target'])] 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42) 



# Build XGBoost model 
xgb_model = xgb.XGBClassifier(random_state=42, gpu_id=0) 
xgb_model.fit(X_train, Y_train.values.ravel()) 
y_pred = xgb_model.predict(X_test) 
accuracy = accuracy_score(y_pred, Y_test) 
print("Accuracy:", accuracy*100)



# SHAP 
import shap 
explainer = shap.TreeExplainer(xgb_model) 
shap_values = explainer.shap_values(X) 
expected_value = explainer.expected_value 



# SUMMARY PLOTS 
shap.summary_plot(shap_values, X, title="SHAP SUMMARY DOT PLOT") 
shap.summary_plot(shap_values, X, plot_type="bar", title="SHAP SUMMARY BAR PLOT") 



# Waterfall plot 
shap.plots._waterfall.waterfall_legacy(expected_value, shap_values[79], features=X.iloc[79, :], feature_names=X.columns, max_display=15, show=True)


# Dependence Plot 
shap.dependence_plot("worst concave points", shap_values, X, interaction_index="mean concave points")


# Multiple Dependence plots 
for i in X_train.columns: 
  shap.dependence_plot(i, shap_values, X)




shap.initjs()
# Force plot - multiple rows 
shap.force_plot(explainer.expected_value, shap_values[:100, :], X.iloc[:100, :]) 
# Force plot - Single row 
shap.force_plot(explainer.expected_value, shap_values[0, :], X.iloc[0, :])



# Decision Plot 
shap.decision_plot(expected_value, shap_values[79], features=X.iloc[79, :], feature_names=(X.columns.tolist()), link="logit", show=True, title="Decision Plot")

"""

Q. explain how the use of that particular XAI technique helped in making the model more explainable.

SHAP is based on Shapley value, a method to calculate the contributions of each feature to the model's prediction. The Shapley value is calculated with all possible combinations of features. Given N players, it has to calculate outcomes for 2^N combinations of features. Calculating the contribution of each feature is not feasible for large numbers of N. For example, for images, N is the number of pixels. Therefore, SHAP does not attempt to calculate the actual Shapley value. Instead, it uses sampling and approximations to calculate the SHAP value.
"""