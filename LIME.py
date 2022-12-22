!pip install lime



import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import lime
import lime.lime_tabular
from lime import submodular_pick
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split 



data = load_breast_cancer()  
df = pd.DataFrame(data.data, columns=data.feature_names) 
df['Target'] = data.target
df.head(5)



X = data['data']
Y = data['target']
features = data.feature_names


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


model = XGBClassifier(n_estimators = 300, random_state = 123)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)


accuracy = accuracy_score(Y_test, Y_pred) 
print("Accuracy: %.2f%%" % (accuracy * 100.0)) 



predict_fn = lambda x: model.predict_proba(x)


np.random.seed(123)

# Defining the LIME explainer object
explainer = lime.lime_tabular.LimeTabularExplainer(df[features].astype(int).values,
                                                   mode='classification',
                                                   class_names=['Negative', 'Positive'],
                                                   training_labels=df['Target'],
                                                   feature_names=features)


# using LIME to get the explanations
i = 5
exp = explainer.explain_instance(df.loc[i,features].astype(int).values, predict_fn, num_features=5)
exp.show_in_notebook(show_table=True)



figure = exp.as_pyplot_figure(label = exp.available_labels()[0])

"""

Q. explain how the use of that particular XAI technique helped in making the model more explainable.

LIME provides local model interpretability. LIME modifies a single data sample by tweaking the feature values and observes the resulting impact on the output. The output of LIME is a list of explanations, reflecting the contribution of each feature to the prediction of a data sample. This provides local interpretability, and it also allows to determine which feature changes will have most impact on the prediction.
"""