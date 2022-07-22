import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from bentoml import env, artifacts, api, BentoService
from bentoml.adapters import DataframeInput, JsonOutput
from bentoml.frameworks.sklearn import SklearnModelArtifact

@env(infer_pip_packages=True)
@artifacts([SklearnModelArtifact('model'), SklearnModelArtifact('encoder')])
class FraudClassifier(BentoService):
    @api(input=DataframeInput(), output=JsonOutput(), batch=True)
    def predict(self, df):
        model = self.artifacts.model
        enc = self.artifacts.encoder

        X = df[["ProductCD", "P_emaildomain", "R_emaildomain", "card4", "M1", "M2", "M3"]]
        X = X.fillna(pd.NA) # ensure all missing values are pandas NA
        X = pd.DataFrame(enc.transform(X).toarray(), columns=enc.get_feature_names_out().reshape(-1))
        X["TransactionAmt"] = df[["TransactionAmt"]].to_numpy()
        return model.predict(X)
