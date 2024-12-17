from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import joblib

def model_dovelopment(data):
    X = data.drop(columns=['readmitted'])
    y = data['readmitted'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Model training columns:", X_train.columns)

    log = LogisticRegression(max_iter=500, random_state=42)
    log.fit(X_train, y_train)
    log_pred = log.predict(X_test)
    print("---------------------------------------")
    print(f"Logistic Regression accuracy_score: {accuracy_score(y_test, log_pred)}")
    print("Logistic Regression Classification Report:\n", classification_report(y_test, log_pred))

    xgb = XGBClassifier(n_estimators=100, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)
    print("---------------------------------------")
    print(f"XGB accuracy_score: {accuracy_score(y_test, xgb_pred)}")
    print("XGB Classification Report:\n", classification_report(y_test, xgb_pred))

    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)        
    rf_pred = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)
    print("---------------------------------------")
    print(f"Random Forest accuracy_score: {rf_accuracy}")
    print("Random Forest Classification Report:\n", classification_report(y_test, rf_pred))
    joblib.dump(rf, r"D:\FREELANCE_PROJECTS\diabetes-client-readmit-prediction\models\rf__model.joblib")
    return rf_accuracy


    """
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'class_weight': ['balanced', None]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_res, y_res)
    print("Best Parameters from Grid Search:", grid_search.best_params_)
    
    best_rf = grid_search.best_estimator_
    joblib.dump(best_rf, r"D:\FREELANCE_PROJECTS\diabetes-client-readmit-prediction\models\rff_model.joblib")

    return rf_accuracy
    """
