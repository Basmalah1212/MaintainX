import joblib
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    r2_score
)

from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              confusion_matrix, classification_report)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.ensemble import ( RandomForestClassifier, ExtraTreesClassifier,GradientBoostingClassifier,AdaBoostClassifier)
from sklearn.metrics import ( accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

req_cols = ['ambient_temperature_c', 'ambient_humidity_pct',
       'dsc_heat_flow_mw', 'sample_temperature_c', 'reference_temperature_c',
       'thermal_gradient_c', 'heating_rate_c_min', 'cooling_rate_c_min',
       'infrared_mean_temp_c', 'infrared_max_temp_c',
       'thermal_hotspot_area_pct', 'phase_transition_onset_c',
       'phase_transition_peak_c', 'transition_enthalpy_j_g',
       'heat_dissipation_rate_w', 'sensor_voltage_v', 'sensor_current_ma',
       'vibration_level_mm_s', 'mqtt_latency_ms', 'packet_loss_pct',
       'edge_cpu_usage_pct', 'rolling_noise_index']

def load_data():
    df = pd.read_csv("C:\\Users\\HPP\\Documents\\NTI\\industrial_calorimetric_predictive_maintenance_dataset.csv")
    return df

def validate_data(df):
    missing = []
    for col in req_cols:
        if col not in df.columns:
            missing.append(col)
    if missing:
        print("The data set is missing these req col: ")
        for col in missing:
            print(col)
        return "invalid"
    print("Dataset validation: OK")
    return "valid"

def Exploring(df):
    print(df.info())
    print(df.columns)
    print(df.describe())
    for col in df.select_dtypes(include=['object']).columns:
        print(col)
        print(df[col].value_counts())
        print('-'*40  )

def cleaning(df):
    df = df.drop_duplicates()
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].median())
    return df   

def drop_cols(df, keep_cols=None):
    if keep_cols is None:
        keep_cols = set(req_cols) | {
            'anomaly_score', 'fault_state', 'maintenance_required',
            'remaining_useful_life_h'}
    cols_to_drop = [c for c in df.columns if c not in keep_cols]
    return df.drop(columns=cols_to_drop)

def split(x,y):
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)
    return x_train, x_test, y_train, y_test

def remove_majority_outliers(x, y):
    majority_label = y.value_counts().idxmax()
    majority_rows = y == majority_label
    numeric_columns = x.select_dtypes(include=['number']).columns
    majority_outliers = pd.Series(False, index=x.index)

    for column in numeric_columns:
        values = x.loc[majority_rows, column]
        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        if interquartile_range == 0:
            continue
        lower_bound = first_quartile - 1.5 * interquartile_range
        upper_bound = third_quartile + 1.5 * interquartile_range
        majority_outliers |= majority_rows & (
            (x[column] < lower_bound) | (x[column] > upper_bound)
        )

    keep_rows = ~majority_outliers
    removed_count = int(majority_outliers.sum())
    print(f"Removed {removed_count} outlier(s) from majority class {majority_label}")
    return x.loc[keep_rows], y.loc[keep_rows]
    
def evaluate_model(name, y_true, y_pred):
    return {
        'Model': name,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision (macro)': precision_score(y_true, y_pred, average='macro'),
        'Recall (macro)': recall_score(y_true, y_pred, average='macro'),
        'F1-score (macro)': f1_score(y_true, y_pred, average='macro')
    }
def preprocess(x):
    num_cols = x.select_dtypes(include=['number']).columns.tolist()
    return ColumnTransformer([
        ('scaler', StandardScaler(), num_cols),
    ])

def evaluate_numeric(name, y_true, y_pred):
    return {
            'Model': name,
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
        }
    
def maintenance_model(df):
    x = df.drop([
        'maintenance_required',
        'remaining_useful_life_h',
        'anomaly_score',
        'fault_state'
    ], axis=1)

    y = df['maintenance_required']
    x, y = remove_majority_outliers(x, y)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = preprocess(x)

    xgb_pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', XGBClassifier(
            n_estimators=80,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            reg_alpha=0.5,
            random_state=42,
            eval_metric='logloss'
        ))
    ])

    xgb_pipeline.fit(x_train, y_train)

    y_pred_train = xgb_pipeline.predict(x_train)
    y_pred_test = xgb_pipeline.predict(x_test)

    print(evaluate_model(
        'XGBoost',
        y_test,
        y_pred_test
    ))

    print("Train Accuracy:", accuracy_score(y_train, y_pred_train))
    print("Test Accuracy:",  accuracy_score(y_test, y_pred_test))
    print("Train F1:",       f1_score(y_train, y_pred_train))
    print("Test F1:",        f1_score(y_test, y_pred_test))
    print("Test Recall:",    recall_score(y_test, y_pred_test))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_test))

    xgb_pipeline.fit(x, y)
    joblib.dump(xgb_pipeline, "maintenance_model.pkl")   

def anomaly_model(df):
    x = df.drop(['maintenance_required', 'remaining_useful_life_h','anomaly_score', 'fault_state'], axis=1)
    anomaly_score = df['anomaly_score']
    x_train, x_test, anomaly_score_train, anomaly_score_test = split(x,anomaly_score)
    
    pipeline = Pipeline([
            ('preprocessor', preprocess(x)),
            ('model', LinearRegression())
            ])
    
    pipeline.fit(x_train,anomaly_score_train)
    anomaly_score_train_pred = pipeline.predict(x_train)
    anomaly_score_pred = pipeline.predict(x_test)
    print(evaluate_numeric("Linear regression Train",anomaly_score_train, anomaly_score_train_pred))
    print(evaluate_numeric("Linear regression Test",anomaly_score_test, anomaly_score_pred))

    pipeline.fit(x, anomaly_score)
    joblib.dump(pipeline, "anomaly_model.pkl")



def fault_state_model(df):
    od = OrdinalEncoder(categories=[['Warning', 'Critical']])
    df = df[df['maintenance_required'] == 1]
    df['fault_state'] = od.fit_transform(df[['fault_state']])
    x = df.drop(['maintenance_required', 'remaining_useful_life_h','anomaly_score', 'fault_state'], axis=1)
    fault_state = df['fault_state']
    x_train, x_test, y_train, y_test = split(x,fault_state)
    preprocessor = preprocess(x)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', LogisticRegression(C=10, max_iter=10000, random_state=42))
    ])

    pipeline.fit(x_train, y_train)
    y_pred_log = pipeline.predict(x_test)
    y_pred_train_log = pipeline.predict(x_train)
    print(evaluate_model('Logistic Regression', y_test, y_pred_log))
    print(evaluate_model('Logistic Regression Train', y_train, y_pred_train_log))

    pipeline.fit(x, fault_state)
    joblib.dump(pipeline, "fault_state_model.pkl")

    
def RUL_model(df):
    x = df.drop(['maintenance_required', 'remaining_useful_life_h','anomaly_score', 'fault_state'], axis=1)
    remaining_useful_life_h = df['remaining_useful_life_h']
    x_train, x_test, y_train, y_test = split(x,remaining_useful_life_h)
    preprocessor = preprocess(x)

    pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ('model', LinearRegression())
        ])
    
    pipeline.fit(x_train, y_train)
    RUL_pred = pipeline.predict(x_test)
    RUL_train_pred = pipeline.predict(x_train)

    print(evaluate_numeric("Linear regression train", y_train, RUL_train_pred))
    print(evaluate_numeric("Linear regression test", y_test, RUL_pred))

    pipeline.fit(x, remaining_useful_life_h)
    joblib.dump(pipeline, "remaining_useful_life_h.pkl")


def main():
    df = load_data()
    assert validate_data(df) == "valid"

    df = cleaning(df)
    df = drop_cols(df)
    print("anomaly: ")
    anomaly_model(df)
    print("fault state: ")
    fault_state_model(df)
    print("maintenance: ")
    maintenance_model(df)
    print("rul: ")
    RUL_model(df)

if __name__ == "__main__":
    main()

