# --- START: New Package Installation Block ---
import subprocess
import sys
import os
# --- All other imports now run *after* installation ---
import io
import time
import boto3
import joblib
import pickle
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder  # needed if encoders are sklearn LabelEncoder

try:
    import xgboost  # noqa: F401
except Exception:
    pass
try:
    import sklearn  # noqa: F401
except Exception:
    pass

def install_packages():
    """
    Installs all required packages using subprocess.
    This runs before any other imports.
    """
    print("--- ðŸ“¦ Starting package installation using subprocess ---")
    # List all required packages here
    packages = [
        'joblib>=1.2.0',
        'pandas>=1.5.0',
        'boto3>=1.26.0',
        'xgboost>=1.7.0',
        'scikit-learn>=1.2.0',
        'plotly>=5.10.0',
        'tqdm>=4.60.0'
    ]
    
    for package in packages:
        try:
            print(f"--- Installing {package}... ---")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"--- âœ… Successfully installed {package} ---")
        except subprocess.CalledProcessError as e:
            print(f"---  ERROR: Failed to install {package}: {e} ---")
            sys.exit(1) # Exit if any installation fails
    print("--- âœ… All packages installed successfully. ---")

# Call the installer function *before* other imports
install_packages()
# --- END: New Package Installation Block ---



# -------------------------------------------------------------------------------

# =========================
# S3 CONFIG â€“ EDIT IF NEEDED
# =========================
S3_REGION = "us-west-2"
S3_MODEL_BUCKET = "mlbucket-sagemaker"
S3_MODEL_KEY = "ANALYTICS_REPORTS/Model_weights/xgb_regression_model_new.pkl"
S3_ENCODERS_KEY = "ANALYTICS_REPORTS/Model_weights/label_encoders_updated.pkl"

# ===========
# S3 Utilities
# ===========
_s3_client = boto3.client("s3", region_name=S3_REGION)

def load_pickle_like_from_s3(bucket: str, key: str):
    """Load an object saved via joblib or pickle directly from S3 into memory."""
    print(f"ðŸ“¦ Loading from s3://{bucket}/{key}")
    obj = _s3_client.get_object(Bucket=bucket, Key=key)
    bytestream = io.BytesIO(obj["Body"].read())  # make it seekable

    # Try joblib first (handles compressed arrays well); fallback to pickle
    try:
        return joblib.load(bytestream)
    except Exception as e:
        print(f"âš ï¸ joblib failed ({e}); trying pickle...")
        bytestream.seek(0)
        return pickle.load(bytestream)


# ======================
# Athena query management
# ======================
class AthenaQueryExecutor:
    """Handles Athena queries and data retrieval."""

    def __init__(self, database: str, output_location: str, region: str = 'us-west-2'):
        self.database = database
        self.output_location = output_location
        self.client = boto3.client('athena', region_name=region)

    def execute_query(self, query: str) -> str:
        print("    Query: Starting execution...")
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': self.database},
            ResultConfiguration={'OutputLocation': self.output_location}
        )
        print(f"    Query: Execution ID: {response['QueryExecutionId']}")
        return response['QueryExecutionId']

    def wait_for_completion(self, execution_id: str):
        print("    Query: Waiting for completion...")
        status = 'RUNNING'
        while status in ['RUNNING', 'QUEUED']:
            response = self.client.get_query_execution(QueryExecutionId=execution_id)
            status = response['QueryExecution']['Status']['State']

            if status == 'FAILED':
                reason = response['QueryExecution']['Status']['StateChangeReason']
                raise Exception(f"Query failed: {reason}")
            elif status == 'CANCELLED':
                raise Exception("Query was cancelled")
            time.sleep(2)
        print("    Query: Succeeded.")

    def fetch_results(self, execution_id: str) -> pd.DataFrame:
        print("    Query: Fetching results...")
        paginator = self.client.get_paginator('get_query_results')
        iterator = paginator.paginate(QueryExecutionId=execution_id)

        columns, rows = [], []
        for i, page in enumerate(iterator):
            for row_index, row in enumerate(page['ResultSet']['Rows']):
                if i == 0 and row_index == 0:
                    columns = [col['VarCharValue'] for col in row['Data']]
                elif row_index == 0 and i > 0:
                    continue
                else:
                    rows.append([col.get('VarCharValue', None) for col in row['Data']])
        print("    Query: Results fetched.")
        return pd.DataFrame(rows, columns=columns)

    def run(self, query: str) -> pd.DataFrame:
        execution_id = self.execute_query(query)
        self.wait_for_completion(execution_id)
        return self.fetch_results(execution_id)


# ======================
# Model + encoders loader
# ======================
class ModelHandler:
    """Loads and manages ML model and encoders from S3."""

    def __init__(self, bucket: str, model_key: str, encoder_key: str):
        self.bucket = bucket
        self.model_key = model_key
        self.encoder_key = encoder_key
        self.model, self.encoders = self._load_model()

    def _load_model(self):
        print("Loading model and encoders from S3...")
        model = load_pickle_like_from_s3(self.bucket, self.model_key)
        encoders = load_pickle_like_from_s3(self.bucket, self.encoder_key)
        print("âœ… Model and encoders loaded.")
        return model, encoders


# =================
# Data Preprocessing
# =================
class DataPreprocessor:
    """Cleans and preprocesses the dataset."""

    @staticmethod
    def preprocess(df: pd.DataFrame) -> pd.DataFrame:
        print("    Preprocessing: Dropping high-null columns...")
        null_percent = df.isnull().mean() * 100
        cols_over_50_nulls = null_percent[null_percent > 50].index.tolist()
        df = df.drop(columns=cols_over_50_nulls)

        print("    Preprocessing: Filling nulls in object columns with 'Unknown'...")
        obj_bool_cols = df.select_dtypes(include=['object', 'bool'])
        obj_cols_with_nulls = obj_bool_cols.columns[obj_bool_cols.isnull().any()].tolist()
        df[obj_cols_with_nulls] = df[obj_cols_with_nulls].fillna("Unknown")

        print("    Preprocessing: Converting string-numeric columns to numeric...")
        numeric_cols_to_convert = ['price', 'outgoing_call_count', 'concierge_triggered', 'expected_revenue']
        
        for col in numeric_cols_to_convert:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print("    Preprocessing: Filling nulls in numeric columns with 0...")
        numeric_cols = df.select_dtypes(include=['float64', 'int64'])
        numeric_cols_with_nulls = numeric_cols.columns[numeric_cols.isnull().any()].tolist()
        df[numeric_cols_with_nulls] = df[numeric_cols_with_nulls].fillna(0)

        print("    Preprocessing: Processing date columns...")
        if 'r_created_at' in df.columns:
            df['r_created_at'] = pd.to_datetime(df['r_created_at'], errors='coerce')
        if 'r_updated_at' in df.columns:
            df['r_updated_at'] = pd.to_datetime(df['r_updated_at'], errors='coerce')
        if 'r_created_at' in df.columns and 'r_updated_at' in df.columns:
            df['date_range'] = (df['r_updated_at'] - df['r_created_at']).dt.days.fillna(0)

        print("    Preprocessing: Dropping unused columns...")
        columns_to_drop = [
            'event_date', 'timestamp', 'departmental_rollup', 'ip_address',
            'user_agent', 'page_url', 'r_created_at', 'agent_commission',
            'source_id', 'connector_id', 'campaign_id'
        ]
        df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True, errors='ignore')
        print("âœ… Preprocessing complete.")
        return df


# =========
# Predictor
# =========
class Predictor:
    """Handles data encoding and prediction."""

    def __init__(self, model_handler: ModelHandler):
        self.model = model_handler.model
        self.encoders = model_handler.encoders or {}
        self.model_feature_names = self._extract_model_feature_names()
        if self.model_feature_names is not None:
            print(f"    Model expects {len(self.model_feature_names)} features. Sample: {self.model_feature_names[:10]}")

    def _extract_model_feature_names(self):
        """Try to extract feature names from several model types."""
        try:
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names is not None:
                    return list(booster.feature_names)
            if hasattr(self.model, "feature_names_in_"):
                return list(self.model.feature_names_in_)
            if hasattr(self.model, "feature_names"):
                return list(self.model.feature_names)
        except Exception as e:
            print(f"    âš ï¸ Could not extract feature names from model: {e}")
        return None

    def _safe_label_encode_series(self, series: pd.Series, le: LabelEncoder):
        """Map series values to encoder integers, unknown -> -1."""
        classes = list(getattr(le, "classes_", []))
        mapping = {c: i for i, c in enumerate(classes)}
        return series.astype(str).map(mapping).fillna(-1).astype(int)

    def encode_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical columns using provided encoders dict."""
        print("    Encoding data for prediction...")
        df = df.copy()
        if not self.encoders:
            print("    No encoders found, skipping encoding.")
            return df

        for col, le in tqdm(self.encoders.items(), desc="    Encoding columns"):
            if col not in df.columns:
                print(f"    âš ï¸ Encoder exists for column '{col}' but column not present. Skipping.")
                continue
            try:
                if hasattr(le, "classes_"):
                    df[col] = self._safe_label_encode_series(df[col], le)
                elif isinstance(le, dict):
                    df[col] = df[col].map(le).fillna(-1).astype(int)
                else:
                    df[col] = df[col].astype(str).apply(lambda x: le.transform([x])[0] if x in getattr(le, "classes_", []) else -1)
            except Exception as e:
                print(f"    âš ï¸ Failed encoding column {col}: {e}. Falling back to -1.")
                df[col] = df[col].astype(str).map({}).fillna(-1).astype(int)
        print("    Encoding complete.")
        return df

    def predict(self, df: pd.DataFrame) -> dict:
        print("    Preparing features for prediction...")
        encoded_df = self.encode_data(df.copy())

        features_df = encoded_df.drop(columns=['r_updated_at', 'expected_revenue'], errors='ignore')

        if self.model_feature_names:
            missing_features = [c for c in self.model_feature_names if c not in features_df.columns]
            if missing_features:
                print(f"    âš ï¸ Adding {len(missing_features)} missing feature columns with zeros.")
                for c in missing_features:
                    features_df[c] = 0

            extra_cols = [c for c in features_df.columns if c not in self.model_feature_names]
            if extra_cols:
                print(f"    Dropping {len(extra_cols)} unexpected columns (e.g., utm_content).")
                features_df = features_df.drop(columns=extra_cols)

            features_df = features_df[self.model_feature_names]
        else:
            features_df = features_df.sort_index(axis=1)
            print("    â„¹ï¸ Model feature names not found; using alphabetic feature ordering.")

        for col in features_df.columns:
            if features_df[col].dtype == 'object' or pd.api.types.is_string_dtype(features_df[col]):
                print(f"    Fixing non-numeric column {col} before prediction...")
                converted = pd.to_numeric(features_df[col], errors='coerce')
                if converted.notna().sum() > 0:
                    features_df[col] = converted.fillna(0)
                else:
                    mapping = {v: i for i, v in enumerate(features_df[col].astype(str).unique())}
                    features_df[col] = features_df[col].astype(str).map(mapping).fillna(0).astype(int)

        try:
            if hasattr(self.model, "get_booster"):
                booster = self.model.get_booster()
                if hasattr(booster, "feature_names") and booster.feature_names is not None:
                    if len(booster.feature_names) != features_df.shape[1]:
                        raise ValueError(f"Feature count mismatch: model expects {len(booster.feature_names)} but input has {features_df.shape[1]}.")
            elif hasattr(self.model, "n_features_in_"):
                if self.model.n_features_in_ != features_df.shape[1]:
                    raise ValueError(f"Feature count mismatch: model expects {self.model.n_features_in_} but input has {features_df.shape[1]}.")
        except Exception as e:
            print(f"    âš ï¸ Pre-predict validation failed: {e}")
            raise

        print("    Running model.predict()...")
        try:
            if hasattr(self.model, "predict_proba"):
                pred = self.model.predict_proba(features_df)[:, 1]
            else:
                pred = self.model.predict(features_df)
        except Exception as e:
            print(f"    âš ï¸ Direct model.predict failed: {e}. Trying numpy array...")
            pred = self.model.predict(features_df.values)

        print(f"âœ… Prediction complete. Generated {len(pred)} predictions.")
        return {"expected_net_revenue_no_buyer": pred.tolist()}


# ================
# Report & Uploads
# ================
class ReportGenerator:
    """Aggregates and saves results, and generates plots."""

    @staticmethod
    def generate_date_summary(df: pd.DataFrame, predictions: dict) -> pd.DataFrame:
        """Generates the original date-level summary for charting."""
        print("    Report: Generating DATE-LEVEL summary...")
        df_date = df.copy() 
        
        if 'r_updated_at' in df_date.columns and not pd.api.types.is_string_dtype(df_date['r_updated_at']):
            # Handle NaT (Not a Time) values before strftime
            df_date['r_updated_at'] = df_date['r_updated_at'].fillna(pd.Timestamp('1970-01-01'))
            df_date['r_updated_at'] = df_date['r_updated_at'].dt.strftime('%d/%m/%Y')
        elif 'r_updated_at' not in df_date.columns:
             df_date['r_updated_at'] = 'unknown_date'
             
        df_date['predicted_revenue'] = predictions['expected_net_revenue_no_buyer']

        if 'expected_revenue' not in df_date.columns:
            df_date['expected_revenue'] = 0.0

        grouped = df_date.groupby('r_updated_at').agg(
            avg_expected_revenue_per_lead=('expected_revenue', 'mean'),
            avg_predicted_revenue_per_lead=('predicted_revenue', 'mean')
        ).reset_index()

        grouped.columns = ['date', 'avg_expected_revenue_per_lead', 'avg_predicted_revenue_per_lead']
        print("    Report: DATE-LEVEL summary complete.")
        return grouped

    @staticmethod
    def generate_session_summary(df: pd.DataFrame, predictions: dict) -> pd.DataFrame:
        """Generates the new session-level summary for the Glue table."""
        print("    Report: Generating SESSION-LEVEL summary...")
        df_session = df.copy()
        df_session['predicted_revenue'] = predictions['expected_net_revenue_no_buyer']

        if 'expected_revenue' not in df_session.columns:
            df_session['expected_revenue'] = 0.0

        if 'utm_content' not in df_session.columns:
            print("    âš ï¸ 'utm_content' (session_id) not found. Using a placeholder.")
            df_session['utm_content'] = 'unknown_session'

        grouped = df_session.groupby('utm_content').agg(
            raw_expected_revenue_per_session=('expected_revenue', 'sum'),
            raw_predicted_revenue_per_session=('predicted_revenue', 'sum'),
            lead_count=('utm_content', 'size')
        ).reset_index()

        grouped.rename(columns={'utm_content': 'session_id'}, inplace=True)
        print("    Report: SESSION-LEVEL summary complete.")
        return grouped

    @staticmethod
    def create_plot(df: pd.DataFrame, file_name: str):
        """Generates the time-series plot from the date-level summary."""
        print("    Report: Creating date-level plot...")
        date = df['date']
        data = {
            "Predicted_RPL": df['avg_expected_revenue_per_lead'],
            "Actual_RPL": df['avg_predicted_revenue_per_lead'],
            "RPL_Goal": [15] * len(df)
        }

        df_plot = pd.DataFrame(data)
        # Avoid division by zero
        df_plot["RPL_Lift"] = ((df_plot["Actual_RPL"] - df_plot["Predicted_RPL"]) / df_plot["Predicted_RPL"].replace(0, 1)) * 100

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=date, y=df_plot["RPL_Lift"], name="RPL Lift",
                                 mode='lines+markers+text', text=[f"{v:.2f}%" for v in df_plot["RPL_Lift"]],
                                 textposition="top center"))
        fig.add_trace(go.Scatter(x=date, y=df_plot["Predicted_RPL"], name="Predicted RPL",
                                 mode='lines+markers', yaxis="y2"))
        fig.add_trace(go.Scatter(x=date, y=df_plot["Actual_RPL"], name="Actual RPL",
                                 mode='lines+markers', yaxis="y2"))

        fig.update_layout(
            title="Part 1 - High Level RPL",
            xaxis=dict(title="Date"),
            yaxis=dict(title="RPL Lift (%)", side="left"),
            yaxis2=dict(title="RPL ($)", overlaying="y", side="right"),
            legend=dict(x=0.5, xanchor="center", orientation="h"),
            plot_bgcolor='white', height=500, margin=dict(l=40, r=40, t=60, b=40)
        )

        fig.write_html(file_name)
        print(f"    Report: Date-level plot saved to {file_name}")

    @staticmethod
    def upload_to_s3(local_path: str, bucket: str, s3_key: str):
        print(f"    Uploading {local_path} to s3://{bucket}/{s3_key}...")
        s3 = boto3.client('s3', region_name=S3_REGION)
        s3.upload_file(Filename=local_path, Bucket=bucket, Key=s3_key)
        print(f"    Upload complete for {local_path}.")


# =========
# Pipeline
# =========
class Pipeline:
    """Main orchestrator that runs the full process."""

    def __init__(self):
        print("Initializing Pipeline...")
        self.athena = AthenaQueryExecutor(
            database='sampledb',
            output_location='s3://mlbucket-sagemaker/ANALYTICS_REPORTS/CHANNEL/',
            region=S3_REGION
        )
        self.model_handler = ModelHandler(
            bucket=S3_MODEL_BUCKET,
            model_key=S3_MODEL_KEY,
            encoder_key=S3_ENCODERS_KEY
        )
        print("âœ… Pipeline initialized.")

    def run(self):
        print("\n--- ðŸš€ Starting Pipeline Run ---")
        query = """SELECT
            u.source_id, u.r_created_at, u.r_updated_at, u.connector_id, u.timestamp,
            u.event_type, u.campaign_id, u.ip_address, u.user_agent, u.status,
            u.page_url, u.device_brand, u.device_model, u.device_os, u.device_os_version,
            u.event_date, leads.source, leads.sub_source, leads.traffic_source,
            leads.marketing_source, leads.marketing_campaign, leads.marketing_channel,
            leads.departmental_rollup, leads.business_product, leads.marketing_is_branded,
            leads.user_type, leads.area_type, leads.property_type, leads.property_use,
            leads.already_has_agent, leads.notes_to_agents, leads.notes_from_agent,
            leads.client_notes, leads.extra_info, leads.fake, leads.qualification_talked_to_client,
            leads.qualification_left_voicemail, leads.qualification_is_an_agent,
            leads.concierge_triggered, leads.is_deal, leads.stage, leads.furthest_stage,
            leads.sell_reason, leads.zip_code, leads.msa_name, leads.city_name, leads.state_name,
            leads.state_code, leads.price, leads.agent_commission, leads.outgoing_call_count,
            leads.expected_revenue,
            u.utm_content
        FROM unified."uct_events" u
        JOIN sampledb."revenue_leads_plus" leads
            ON u.utm_content = leads.utm_content
        WHERE
            u.connector_id = '1'
            AND date_parse(substr(u.r_updated_at, 1, 19), '%Y-%m-%d %H:%i:%s') >= date_add('day', -112, current_date)
        LIMIT 50000;
        """

        print("1. Running Athena query...")
        df = self.athena.run(query)
        print(f"âœ… Athena query complete. Fetched {len(df)} rows.")

        print("\n2. Preprocessing data...")
        df_cleaned = DataPreprocessor.preprocess(df.copy()) # Use a copy to be safe

        print("\n3. Making predictions...")
        predictor = Predictor(self.model_handler)
        predictions = predictor.predict(df_cleaned) 
        
        today_str = datetime.today().strftime("%d%m%Y_%H%M%S")

        # --- PART 4A: Date-Level Report & Chart ---
        print("\n4A. Generating Date-Level Report & Chart...")
        date_report = ReportGenerator.generate_date_summary(df_cleaned.copy(), predictions)
        
        graph_html = f"graph_{today_str}.html"
        time.sleep(2)
        ReportGenerator.create_plot(date_report, graph_html)
        ReportGenerator.upload_to_s3(
            graph_html,
            S3_MODEL_BUCKET,
            'ANALYTICS_REPORTS/Model_charts/graph.html'
        )
        print("âœ… Date-level chart uploaded.")

        # --- PART 4B: Session-Level Report & Glue Table ---
        print("\n4B. Generating Session-Level Report & Glue Table...")
        session_report = ReportGenerator.generate_session_summary(df_cleaned.copy(), predictions)
        
        session_csv = f"result_session_summary_{today_str}.csv"
        session_report.to_csv(session_csv, index=False)
        print(f"    Saved session report to {session_csv}")

        # --- Define new, dedicated S3 paths ---
        session_s3_key = 'ANALYTICS_REPORTS/RPL_session_summary/result_session_summary.csv'
        session_s3_location = "s3://mlbucket-sagemaker/ANALYTICS_REPORTS/RPL_session_summary/"

        ReportGenerator.upload_to_s3(
            session_csv,
            S3_MODEL_BUCKET,
            session_s3_key  # Use the new, dedicated key
        )
        print("âœ… Session-level report uploaded to its dedicated folder.")
        
        print("\n5. Updating Glue table 'rpl_session_summary'...")
        glue = boto3.client('glue', region_name=S3_REGION)
        database_name = "ml_metrics"
        table_name = "rpl_session_summary_final" 
        
        # --- Point Glue table to the new dedicated location ---
        s3_path = session_s3_location 

        columns = [
            {"Name": "session_id", "Type": "string"},
            {"Name": "raw_expected_revenue_per_session", "Type": "double"},
            {"Name": "raw_predicted_revenue_per_session", "Type": "double"},
            {"Name": "lead_count", "Type": "bigint"}
        ]

        try:
            glue.get_table(DatabaseName=database_name, Name=table_name)
            print(f"    Glue table {database_name}.{table_name} already exists. Skipping creation.")
            print(f"    !!!! PLEASE VERIFY: Make sure this table's location in Glue is {s3_path} !!!!")

        except glue.exceptions.EntityNotFoundException:
            print(f"    Creating Glue table {database_name}.{table_name}...")
            glue.create_table(
                DatabaseName=database_name,
                TableInput={
                    'Name': table_name,
                    'StorageDescriptor': {
                        'Columns': columns,
                        'Location': s3_path, # This now correctly uses the new path
                        'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
                        'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
                        'SerdeInfo': {
                            'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                            'Parameters': {'field.delim': ','}
                        }
                    },
                    'TableType': 'EXTERNAL_TABLE',
                    'Parameters': {'EXTERNAL': 'TRUE'}
                }
            )
            print(f"    âœ… Glue table {database_name}.{table_name} created successfully!")
        
        print("\n--- ðŸŽ‰ Pipeline execution complete! ---")


if __name__ == "__main__":
    
    # The main execution block
    
    # After installing packages, run the main pipeline
    pipeline = Pipeline()
    pipeline.run()
