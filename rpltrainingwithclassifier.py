import boto3
import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging 
from xgboost import XGBClassifier

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO, # Set the minimum level to log (INFO, DEBUG, WARNING, ERROR)
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------

class AthenaDataFetcher:
    def __init__(self, region, database, bucket_name, s3_folder):
        self.athena_client = boto3.client('athena', region_name=region)
        self.database = database
        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        logger.info(f"Initialized AthenaDataFetcher for database: {database}")

    def run_query(self, query):
        logger.info("Starting Athena query execution...")
        # Start query execution
        response = self.athena_client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={'Database': self.database},
            ResultConfiguration={'OutputLocation': f's3://{self.bucket_name}/{self.s3_folder}'}
        )
        
        query_execution_id = response['QueryExecutionId']
        logger.info(f"Query Execution ID: {query_execution_id}. Waiting for completion.")
        status = 'RUNNING'

        # Wait for query execution to complete
        while status in ['RUNNING', 'QUEUED']:
            response = self.athena_client.get_query_execution(QueryExecutionId=query_execution_id)
            status = response['QueryExecution']['Status']['State']
            if status == 'FAILED':
                logger.error(f"Athena query failed: {response['QueryExecution']['Status']['StateChangeReason']}")
                raise Exception('Query failed: ' + response['QueryExecution']['Status']['StateChangeReason'])
            elif status == 'CANCELLED':
                logger.warning("Athena query was cancelled.")
                raise Exception('Query was cancelled')

        logger.info("Athena query execution finished. Fetching results.")

        # Paginate and fetch query results
        results_paginator = self.athena_client.get_paginator('get_query_results')
        results_iterator = results_paginator.paginate(QueryExecutionId=query_execution_id)

        # Parse results into a DataFrame
        columns = []
        rows = []

        for i, page in enumerate(results_iterator):
            for row_index, row in enumerate(page['ResultSet']['Rows']):
                # First row has column headers
                if i == 0 and row_index == 0:
                    columns = [col['VarCharValue'] for col in row['Data']]
                elif row_index == 0 and i > 0:
                    continue  # skip headers repeated on each page
                else:
                    rows.append([col.get('VarCharValue', None) for col in row['Data']])

        # Create DataFrame
        df = pd.DataFrame(rows, columns=columns)
        logger.info(f"Successfully fetched {len(df)} records from Athena.")

        # Convert specific columns to numeric types
        numeric_columns = ['source_id', 'campaign_id', 'price', 'agent_commission', 'outgoing_call_count', 'expected_revenue']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert columns to numeric, coerce errors to NaN
        
        logger.info("Numeric columns converted and returned DataFrame.")
        return df
    
# ----------------------------------------------------------------------

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        logger.info(f"Initialized DataPreprocessor with a DataFrame of size: {len(df)}")

    def clean_data(self):
        logger.info("Starting data cleaning...")
        
        # Drop columns with more than 50% null values
        null_percent = self.df.isnull().mean() * 100
        cols_over_50_nulls = null_percent[null_percent > 50].index.tolist()
        self.df = self.df.drop(columns=cols_over_50_nulls)
        logger.info(f"Dropped {len(cols_over_50_nulls)} columns with >50% nulls.")

        # Fill null values for object and bool columns
        obj_bool_cols = self.df.select_dtypes(include=['object', 'bool'])
        cols_with_nulls = obj_bool_cols.columns[obj_bool_cols.isnull().any()].tolist()
        self.df[cols_with_nulls] = self.df[cols_with_nulls].fillna("Unknown")
        logger.info(f"Filled nulls in {len(cols_with_nulls)} categorical columns.")

        # Fill null values for numeric columns
        numeric_cols = self.df.select_dtypes(include=['float64', 'int64'])
        numeric_cols_with_nulls = numeric_cols.columns[numeric_cols.isnull().any()].tolist()
        self.df[numeric_cols_with_nulls] = self.df[numeric_cols_with_nulls].fillna(0)
        logger.info(f"Filled nulls in {len(numeric_cols_with_nulls)} numeric columns with 0.")

        # Handle date columns
        self.df['r_created_at'] = pd.to_datetime(self.df['r_created_at'], errors='coerce')
        self.df['r_updated_at'] = pd.to_datetime(self.df['r_updated_at'], errors='coerce')
        self.df['date_range'] = (self.df['r_updated_at'] - self.df['r_created_at']).dt.days.fillna(0)
        logger.info("Created 'date_range' feature.")

        # Drop unnecessary columns
        columns_to_drop = [
            'event_date', 'timestamp', 'departmental_rollup', 'r_updated_at', 'ip_address', 
            'user_agent', 'page_url', 'r_created_at', 'r_created_at', 'agent_commission', 
            'connector_id'
        ]
        columns_dropped_final = [col for col in columns_to_drop if col in self.df.columns] # Check which exist
        self.df.drop(columns=columns_dropped_final, axis=1, inplace=True)
        logger.info(f"Dropped final list of {len(columns_dropped_final)} unnecessary columns.")

        # Ensure non-negative expected_revenue
        initial_negative_count = (self.df['expected_revenue'] < 0).sum()
        self.df['expected_revenue'] = self.df['expected_revenue'].apply(lambda x: 0 if x < 0 else x)
        logger.info(f"Set {initial_negative_count} negative 'expected_revenue' values to 0.")
        logger.info(f"Data cleaning complete. Final DataFrame shape: {self.df.shape}")

    def encode_categorical(self):
        logger.info("Starting categorical encoding...")
        encoders = {}
        cat_cols = self.df.select_dtypes(include=['object', 'bool'])

        for col in cat_cols:
            le = LabelEncoder()
            self.df[col] = le.fit_transform(self.df[col].astype(str))
            encoders[col] = le
        
        logger.info(f"Encoded {len(cat_cols.columns)} categorical columns using LabelEncoder.")
        return self.df, encoders

# ----------------------------------------------------------------------

class ModelTrainer:
    def __init__(self, df):
        self.df = df
        logger.info("Initialized ModelTrainer.")

    def split_data(self):
        # Drop only 'expected_revenue' for the *Classifier* (runs on ALL data)
        # We assume 'payout_lead' is the target (binary)
        X = self.df.drop(columns=['expected_revenue', 'payout_lead']) 
        y_regressor = self.df['expected_revenue']
        y_classifier = self.df['payout_lead']
        
        # Split features for training. We will use y_regressor later.
        # Use X, y_regressor for splitting to maintain consistent indices
        X_train, X_test, y_reg_train, y_reg_test = train_test_split(X, y_regressor, test_size=0.1, random_state=45)
        # Realign classifier target indices
        y_cls_train = y_classifier.loc[X_train.index]
        y_cls_test = y_classifier.loc[X_test.index]
        
        logger.info(f"Data split into train ({len(X_train)} rows) and test ({len(X_test)} rows).")
        return X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test

    # MODIFIED: Trains BOTH the Classifier (on ALL data) and Regressor (on filtered data)
    def train_models(self, X_train_full, X_test_full, y_reg_train_full, y_reg_test_full, y_cls_train_full, y_cls_test_full):
        
        # 1. Train Classifier (Uses ALL data, with 'payout_lead' as target)
        logger.info("Starting Classifier Training (XGBClassifier)...")
        classifier = XGBClassifier(
            n_estimators=300, 
            learning_rate=0.05, 
            max_depth=7, 
            random_state=45,
            use_label_encoder=False, 
            eval_metric="logloss"
        )
        classifier.fit(X_train_full, y_cls_train_full)
        y_cls_pred = classifier.predict(X_test_full)
        logger.info(f"Classifier Accuracy: {accuracy_score(y_cls_test_full, y_cls_pred):.4f}")
        
        # 2. Prepare data for Regressor (Filter the training set for positive examples)
        X_train_regressor = X_train_full[y_cls_train_full == 1]
        y_train_regressor = y_reg_train_full[y_cls_train_full == 1]
        
        X_test_regressor = X_test_full[y_cls_test_full == 1]
        y_test_regressor = y_reg_test_full[y_cls_test_full == 1]
        
        logger.info(f"Filtered Regressor Training Data Size: {len(X_train_regressor)}")
        
        # 3. Train Regressor (Uses ONLY filtered data)
        logger.info("Starting Regressor Training (XGBRegressor)...")
        regressor = xgb.XGBRegressor(
            n_estimators=600,
            learning_rate=0.04,
            max_depth=9,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=45,
            eval_metric="rmse"
        )

        eval_set = [(X_train_regressor, y_train_regressor), (X_test_regressor, y_test_regressor)]
        regressor.fit(X_train_regressor, y_train_regressor, eval_set=eval_set, verbose=25)

        # Regressor Evaluation (only on positive test data)
        y_pred_regressor = regressor.predict(X_test_regressor)
        logger.info(f"Regressor R² Score (on payout leads): {r2_score(y_test_regressor, y_pred_regressor):.4f}")

        return classifier, regressor


# ----------------------------------------------------------------------

class S3Uploader:
    def __init__(self, bucket_name, s3_folder):
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.s3_folder = s3_folder
        logger.info(f"Initialized S3Uploader for bucket: {bucket_name}")

    def upload_to_s3(self, file_name):
        s3_key = f"{self.s3_folder}{file_name}"
        logger.info(f"Attempting to upload {file_name} to S3 location: {s3_key}")
        try:
            self.s3_client.upload_file(file_name, self.bucket_name, s3_key)
            logger.info(f"Successfully uploaded {file_name} to S3!")
        except Exception as e:
            logger.error(f"Error uploading {file_name} to S3: {e}")
            raise

# ----------------------------------------------------------------------

def main():
    logger.info("--- Starting RPL Model Training Pipeline ---")
    
    # STEP 1: Athena data fetch
    logger.info("STEP 1: Initializing Athena Data Fetcher.")
    athena_fetcher = AthenaDataFetcher(
        region='us-west-2', 
        database='sampledb', 
        bucket_name='mlbucket-sagemaker', 
        s3_folder='ANALYTICS_REPORTS/CHANNEL/'
    )
    query = """
    SELECT
        u.source_id, u.utm_content, u.r_created_at, u.r_updated_at, u.connector_id, u.timestamp,
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
        leads.expected_revenue
    FROM unified."uct_events" u
    JOIN sampledb."revenue_leads_plus" leads
        ON u.utm_content = leads.utm_content
    WHERE
        u.connector_id = '1'
    LIMIT 3000000;
    """
    df = athena_fetcher.run_query(query)
    
    # STEP 2: Data preprocessing
    logger.info("STEP 2: Starting Data Preprocessing.")
    preprocessor = DataPreprocessor(df)
    preprocessor.clean_data()
    df_cleaned, encoders = preprocessor.encode_categorical()

    # STEP 3: Classifier Target Creation (No filtering yet, keep all data)
    logger.info("STEP 3: Creating Binary Classifier Target 'payout_lead'.")
    # This column is used for the Classifier model target and Regressor filtering
    df_cleaned['payout_lead'] = np.where(df_cleaned['expected_revenue'] >= 2.0, 1, 0)
    logger.info(f"Target distribution: 1s={df_cleaned['payout_lead'].sum()}, 0s={len(df_cleaned) - df_cleaned['payout_lead'].sum()}")
    
    # STEP 4: Model training (Train both Classifier and Regressor)
    logger.info("STEP 4: Training both Classifier and Regressor Models.")
    model_trainer = ModelTrainer(df_cleaned) 
    X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test = model_trainer.split_data()
    
    classifier_model, regressor_model = model_trainer.train_models(
        X_train, X_test, y_reg_train, y_reg_test, y_cls_train, y_cls_test
    )

    # STEP 5: Save models locally
    logger.info("STEP 5: Saving Models Locally.")
    today_str = datetime.today().strftime("%d%m%Y_%H%M%S")
    
    # Naming convention change: Save both models and encoders
    encoder_file = f"encoders_{today_str}.pkl"
    classifier_file = f"model_{today_str}_classifier.pkl"
    regressor_file = f"model_{today_str}_regressor.pkl"
    
    joblib.dump(encoders, encoder_file)
    joblib.dump(classifier_model, classifier_file) # Save Classifier
    joblib.dump(regressor_model, regressor_file)   # Save Regressor
    logger.info(f"Saved three artifacts: encoders, classifier, regressor locally.")

    # STEP 6: Uploading Models to S3 
    logger.info("STEP 6: Uploading Models to S3.")
    s3_uploader = S3Uploader(
        bucket_name='mlbucket-sagemaker', 
        s3_folder='ANALYTICS_REPORTS/Model_weights/'
    )
    s3_uploader.upload_to_s3(encoder_file)
    s3_uploader.upload_to_s3(classifier_file)
    s3_uploader.upload_to_s3(regressor_file)
    
    logger.info("--- Pipeline Execution Complete ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.critical(f"A critical error occurred in the pipeline: {e}", exc_info=True)
