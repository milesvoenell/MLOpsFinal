import psycopg2
import pandas as pd
import boto3
from urllib.parse import urlparse

# -----------------------------
# Neon Postgres connection info
# -----------------------------
conn = psycopg2.connect(
    "postgresql://neondb_owner:npg_pIP6NB8xRSEF@ep-lucky-cloud-adov6iq2-pooler.c-2.us-east-1.aws.neon.tech:5432/neondb"
)

# -----------------------------
# Query only the runs table
# -----------------------------
query = """
SELECT
    run_uuid AS run_id,
    name AS run_name,
    status AS run_status,
    artifact_uri
FROM runs
WHERE experiment_id = 1
ORDER BY name;
"""

# -----------------------------
# Execute query and load into pandas
# -----------------------------
df = pd.read_sql_query(query, conn)
conn.close()

# -----------------------------
# Set up AWS S3 client
# -----------------------------
s3 = boto3.client('s3')

def check_s3_artifact(uri):
    if uri is None:
        return False
    parsed = urlparse(uri)
    bucket = parsed.netloc
    prefix = parsed.path.lstrip('/')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        return 'Contents' in response
    except Exception as e:
        print(f"Error checking {uri}: {e}")
        return False

# -----------------------------
# Check each artifact
# -----------------------------
df['artifact_exists'] = df['artifact_uri'].apply(check_s3_artifact)

# -----------------------------
# Display results
# -----------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 150)
print(df)
