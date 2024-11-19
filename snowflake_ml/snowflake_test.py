import snowflake.connector
import pandas as pd

# Define connection parameters
snowflake_config = {
    'user': 'samyanandy',
    'password': 'Snowflake@890',
    'account': 'aw86903.central-india.azure',
    'warehouse': 'COMPUTE_WH',
    'database': 'MEDICARE',
    'schema': 'MEDICARE_TABLE'
}

# Establish the connection
conn = snowflake.connector.connect(
    user=snowflake_config['user'],
    password=snowflake_config['password'],
    account=snowflake_config['account'],
    warehouse=snowflake_config['warehouse'],
    database=snowflake_config['database'],
    schema=snowflake_config['schema']
)

# Execute a query and fetch the data
query = 'SELECT * FROM MEDICARE.PUBLIC.MEDICARE_TABLE'
df = pd.read_sql(query, conn)

# Close the connection
conn.close()

# Display the data
print(df.head())
