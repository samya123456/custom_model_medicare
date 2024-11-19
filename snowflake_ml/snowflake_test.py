# main.py

from snowflake_conn_util import SnowflakeUtil  # Import the SnowflakeUtil class
import pandas as pd


def fetch_data_from_snowflake():
    # Get the connection
    conn = SnowflakeUtil.get_connection()

    try:
        # Execute a query and fetch the data
        query = 'SELECT * FROM MEDICARE.PUBLIC.MEDICARE_TABLE'
        df = pd.read_sql(query, conn)
    finally:
        # Close the connection
        conn.close()

    # Display the data
    print(df.head())


# Call the function to fetch data
fetch_data_from_snowflake()
