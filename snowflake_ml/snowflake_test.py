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


def call_addone_function():
    # Define the SQL to call the function
    call_function_sql = "SELECT addone(5);"

    # Call the function and get the result
    result = SnowflakeUtil.call_function(call_function_sql)
    print("Result of addone(5):", result)


# Execute the function call
call_addone_function()

# fetch_data_from_snowflake()
