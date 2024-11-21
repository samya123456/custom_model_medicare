# snowflake_util.py

import snowflake.connector
import pandas as pd


class SnowflakeUtil:
    # Configuration for Snowflake connection
    snowflake_config = {
        'user': 'samyanandy',
        'password': 'Snowflake@890',
        'account': 'aw86903.central-india.azure',
        'warehouse': 'COMPUTE_WH',
        'database': 'MEDICARE',
        'schema': 'MEDICARE_TABLE'
    }

    @staticmethod
    def get_connection():
        """Establishes a connection to Snowflake and returns the connection object."""
        conn = snowflake.connector.connect(
            user=SnowflakeUtil.snowflake_config['user'],
            password=SnowflakeUtil.snowflake_config['password'],
            account=SnowflakeUtil.snowflake_config['account'],
            warehouse=SnowflakeUtil.snowflake_config['warehouse'],
            database=SnowflakeUtil.snowflake_config['database'],
            schema=SnowflakeUtil.snowflake_config['schema']
        )
        return conn

    @staticmethod
    def call_function(sql_query):
        """Executes a SQL function call and returns the result."""
        conn = SnowflakeUtil.get_connection()

        try:
            with conn.cursor() as cur:
                cur.execute(sql_query)
                result = cur.fetchone()
                return result[0] if result else None
        except Exception as e:
            print(e)
        finally:
            conn.close()
