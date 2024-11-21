import snowflake.connector

from snowflake_conn_util import SnowflakeUtil
conn = SnowflakeUtil.get_connection()
try:
    cur = conn.cursor()
    cur.execute("SELECT MEDICARE.PUBLIC.DATA_PREPROCESSING('input_data')")
    result = cur.fetchone()
    print(f"Model saved at path: {result[0]}")

finally:
    cur.close()
    conn.close()
