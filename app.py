from flask import Flask, render_template
import pyodbc

app = Flask(__name__)

# Database configuration
server = 'retaildata-server.database.windows.net'
database = 'RetailData'
username = 'user'
password = 'Retail@1234'

# SQL Query
query = """
SELECT 
    h.HSHD_NUM,
    t.BASKET_NUM, t.PURCHASE, t.PRODUCT_NUM, p.DEPARTMENT, p.COMMODITY, t.SPEND, t.UNITS, t.STORE_R, t.WEEK_NUM, t.YEAR, h.L,
    h.AGE_RANGE, h.MARITAL,
    h.INCOME_RANGE, h.HOMEOWNER, h.HSHD_COMPOSITION, h.HH_SIZE, h.CHILDREN
FROM 
    transactions t
JOIN 
    households h ON t.HSHD_NUM = h.HSHD_NUM
JOIN 
    products p ON t.PRODUCT_NUM = p.PRODUCT_NUM
WHERE 
    h.HSHD_NUM = 10
ORDER BY 
    h.HSHD_NUM, 
    t.BASKET_NUM, 
    t.YEAR, 
    t.WEEK_NUM, 
    t.PRODUCT_NUM, 
    p.DEPARTMENT, 
    p.COMMODITY;
"""

@app.route('/')
def index():
    # Connect to Azure SQL Database
    conn = pyodbc.connect(f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}')
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    # Pass data to the template
    return render_template('dashboard.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
