from flask import Flask, render_template, request, jsonify, redirect, url_for
import pyodbc
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import urllib.parse
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import joblib
from datetime import datetime
import os
from dotenv import load_dotenv

app = Flask(__name__)


server = 'retaildata-server.database.windows.net'
database = 'RetailData'
username = 'user'
password = 'Retail@1234'
password = urllib.parse.quote_plus(password)

connection_string = f"mssql+pyodbc://{username}:{password}@{server}/{database}?driver=ODBC+Driver+17+for+SQL+Server"

# Create an SQLAlchemy engine
engine = create_engine(connection_string)
#engine = create_engine("mysql+pymysql://{username}:{password}@{server}/{database}"
 #       .format(server='retaildata-server.database.windows.net', database='RetailData', username='user', password='Retail@1234'))

# Database connection function using pyodbc
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=retaildata-server.database.windows.net;'
        'DATABASE=RetailData;'
        'UID=user;'
        'PWD=Retail@1234'
    )
    return conn

@app.route('/demographicsandengagement', methods=['GET'])
def demographicsandengagement():
    conn = get_db_connection()
    
    # Query to analyze total spend based on household size, income range, and presence of children
    query = """
        SELECT hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN, SUM(tr.SPEND) AS TOTAL_SPEND
        FROM transactions tr
        INNER JOIN households hh ON tr.HSHD_NUM = hh.HSHD_NUM
        GROUP BY hh.HH_SIZE, hh.INCOME_RANGE, hh.CHILDREN
    """
    data_df = pd.read_sql_query(query, conn)

    # Query to analyze year-over-year household spending
    yoy_spend_query = """
        SELECT YEAR, SUM(SPEND) AS TOTAL_SPEND
        FROM transactions
        GROUP BY YEAR
        ORDER BY YEAR
    """
    yoy_spend_df = pd.read_sql_query(yoy_spend_query, conn)

    # Query to analyze product category popularity
    category_popularity_query = """
        SELECT pr.DEPARTMENT, SUM(tr.UNITS) AS TOTAL_UNITS
        FROM transactions tr
        INNER JOIN products pr ON tr.PRODUCT_NUM = pr.PRODUCT_NUM
        GROUP BY pr.DEPARTMENT
        ORDER BY TOTAL_UNITS DESC
    """
    category_popularity_df = pd.read_sql_query(category_popularity_query, conn)

    # Query to analyze product combinations driving cross-selling
    cross_sell_query = """
        SELECT tr1.PRODUCT_NUM AS PRODUCT_A, tr2.PRODUCT_NUM AS PRODUCT_B, COUNT(*) AS PAIR_COUNT
        FROM transactions tr1
        INNER JOIN transactions tr2 ON tr1.BASKET_NUM = tr2.BASKET_NUM AND tr1.PRODUCT_NUM < tr2.PRODUCT_NUM
        GROUP BY tr1.PRODUCT_NUM, tr2.PRODUCT_NUM
        ORDER BY PAIR_COUNT DESC
    """
    cross_sell_df = pd.read_sql_query(cross_sell_query, conn)

    # Query to analyze seasonal patterns
    seasonal_query = """
        SELECT WEEK_NUM, SUM(SPEND) AS TOTAL_SPEND
        FROM transactions
        GROUP BY WEEK_NUM
        ORDER BY WEEK_NUM
    """
    seasonal_df = pd.read_sql_query(seasonal_query, conn)

    # Query to analyze preferences for private vs. national brands and organic items
    brand_pref_query = """
        SELECT pr.BRAND_TY, pr.NATURAL_ORGANIC_FLAG, SUM(tr.UNITS) AS TOTAL_UNITS
        FROM transactions tr
        INNER JOIN products pr ON tr.PRODUCT_NUM = pr.PRODUCT_NUM
        GROUP BY pr.BRAND_TY, pr.NATURAL_ORGANIC_FLAG
    """
    brand_pref_df = pd.read_sql_query(brand_pref_query, conn)

    conn.close()

    # Household Size Analysis
    hh_size_analysis = data_df.groupby("HH_SIZE")["TOTAL_SPEND"].sum().reset_index()
    hh_size_fig = px.bar(hh_size_analysis, x="HH_SIZE", y="TOTAL_SPEND", title="Household Size vs Total Spend")

    # Income Range Analysis
    income_analysis = data_df.groupby("INCOME_RANGE")["TOTAL_SPEND"].sum().reset_index()
    income_fig = px.bar(income_analysis, x="INCOME_RANGE", y="TOTAL_SPEND", title="Income Range vs Total Spend")

    # Children Analysis
    children_analysis = data_df.groupby("CHILDREN")["TOTAL_SPEND"].sum().reset_index()
    children_fig = px.bar(children_analysis, x="CHILDREN", y="TOTAL_SPEND", title="Presence of Children vs Total Spend")

    # Year-over-Year Spend Analysis
    yoy_spend_fig = px.line(yoy_spend_df, x="YEAR", y="TOTAL_SPEND", title="Year-over-Year Household Spending")

    # Product Category Popularity Analysis
    category_popularity_fig = px.bar(category_popularity_df, x="DEPARTMENT", y="TOTAL_UNITS", title="Product Categories by Popularity")

    # Cross-Selling Analysis
    cross_sell_fig = px.scatter(cross_sell_df.head(20), x="PRODUCT_A", y="PRODUCT_B", size="PAIR_COUNT", title="Top 20 Product Combinations Driving Cross-Selling")

    # Seasonal Spending Patterns
    seasonal_fig = px.line(seasonal_df, x="WEEK_NUM", y="TOTAL_SPEND", title="Seasonal Spending Patterns")

    # Brand and Organic Preferences
    brand_pref_fig = px.bar(brand_pref_df, x="BRAND_TY", y="TOTAL_UNITS", color="NATURAL_ORGANIC_FLAG", 
                            barmode="group", title="Customer Preferences for Private vs. National Brands and Organic Items")

    # Convert plots to JSON for rendering
    hh_size_plot = hh_size_fig.to_html(full_html=False)
    income_plot = income_fig.to_html(full_html=False)
    children_plot = children_fig.to_html(full_html=False)
    yoy_spend_plot = yoy_spend_fig.to_html(full_html=False)
    category_popularity_plot = category_popularity_fig.to_html(full_html=False)
    cross_sell_plot = cross_sell_fig.to_html(full_html=False)
    seasonal_plot = seasonal_fig.to_html(full_html=False)
    brand_pref_plot = brand_pref_fig.to_html(full_html=False)

    return render_template('demographicsandengagement.html', 
                           hh_size_plot=hh_size_plot, 
                           income_plot=income_plot, 
                           children_plot=children_plot, 
                           yoy_spend_plot=yoy_spend_plot, 
                           category_popularity_plot=category_popularity_plot, 
                           cross_sell_plot=cross_sell_plot,
                           seasonal_plot=seasonal_plot,
                           brand_pref_plot=brand_pref_plot)

@app.route('/')
def index():
    return redirect(url_for('signup'))


@app.route('/home')
def home():
    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Query to fetch all HSHD_NUM values
    query = "SELECT DISTINCT HSHD_NUM FROM households"
    cur.execute(query)
    
    # Fetch all the HSHD_NUM values
    hshd_nums = cur.fetchall()
    
    # Close the database connection
    cur.close()
    conn.close()
    
    # Render the search page with the list of HSHD_NUM values
    return render_template('home.html', hshd_nums=hshd_nums)

@app.route('/get_dashboard_data', methods=['GET'])
def get_dashboard_data():
    hshd_num = request.args.get('hshd_num')
    
    # Connect to the database
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Execute the query with user input for HSHD_NUM
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
            h.HSHD_NUM = ?
        ORDER BY 
            h.HSHD_NUM, 
            t.BASKET_NUM, 
            t.YEAR, 
            t.WEEK_NUM, 
            t.PRODUCT_NUM, 
            p.DEPARTMENT, 
            p.COMMODITY;
    """
    
    # Execute query with the HSHD_NUM from the form
    cur.execute(query, (hshd_num,))
    rows = cur.fetchall()
    
    # Explicitly define the column names in the order of the SQL query
    columns = [
        'HSHD_NUM', 'BASKET_NUM', 'PURCHASE', 'PRODUCT_NUM', 'DEPARTMENT', 'COMMODITY', 'SPEND', 'UNITS', 'STORE_R', 'WEEK_NUM', 'YEAR',
        'L', 'AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HOMEOWNER', 'HSHD_COMPOSITION', 'HH_SIZE', 'CHILDREN'
    ]
    
    # Convert the rows to dictionaries using the correct column order
# Clean the data while creating dictionaries
    data = []
    for row in rows:
        cleaned_row = {}
        for i, value in enumerate(row):
            if value == "null":
                cleaned_row[columns[i]] = "N/A"
            if value is None:
                cleaned_row[columns[i]] = "N/A"
            elif isinstance(value, str):
                cleaned_value = value.strip()
                cleaned_row[columns[i]] = "N/A" if cleaned_value == "" else cleaned_value
            else:
                cleaned_row[columns[i]] = value
        data.append(cleaned_row) 
    # Close the database connection
    cur.close()
    conn.close()
    
    # Return the data as JSON
    return jsonify(data)

def load_data():
    try:
        conn = get_db_connection()
        
        # SQL queries
        households_query = "SELECT * FROM households"
        products_query = "SELECT * FROM products"
        transactions_query = "SELECT * FROM transactions"
        
        # Read data into pandas DataFrames
        households = pd.read_sql(households_query, conn)
        products = pd.read_sql(products_query, conn)
        transactions = pd.read_sql(transactions_query, conn)
        
        conn.close()
        return households, products, transactions
        
    except Exception as e:
        print(f"Database error: {str(e)}")
        return None, None, None

def prepare_features(households, products, transactions):
    # Clean numeric columns before merging
    numeric_columns = {
        'households': ['HH_SIZE', 'CHILDREN'],
        'transactions': ['SPEND', 'UNITS', 'WEEK_NUM', 'YEAR'],
    }

    # Clean households numeric columns
    for col in numeric_columns['households']:
        households[col] = (households[col]
            .astype(str)
            .str.strip()
            .replace('null', pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(0)
        )

    # Clean transactions numeric columns
    for col in numeric_columns['transactions']:
        transactions[col] = (transactions[col]
            .astype(str)
            .str.strip()
            .replace('null', pd.NA)
            .pipe(pd.to_numeric, errors='coerce')
            .fillna(0)
        )

    # Merge datasets
    merged_data = transactions.merge(households, on='HSHD_NUM', how='left')
    merged_data = merged_data.merge(products, on='PRODUCT_NUM', how='left')
    
    # Define categorical columns
    categorical_columns = ['AGE_RANGE', 'MARITAL', 'INCOME_RANGE', 'HOMEOWNER', 
                         'HSHD_COMPOSITION', 'DEPARTMENT', 'COMMODITY', 
                         'BRAND_TY', 'NATURAL_ORGANIC_FLAG', 'STORE_R']
    
    # Clean categorical columns
    for col in categorical_columns:
        merged_data[col] = merged_data[col].astype(str).str.strip()
        merged_data[col] = merged_data[col].replace('null', merged_data[col].mode()[0])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in categorical_columns:
        merged_data[col + '_encoded'] = le.fit_transform(merged_data[col])
    
    # Create feature matrix
    features = ['AGE_RANGE_encoded', 'MARITAL_encoded', 'INCOME_RANGE_encoded',
               'HOMEOWNER_encoded', 'HSHD_COMPOSITION_encoded', 'HH_SIZE',
               'CHILDREN', 'DEPARTMENT_encoded', 'COMMODITY_encoded',
               'BRAND_TY_encoded', 'NATURAL_ORGANIC_FLAG_encoded',
               'STORE_R_encoded', 'WEEK_NUM', 'YEAR']
    
    X = merged_data[features]
    y = merged_data['SPEND']
    
    return X, y, merged_data

@app.route('/train_model')
def train_model():
    # Load data from Azure SQL
    households, products, transactions = load_data()
    
    if households is None or products is None or transactions is None:
        return jsonify({'error': 'Failed to load data from database'})
    
    # Prepare features
    X, y, merged_data = prepare_features(households, products, transactions)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Initialize and train model
    gb_model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    
    gb_model.fit(X_train, y_train)
    # Save model
    joblib.dump(gb_model, 'gb_model.pkl')
    
    # Make predictions
    y_pred = gb_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    
    # Save model with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f'gb_model_{timestamp}.pkl'
    joblib.dump(gb_model, model_filename)
    
    # Define feature names
    feature_names = [
        'Age Range', 'Marital Status', 'Income Range',
        'Homeowner Status', 'Household Composition', 'Household Size',
        'Children', 'Department', 'Commodity',
        'Brand Type', 'Natural Organic Flag',
        'Store Region', 'Week Number', 'Year'
    ]

    # Create feature importance with proper labels
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': gb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Store results in database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create table for model metrics if it doesn't exist
        cursor.execute("""
            IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='model_metrics' AND xtype='U')
            CREATE TABLE model_metrics (
                id INT IDENTITY(1,1) PRIMARY KEY,
                model_filename VARCHAR(255),
                mse FLOAT,
                timestamp DATETIME
            )
        """)
        
        # Insert metrics
        cursor.execute("""
            INSERT INTO model_metrics (model_filename, mse, timestamp)
            VALUES (?, ?, GETDATE())
        """, (model_filename, mse))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Error storing metrics: {str(e)}")
    
    return jsonify({
        'mse': mse,
        'feature_importance': {
            'features': feature_names,
            'importance': gb_model.feature_importances_.tolist(),
            'sorted_indices': feature_importance.index.tolist()
        },
        'model_status': 'trained',
        'model_filename': model_filename
    })

@app.route('/predict')
def predict():
    gb_model = joblib.load('gb_model.pkl')
    
    # Load new data
    households, products, transactions = load_data()
    X, _, merged_data = prepare_features(households, products, transactions)
    
    # Make predictions
    predictions = gb_model.predict(X)
    
    # Get top product combinations with more details
    merged_data['predicted_spend'] = predictions
    top_combinations = (merged_data.groupby(['DEPARTMENT', 'COMMODITY'])
        .agg({
            'predicted_spend': 'mean',
            'PRODUCT_NUM': 'count'
        })
        .sort_values('predicted_spend', ascending=False)
        .head(10)
    )
    
    recommendations = {
        'department_commodity': top_combinations.index.tolist(),
        'predicted_spend': top_combinations['predicted_spend'].round(2).tolist(),
        'product_count': top_combinations['PRODUCT_NUM'].tolist()
    }
    
    return jsonify(recommendations)

@app.route('/get_analytics')
def get_analytics():
    # Load data
    households, products, transactions = load_data()
    X, y, merged_data = prepare_features(households, products, transactions)
    
    # Define the features list
    features = [
        'login_frequency',
        'session_duration',
        'interaction_count',
        'purchase_value',
        'support_tickets',
        'email_response_rate',
        'customer_status'
    ]
    
    # Add these columns to merged_data if they don't exist
    for feature in features:
        if feature not in merged_data.columns:
            # Generate dummy data for demonstration
            if feature == 'customer_status':
                merged_data[feature] = np.random.choice([0, 1], size=len(merged_data))
            else:
                merged_data[feature] = np.random.random(size=len(merged_data))
    
    # Calculate correlation matrix
    correlation_matrix = merged_data[features].corr()
    
    # Calculate feature importance
    feature_importance = correlation_matrix['customer_status'].sort_values(ascending=True)
    
    return jsonify({
        'correlation_matrix': {
            'z': correlation_matrix.values.tolist(),
            'x': correlation_matrix.columns.tolist(),
            'y': correlation_matrix.columns.tolist(),
            'text': correlation_matrix.round(2).values.tolist()
        },
        'feature_importance': {
            'features': feature_importance.index.tolist(),
            'importance': feature_importance.values.tolist()
        }
    })

@app.route('/analyze_transactions')
def analyze_transactions():
    households, products, transactions = load_data()
    
    # Merge datasets
    merged_data = transactions.merge(households, on='HSHD_NUM', how='left')
    merged_data = merged_data.merge(products, on='PRODUCT_NUM', how='left')
    
    # Calculate key metrics
    metrics = {
        'transaction_frequency': merged_data.groupby('HSHD_NUM')['BASKET_NUM'].count(),
        'average_spend': merged_data.groupby('HSHD_NUM')['SPEND'].mean(),
        'product_diversity': merged_data.groupby('HSHD_NUM')['COMMODITY'].nunique()
    }
    
    # Create engagement score
    engagement_data = pd.DataFrame(metrics)
    engagement_data['engagement_score'] = (
        (engagement_data['transaction_frequency'] / engagement_data['transaction_frequency'].max()) +
        (engagement_data['average_spend'] / engagement_data['average_spend'].max()) +
        (engagement_data['product_diversity'] / engagement_data['product_diversity'].max())
    ) / 3
    
    # Identify at-risk customers (bottom 20% engagement)
    risk_threshold = engagement_data['engagement_score'].quantile(0.2)
    at_risk = engagement_data[engagement_data['engagement_score'] < risk_threshold]
    
    return jsonify({
        'engagement_metrics': {
            'scores': engagement_data['engagement_score'].tolist(),
            'households': engagement_data.index.tolist(),
            'risk_threshold': float(risk_threshold)
        },
        'at_risk_count': len(at_risk),
        'total_customers': len(engagement_data)
    })

@app.route('/retention_analysis')
def retention_analysis():
    households, products, transactions = load_data()
    
    # Calculate purchase gaps
    transactions['PURCHASE_DATE'] = pd.to_datetime(transactions['WEEK_NUM'].astype(str) + 
                                                 transactions['YEAR'].astype(str), format='%W%Y')
    
    purchase_gaps = transactions.groupby('HSHD_NUM')['PURCHASE_DATE'].agg(['min', 'max'])
    purchase_gaps['days_active'] = (purchase_gaps['max'] - purchase_gaps['min']).dt.days
    
    # Calculate customer value
    customer_value = transactions.groupby('HSHD_NUM').agg({
        'SPEND': 'sum',
        'BASKET_NUM': 'count'
    })
    
    return jsonify({
        'customer_lifetime': purchase_gaps['days_active'].tolist(),
        'customer_value': customer_value['SPEND'].tolist(),
        'basket_count': customer_value['BASKET_NUM'].tolist()
    })

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get form data
        username = request.form['username']
        password = request.form['password']
        email = request.form['email']

        # Validate inputs (basic example, extend as needed)
        if not username or not password or not email:
            return "All fields are required!", 400

        # Insert data into the database
        conn = get_db_connection()
        cur = conn.cursor()
        query = "INSERT INTO Users (Username, Password, Email) VALUES (?, ?, ?)"
        cur.execute(query, (username, password, email))
        conn.commit()
        cur.close()
        conn.close()

        return redirect(url_for('home'))

    # Render the signup form
    return render_template('signup.html')

@app.route('/predictproducts')
def predictproducts():
    return render_template('predict.html')

@app.route('/logout')
def logout():
    # Add any session clearing logic here if needed
    return redirect(url_for('signup'))

if __name__ == '__main__':
    app.run(debug=True)
