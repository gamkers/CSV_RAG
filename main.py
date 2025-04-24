import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
import time
import base64
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
from datetime import datetime
import json
import re

# Page configuration
st.set_page_config(layout="wide", page_title="ClearVision Analytics", page_icon="📊")

# Initialize session state variables if they don't exist
if 'history' not in st.session_state:
    st.session_state.history = []
if 'dataframes' not in st.session_state:
    st.session_state.dataframes = {}
if 'current_df_name' not in st.session_state:
    st.session_state.current_df_name = None

# Custom CSS
st.markdown("""
<style>
    .main .block-container {padding-top: 2rem;}
    .stButton>button {width: 100%;}
    .stDataFrame {border-radius: 5px; overflow: hidden;}
    div[data-testid="stHeader"] {background-color: #f0f2f6;}
    .css-1d391kg {padding-top: 3rem;}
    .report-header {background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin-bottom: 1rem;}
    div.stAlert > div {padding-top: 15px; padding-bottom: 15px;}
</style>
""", unsafe_allow_html=True)

# Initialize ChatGroq with API key from environment
@st.cache_resource
def get_llm():
    return ChatGroq(model_name='llama3-70b-8192', api_key="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")

llm = get_llm()

# Enhanced file reading with multiple options
@st.cache_data
def read_file(uploaded_file, **kwargs):
    """
    Enhanced file reading with more options and better error handling
    """
    try:
        file_type = uploaded_file.name.split('.')[-1].lower()
        
        if file_type == 'csv':
            # Get delimiter from kwargs or use default
            delimiter = kwargs.get('delimiter', ',')
            encoding = kwargs.get('encoding', 'utf-8')
            skiprows = kwargs.get('skiprows', 0)
            
            return pd.read_csv(
                uploaded_file, 
                delimiter=delimiter,
                encoding=encoding,
                low_memory=True,
                skiprows=skiprows,
                on_bad_lines='warn'
            )
        elif file_type in ['xlsx', 'xls']:
            sheet_name = kwargs.get('sheet_name', 0)
            skiprows = kwargs.get('skiprows', 0)
            
            return pd.read_excel(
                uploaded_file, 
                engine='openpyxl',
                sheet_name=sheet_name,
                skiprows=skiprows
            )
        elif file_type == 'json':
            return pd.read_json(uploaded_file)
        elif file_type == 'parquet':
            return pd.read_parquet(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, JSON, or Parquet file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

# Function to get basic dataframe info
def get_df_info(df):
    buffer = StringIO()
    df.info(buf=buffer, memory_usage='deep')
    return buffer.getvalue()

# Function to detect column data types
def detect_column_types(df):
    type_map = {}
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            if any(df[col].apply(lambda x: isinstance(x, float))):
                type_map[col] = "Numeric (Float)"
            else:
                type_map[col] = "Numeric (Integer)"
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            type_map[col] = "Date/Time"
        elif pd.api.types.is_categorical_dtype(df[col]):
            type_map[col] = "Categorical"
        else:
            # Check if column might be a date
            try:
                pd.to_datetime(df[col].iloc[0:100])
                type_map[col] = "Possible Date/Time"
            except:
                # Count unique values to identify potential categories
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.1:  # If less than 10% unique values
                    type_map[col] = "Categorical"
                else:
                    type_map[col] = "Text"
    
    return pd.DataFrame({"Column": type_map.keys(), "Data Type": type_map.values()})

# Function to generate data quality report
def data_quality_report(df):
    total_rows = len(df)
    quality_report = []
    
    for column in df.columns:
        missing_values = df[column].isna().sum()
        missing_percentage = (missing_values / total_rows) * 100
        unique_values = df[column].nunique()
        unique_percentage = (unique_values / total_rows) * 100
        
        quality_report.append({
            "Column": column,
            "Missing Values": missing_values,
            "Missing (%)": round(missing_percentage, 2),
            "Unique Values": unique_values,
            "Unique (%)": round(unique_percentage, 2),
            "Data Type": df[column].dtype
        })
    
    return pd.DataFrame(quality_report)

# Generate a downloadable link for any dataframe
def get_download_link(df, filename, link_text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{link_text}</a>'
    return href

# Function to create quick visualizations
def create_quick_viz(df, viz_type, x_col=None, y_col=None, color_col=None):
    try:
        if viz_type == "Histogram":
            fig = px.histogram(df, x=x_col, title=f"Histogram of {x_col}")
        elif viz_type == "Bar Chart":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title=f"Bar Chart: {y_col} by {x_col}")
        elif viz_type == "Scatter Plot":
            fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {y_col} vs {x_col}")
        elif viz_type == "Box Plot":
            fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box Plot: {y_col} by {x_col}")
        elif viz_type == "Line Chart":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Line Chart: {y_col} over {x_col}")
        elif viz_type == "Pie Chart":
            # Get value counts and use them for the pie chart
            value_counts = df[x_col].value_counts().reset_index()
            fig = px.pie(value_counts, values=x_col, names="index", title=f"Pie Chart: Distribution of {x_col}")
        elif viz_type == "Heatmap":
            # Calculate correlation matrix
            corr_matrix = df.select_dtypes(include=['number']).corr()
            fig = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale='Viridis'
            ))
            fig.update_layout(title="Correlation Heatmap")
        else:
            st.error("Unsupported visualization type")
            return None
        
        fig.update_layout(height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating visualization: {str(e)}")
        return None

# Data transformation functions
def transform_data(df, operation):
    """Apply various data transformations."""
    
    try:
        if operation == "Drop NA Rows":
            return df.dropna(), "Dropped rows with any missing values"
        
        elif operation == "Drop NA Columns":
            return df.dropna(axis=1), "Dropped columns with any missing values"
        
        elif operation == "Fill NA with Mean":
            numeric_cols = df.select_dtypes(include=['number']).columns
            df_new = df.copy()
            for col in numeric_cols:
                df_new[col] = df_new[col].fillna(df_new[col].mean())
            return df_new, "Filled missing values in numeric columns with mean"
        
        elif operation == "Fill NA with Median":
            numeric_cols = df.select_dtypes(include=['number']).columns
            df_new = df.copy()
            for col in numeric_cols:
                df_new[col] = df_new[col].fillna(df_new[col].median())
            return df_new, "Filled missing values in numeric columns with median"
            
        elif operation == "Fill NA with Zero":
            df_new = df.copy()
            return df_new.fillna(0), "Filled all missing values with zero"
            
        elif operation == "Convert to Datetime":
            col = st.selectbox("Select column to convert to datetime:", df.columns)
            df_new = df.copy()
            df_new[col] = pd.to_datetime(df_new[col], errors='coerce')
            return df_new, f"Converted {col} to datetime format"
            
        elif operation == "Normalize Column":
            col = st.selectbox("Select column to normalize:", df.select_dtypes(include=['number']).columns)
            df_new = df.copy()
            df_new[f"{col}_normalized"] = (df_new[col] - df_new[col].min()) / (df_new[col].max() - df_new[col].min())
            return df_new, f"Added normalized version of {col}"
            
        elif operation == "Add Column":
            col_name = st.text_input("New column name:")
            expression = st.text_input("Expression (use df for dataframe):")
            if col_name and expression:
                df_new = df.copy()
                try:
                    # Safely evaluate the expression
                    local_dict = {"df": df_new, "np": np, "pd": pd}
                    df_new[col_name] = eval(expression, {"__builtins__": {}}, local_dict)
                    return df_new, f"Added new column '{col_name}'"
                except Exception as e:
                    return df, f"Error adding column: {str(e)}"
            return df, "No changes made"
            
        elif operation == "Drop Column":
            cols_to_drop = st.multiselect("Select columns to drop:", df.columns)
            if cols_to_drop:
                df_new = df.copy()
                df_new = df_new.drop(columns=cols_to_drop)
                return df_new, f"Dropped columns: {', '.join(cols_to_drop)}"
            return df, "No changes made"
            
        elif operation == "Filter Rows":
            col = st.selectbox("Select column to filter on:", df.columns)
            condition = st.selectbox("Condition:", ["equals", "not equals", "greater than", "less than", "contains"])
            value = st.text_input("Value:")
            
            if value:
                df_new = df.copy()
                try:
                    if condition == "equals":
                        mask = df_new[col] == value
                    elif condition == "not equals":
                        mask = df_new[col] != value
                    elif condition == "greater than":
                        mask = df_new[col] > float(value)
                    elif condition == "less than":
                        mask = df_new[col] < float(value)
                    elif condition == "contains":
                        mask = df_new[col].astype(str).str.contains(value)
                    
                    df_filtered = df_new[mask]
                    return df_filtered, f"Filtered to {len(df_filtered)} rows where {col} {condition} {value}"
                except Exception as e:
                    return df, f"Error filtering rows: {str(e)}"
            return df, "No changes made"
            
        elif operation == "Sort Values":
            col = st.selectbox("Select column to sort by:", df.columns)
            ascending = st.checkbox("Ascending", value=True)
            df_new = df.copy().sort_values(by=col, ascending=ascending)
            return df_new, f"Sorted by {col} ({'ascending' if ascending else 'descending'})"
            
        elif operation == "Group By":
            group_cols = st.multiselect("Select columns to group by:", df.columns)
            agg_col = st.selectbox("Select column to aggregate:", df.select_dtypes(include=['number']).columns)
            agg_func = st.selectbox("Select aggregation function:", ["mean", "sum", "count", "min", "max"])
            
            if group_cols and agg_col:
                df_new = df.copy()
                result = df_new.groupby(group_cols)[agg_col].agg(agg_func).reset_index()
                return result, f"Grouped by {', '.join(group_cols)} and calculated {agg_func} of {agg_col}"
            return df, "No changes made"
            
        elif operation == "One-Hot Encode":
            col = st.selectbox("Select categorical column to encode:", df.select_dtypes(exclude=['number']).columns)
            df_new = df.copy()
            encoded = pd.get_dummies(df_new[col], prefix=col)
            df_new = pd.concat([df_new, encoded], axis=1)
            return df_new, f"One-hot encoded column {col}"
            
        elif operation == "Bin Values":
            col = st.selectbox("Select numeric column to bin:", df.select_dtypes(include=['number']).columns)
            num_bins = st.slider("Number of bins:", 2, 20, 5)
            df_new = df.copy()
            df_new[f"{col}_binned"] = pd.cut(df_new[col], bins=num_bins, labels=[f"Bin {i+1}" for i in range(num_bins)])
            return df_new, f"Created {num_bins} bins for column {col}"
    
        else:
            return df, "No transformation applied"
    
    except Exception as e:
        st.error(f"Error during transformation: {str(e)}")
        return df, f"Error: {str(e)}"

# Process data with caching when possible
@st.cache_data(ttl=600)  # Cache results for 10 minutes
def process_data(df, query):
    """
    Process the DataFrame using SmartDataframe and ChatGroq with progress tracking.
    """
    # Create a copy to avoid modifying the original
    data = df.copy()
    
    # For very large datasets, consider sampling
    row_count = len(data)
    if row_count > 500000:
        st.warning(f"Your dataset is very large ({row_count:,} rows). Processing a sample of 500,000 rows.")
        data = data.sample(500000, random_state=42)
    
    # Initialize SmartDataframe with optimal configurations
    smart_df = SmartDataframe(
        data, 
        config={
            'llm': llm,
            'open_charts': False,  # Don't open charts automatically
            'save_charts': True,   # Save charts for display
            'verbose': True,       # Show processing details
            'conversational': True # Enable follow-up questions
        }
    )
    
    # Process query
    try:
        return smart_df.chat(query)
    except Exception as e:
        return f"Error processing query: {str(e)}"

# Save insights to a report
def save_to_report(title, content):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if 'report' not in st.session_state:
        st.session_state.report = []
    
    st.session_state.report.append({
        "title": title,
        "content": content,
        "timestamp": timestamp
    })
    
    st.success(f"Saved '{title}' to report")

# Export report to HTML
def export_report_html():
    if 'report' not in st.session_state or not st.session_state.report:
        return None
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ClearVision Analytics Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .report-item { margin-bottom: 30px; border-bottom: 1px solid #eee; padding-bottom: 20px; }
            .timestamp { color: #888; font-size: 0.8em; }
            h1 { color: #2c3e50; }
            h2 { color: #3498db; }
            pre { background-color: #f8f9fa; padding: 10px; border-radius: 4px; overflow: auto; }
            img { max-width: 100%; height: auto; }
        </style>
    </head>
    <body>
        <h1>ClearVision Analytics Report</h1>
        <p>Generated on: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + """</p>
    """
    
    for item in st.session_state.report:
        html_content += f"""
        <div class="report-item">
            <h2>{item['title']}</h2>
            <p class="timestamp">Added on: {item['timestamp']}</p>
            <div class="content">
                {item['content']}
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    return html_content

# Function for automated insights
def generate_automated_insights(df):
    insights = []
    
    # Check for missing values
    missing_vals = df.isna().sum().sum()
    if missing_vals > 0:
        insights.append(f"Dataset contains {missing_vals:,} missing values across all columns.")
    
    # Look for potential outliers in numeric columns
    for col in df.select_dtypes(include=['number']).columns[:5]:  # Limit to first 5 numeric columns
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))][col].count()
            if outliers > 0:
                insights.append(f"Column '{col}' has {outliers} potential outliers (using IQR method).")
        except:
            pass
    
    # Check for highly correlated numeric features
    try:
        corr_matrix = df.select_dtypes(include=['number']).corr()
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr.append(f"Strong correlation ({corr_matrix.iloc[i, j]:.2f}) between '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}'")
        
        if high_corr:
            insights.extend(high_corr[:3])  # Limit to top 3 correlations
    except:
        pass
    
    # Check for columns with low variance (potentially useless features)
    for col in df.select_dtypes(include=['number']).columns:
        try:
            if df[col].var() < 0.01:
                insights.append(f"Column '{col}' has very low variance, might not be useful for analysis.")
        except:
            pass
    
    # Check for unique identifiers
    for col in df.columns:
        if df[col].nunique() == len(df):
            insights.append(f"Column '{col}' contains unique values for each row - likely an ID column.")
    
    # Return findings with markdown formatting
    return "\n\n".join([f"* {insight}" for insight in insights])

# Main application
def main():
    # Set up the page structure
    st.title('ClearVision Analytics Platform')
    dark_purple_theme = """
<style>
    /* Main theme colors */
    /* ClearVision Analytics Platform - Main Stylesheet */

/* Global Variables */
:root {
  --primary-color: #4e73df;
  --primary-light: #6f8df7;
  --primary-dark: #3a56b0;
  --secondary-color: #1cc88a;
  --danger-color: #e74a3b;
  --warning-color: #f6c23e;
  --info-color: #36b9cc;
  --dark-color: #5a5c69;
  --light-color: #f8f9fc;
  --white-color: #ffffff;
  --gray-100: #f8f9fc;
  --gray-200: #eaecf4;
  --gray-300: #dddfeb;
  --gray-400: #d1d3e2;
  --gray-500: #b7b9cc;
  --gray-600: #858796;
  --gray-700: #6e707e;
  --gray-800: #5a5c69;
  --gray-900: #3a3b45;
  --shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

/* Base styles */
body {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
  background-color: var(--gray-100);
  color: var(--gray-800);
}

/* Main layout */
.main {
  background-color: var(--white-color);
  border-radius: 10px;
  box-shadow: var(--shadow);
  margin: 1rem;
}

/* Header styles */
h1, h2, h3, h4, h5, h6 {
  font-weight: 600;
  color: var(--gray-900);
  margin-bottom: 1rem;
}

h1 {
  font-size: 2rem;
  padding: 1rem 0;
  border-bottom: 1px solid var(--gray-300);
}

h2 {
  font-size: 1.75rem;
}

h3 {
  font-size: 1.5rem;
}

/* Sidebar styling */
.sidebar {
  background-color: var(--white-color);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 1.5rem;
}

.sidebar .stButton > button {
  width: 100%;
  background-color: var(--primary-color);
  border: none;
  color: var(--white-color);
  padding: 0.6rem 1rem;
  font-weight: 500;
  border-radius: 6px;
  margin-bottom: 0.75rem;
  transition: background-color 0.2s ease;
}

.sidebar .stButton > button:hover {
  background-color: var(--primary-dark);
}

.sidebar h2, .sidebar h3, .sidebar h4 {
  margin-top: 1.5rem;
  margin-bottom: 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 1px solid var(--gray-200);
}

.sidebar .stFileUploader {
  background-color: var(--gray-100);
  border-radius: 8px;
  padding: 0.5rem;
  margin-bottom: 1rem;
}

/* Expander styling */
.streamlit-expanderHeader {
  background-color: var(--gray-100);
  border-radius: 8px;
  font-weight: 500;
}

.streamlit-expanderContent {
  border-left: 2px solid var(--gray-300);
  padding-left: 1rem;
  margin-left: 0.5rem;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
  gap: 2px;
  background-color: var(--gray-200);
  border-radius: 10px;
  padding: 4px;
}

.stTabs [data-baseweb="tab"] {
  height: 42px;
  border-radius: 8px;
  padding: 0 16px;
  font-weight: 500;
}

.stTabs [aria-selected="true"] {
  background-color: var(--white-color);
  color: var(--primary-color);
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}

/* Buttons styling */
.stButton > button {
  background-color: var(--primary-color);
  color: var(--white-color);
  border: none;
  padding: 0.5rem 1rem;
  border-radius: 6px;
  font-weight: 500;
  transition: all 0.2s ease;
}

.stButton > button:hover {
  background-color: var(--primary-dark);
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
}

/* Input widgets */
.stTextInput > div > div > input, 
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea {
  border-radius: 6px;
  border: 1px solid var(--gray-400);
  padding: 0.75rem;
  box-shadow: none;
  transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.stTextInput > div > div > input:focus, 
.stNumberInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--primary-color);
  box-shadow: 0 0 0 2px rgba(78, 115, 223, 0.2);
}

/* Select boxes */
.stSelectbox > div > div {
  border-radius: 6px;
}

/* Dataframe/table styling */
.stDataFrame {
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

.stDataFrame table {
  width: 100%;
}

.stDataFrame thead {
  background-color: var(--gray-200);
}

.stDataFrame th {
  padding: 0.75rem 1rem;
  font-weight: 600;
  text-align: left;
  border-bottom: 2px solid var(--gray-300);
}

.stDataFrame td {
  padding: 0.75rem 1rem;
  border-bottom: 1px solid var(--gray-200);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.stDataFrame tr:nth-child(even) {
  background-color: var(--gray-100);
}

.stDataFrame tr:hover {
  background-color: rgba(78, 115, 223, 0.05);
}

/* Metrics styling */
.stMetric {
  background-color: var(--white-color);
  border-radius: 10px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
  padding: 1rem;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.stMetric:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.stMetric .metric-label {
  font-weight: 600;
  color: var(--gray-700);
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.stMetric .metric-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--gray-900);
}

/* Info/alert boxes */
.stAlert {
  border-radius: 8px;
  border-width: 1px;
  padding: 1rem;
}

.stAlert.success {
  background-color: rgba(28, 200, 138, 0.1);
  border-color: var(--secondary-color);
  color: var(--secondary-color);
}

.stAlert.warning {
  background-color: rgba(246, 194, 62, 0.1);
  border-color: var(--warning-color);
  color: var(--warning-color);
}

.stAlert.error {
  background-color: rgba(231, 74, 59, 0.1);
  border-color: var(--danger-color);
  color: var(--danger-color);
}

.stAlert.info {
  background-color: rgba(54, 185, 204, 0.1);
  border-color: var(--info-color);
  color: var(--info-color);
}

/* Spinner */
.stSpinner > div {
  border-color: var(--primary-color) transparent transparent transparent;
}

/* Progress bars */
.stProgress > div > div {
  background-color: var(--primary-color);
}

/* Chart visualization container */
.chart-container {
  background-color: var(--white-color);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 1rem;
  margin-bottom: 1.5rem;
}

/* Data Explorer section */
.data-explorer-container {
  background-color: var(--white-color);
  border-radius: 10px;
  box-shadow: var(--shadow);
  padding: 1.5rem;
  margin-bottom: 2rem;
}

.column-stats {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
  gap: 1rem;
  margin-top: 1rem;
}

.stat-card {
  background-color: var(--gray-100);
  border-radius: 8px;
  padding: 1rem;
  display: flex;
  flex-direction: column;
}

.stat-card .stat-label {
  font-size: 0.85rem;
  color: var(--gray-600);
  margin-bottom: 0.25rem;
}

.stat-card .stat-value {
  font-size: 1.25rem;
  font-weight: 600;
  color: var(--gray-900);
}

/* Query section styling */
.query-container {
  background-color: var(--white-color);
  border-radius: 10px;
  padding: 1.5rem;
  box-shadow: var(--shadow);
  margin-bottom: 2rem;
}

.example-query {
  display: inline-block;
  background-color: var(--gray-100);
  border-radius: 20px;
  padding: 0.5rem 1rem;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
  font-size: 0.9rem;
  color: var(--gray-800);
  cursor: pointer;
  transition: background-color 0.2s ease;
}

.example-query:hover {
  background-color: var(--gray-300);
}

.result-container {
  background-color: var(--gray-100);
  border-radius: 8px;
  padding: 1.5rem;
  margin-top: 1rem;
}

/* Visualization studio */
.viz-controls {
  background-color: var(--gray-100);
  border-radius: 10px;
  padding: 1.5rem;
}

.viz-preview {
  background-color: var(--white-color);
  border-radius: 10px;
  padding: 1rem;
  box-shadow: var(--shadow);
}

/* Welcome screen */
.welcome-container {
  text-align: center;
  max-width: 800px;
  margin: 3rem auto;
  padding: 2rem;
  background-color: var(--white-color);
  border-radius: 12px;
  box-shadow: var(--shadow);
}

.feature-card {
  background-color: var(--gray-100);
  border-radius: 10px;
  padding: 1.5rem;
  text-align: center;
  transition: transform 0.2s ease;
  height: 100%;
}

.feature-card:hover {
  transform: translateY(-5px);
  box-shadow: var(--shadow);
}

.feature-card h4 {
  color: var(--primary-color);
  margin-bottom: 1rem;
}

.feature-icon {
  font-size: 2.5rem;
  color: var(--primary-color);
  margin-bottom: 1.5rem;
}

/* Report builder */
.report-item {
  background-color: var(--white-color);
  border-radius: 8px;
  padding: 1rem;
  margin-bottom: 1rem;
  border-left: 4px solid var(--primary-color);
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
}

.report-actions {
  margin-top: 1rem;
  display: flex;
  gap: 0.5rem;
}

/* Download links styling */
a.download-link {
  display: inline-block;
  background-color: var(--primary-color);
  color: var(--white-color);
  padding: 0.75rem 1.5rem;
  border-radius: 6px;
  text-decoration: none;
  font-weight: 500;
  margin-top: 1rem;
  transition: background-color 0.2s ease;
}

a.download-link:hover {
  background-color: var(--primary-dark);
}

/* Special elements */
.highlight-box {
  background-color: rgba(78, 115, 223, 0.05);
  border-left: 4px solid var(--primary-color);
  padding: 1rem;
  margin: 1rem 0;
  border-radius: 0 8px 8px 0;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .column-stats {
    grid-template-columns: 1fr;
  }
  
  .sidebar {
    margin-bottom: 1.5rem;
  }
}

/* Custom scrollbar */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--gray-200);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb {
  background: var(--gray-500);
  border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
  background: var(--gray-600);
}
"""
    st.markdown(dark_purple_theme, unsafe_allow_html=True)
    # Side panel for data management and settings
    with st.sidebar:
        st.header("Data Management")
        
        # File upload section
        st.subheader("Upload Data")
        uploaded_file = st.file_uploader("Upload a file", type=["csv", "xlsx", "xls", "json", "parquet"])
        
        # Advanced import options (expandable)
        with st.expander("Advanced Import Options"):
            delimiter = st.text_input("CSV Delimiter", value=",")
            encoding = st.selectbox("Encoding", options=["utf-8", "latin1", "cp1252", "ISO-8859-1"])
            skiprows = st.number_input("Skip Rows", value=0, min_value=0)
            sheet_name = st.text_input("Excel Sheet Name (leave blank for first sheet)")
        
        # Loading data
        if uploaded_file is not None:
            with st.spinner('Reading file...'):
                start_time = time.time()
                
                # Prepare kwargs for file reading
                kwargs = {
                    'delimiter': delimiter,
                    'encoding': encoding,
                    'skiprows': skiprows
                }
                
                if sheet_name:
                    kwargs['sheet_name'] = sheet_name
                
                df = read_file(uploaded_file, **kwargs)
                load_time = time.time() - start_time
            
            if df is not None:
                # Save to session state with a unique name
                df_name = uploaded_file.name.split('.')[0]
                if df_name in st.session_state.dataframes:
                    df_name = f"{df_name}_{len(st.session_state.dataframes)}"
                
                st.session_state.dataframes[df_name] = df
                st.session_state.current_df_name = df_name
                
                st.success(f"'{df_name}' loaded successfully in {load_time:.2f} seconds!")
                st.write(f"Rows: {len(df):,} | Columns: {len(df.columns):,}")
        
        # Dataset selector (when multiple datasets are loaded)
        if st.session_state.dataframes:
            st.subheader("Select Dataset")
            selected_df = st.selectbox(
                "Choose a dataset:", 
                options=list(st.session_state.dataframes.keys()),
                index=list(st.session_state.dataframes.keys()).index(st.session_state.current_df_name) 
                    if st.session_state.current_df_name in st.session_state.dataframes else 0
            )
            
            # Update current dataframe if selection changed
            if selected_df != st.session_state.current_df_name:
                st.session_state.current_df_name = selected_df
                st.experimental_rerun()
        
        # Add sample data option
        st.subheader("Sample Datasets")
        sample_data = st.selectbox(
            "Load a sample dataset:", 
            ["None", "Iris Flowers", "Titanic Passengers", "Boston Housing", "Wine Quality"]
        )
        
        if sample_data != "None" and st.button("Load Sample"):
            with st.spinner("Loading sample dataset..."):
                if sample_data == "Iris Flowers":
                    from sklearn.datasets import load_iris
                    data = load_iris()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['species'] = [data.target_names[i] for i in data.target]
                    
                elif sample_data == "Titanic Passengers":
                    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                    df = pd.read_csv(url)
                    
                elif sample_data == "Boston Housing":
                    from sklearn.datasets import load_boston
                    data = load_boston()
                    df = pd.DataFrame(data.data, columns=data.feature_names)
                    df['PRICE'] = data.target
                    
                elif sample_data == "Wine Quality":
                    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
                    df = pd.read_csv(url, delimiter=";")
                
                # Save to session state
                df_name = sample_data.replace(" ", "_").lower()
                st.session_state.dataframes[df_name] = df
                st.session_state.current_df_name = df_name
                st.success(f"Loaded {sample_data} dataset")
                st.experimental_rerun()
    
    # Main content area
    if st.session_state.current_df_name:
        # Get current dataframe
        df = st.session_state.dataframes[st.session_state.current_df_name]
        
        # Main sections using tabs
        main_tabs = st.tabs([
            "Data Explorer", 
            "Smart Query", 
            "Visualization Studio", 
            "Data Processor", 
            "Insights & Report"
        ])
        
        # Tab 1: Data Explorer
        with main_tabs[0]:
            st.header("Data Explorer")
            
            explorer_tabs = st.tabs(["Dataset Preview", "Data Profile", "Column Stats", "Data Quality"])
            
            # Dataset Preview subtab
            with explorer_tabs[0]:
                # Preview with pagination options
                row_count = len(df)
                max_pages = max(1, row_count // 100)
                
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.subheader(f"Preview: {st.session_state.current_df_name}")
                with col2:
                    page = st.number_input("Page", min_value=1, max_value=max_pages, value=1)
                
                start_idx = (page - 1) * 100
                end_idx = min(start_idx + 100, row_count)
                
                st.dataframe(df.iloc[start_idx:end_idx], use_container_width=True)
                st.caption(f"Showing rows {start_idx}-{end_idx} of {row_count:,}")
                
                # Export options
                st.markdown(get_download_link(df, f"{st.session_state.current_df_name}.csv", "📥 Download as CSV"), unsafe_allow_html=True)
            
            # Data Profile subtab
            with explorer_tabs[1]:
                st.subheader("Data Profile")
                
                # Basic dataset info
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Rows", f"{len(df):,}")
                col2.metric("Columns", len(df.columns))
                col3.metric("Missing Values", f"{df.isna().sum().sum():,}")
                
                # Memory usage
                memory_usage = df.memory_usage(deep=True).sum() / (1024 * 1024)
                col4.metric("Memory Usage", f"{memory_usage:.2f} MB")
                
                # Column data types
                st.subheader("Column Data Types")
                column_types = detect_column_types(df)
                st.dataframe(column_types, use_container_width=True)
                
                # Display full info
                with st.expander("Detailed DataFrame Info"):
                    st.text(get_df_info(df))
            
            # Column Stats subtab
            with explorer_tabs[2]:
                st.subheader("Column Statistics")
                
                # Column selector
                col_selected = st.selectbox("Select Column", df.columns)
                
                if col_selected:
                    col1, col2 = st.columns(2)
                    
                    # Basic stats for the selected column
                    with col1:
                        st.subheader(f"Statistics: {col_selected}")
                        
                        if pd.api.types.is_numeric_dtype(df[col_selected]):
                            stats = df[col_selected].describe()
                            stats_df = pd.DataFrame({
                                'Statistic': stats.index,
                                'Value': stats.values
                            })
                            st.dataframe(stats_df, use_container_width=True)
                            
                            # Additional numeric stats
                            st.metric("Skewness", f"{df[col_selected].skew():.4f}")
                            st.metric("Kurtosis", f"{df[col_selected].kurtosis():.4f}")
                        else:
                            # For non-numeric columns
                            value_counts = df[col_selected].value_counts().head(10)
                            st.write("Top Values:")
                            st.dataframe(value_counts.reset_index().rename(
                                columns={"index": col_selected, col_selected: "Count"}
                            ), use_container_width=True)
                            
                            st.metric("Unique Values", df[col_selected].nunique())
                            st.metric("Most Common", df[col_selected].mode()[0] if not df[col_selected].mode().empty else "None")
                    
                    # Visualization for the column
                    with col2:
                        st.subheader(f"Visualization: {col_selected}")
                        
                        if pd.api.types.is_numeric_dtype(df[col_selected]):
                            # Histogram for numeric data
                            fig = px.histogram(df, x=col_selected, nbins=30)
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Box plot for numeric data
                            fig = px.box(df, y=col_selected)
                            fig.update_layout(height=300)
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            # Bar chart for categorical data
                            value_counts = df[col_selected].value_counts().head(15)
                            fig = px.bar(
                                x=value_counts.index, 
                                y=value_counts.values,
                                labels={'x': col_selected, 'y': 'Count'}
                            )
                            fig.update_layout(height=400)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Missing values analysis
                    st.subheader("Missing Values Analysis")
                    missing_count = df[col_selected].isna().sum()
                    missing_percent = (missing_count / len(df)) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Missing Count", missing_count)
                    col2.metric("Missing Percentage", f"{missing_percent:.2f}%")
            
            # Data Quality subtab
            with explorer_tabs[3]:
                st.subheader("Data Quality Report")
                
                quality_report = data_quality_report(df)
                st.dataframe(quality_report, use_container_width=True)
                
                # Highlight columns with high missing values
                high_missing = quality_report[quality_report["Missing (%)"] > 20]["Column"].tolist()
                if high_missing:
                    st.warning(f"⚠️ Columns with >20% missing values: {', '.join(high_missing)}")
                
                # Highlight columns with potential duplicates
                potential_duplicates = []
                for col in df.columns:
                    if df[col].duplicated().sum() > 0:
                        potential_duplicates.append(col)
                
                if potential_duplicates:
                    st.info(f"ℹ️ Columns with duplicate values: {', '.join(potential_duplicates)}")
                
                # Check for row-level duplicates
                duplicate_rows = df.duplicated().sum()
                if duplicate_rows > 0:
                    st.warning(f"⚠️ Found {duplicate_rows} duplicate rows in the dataset.")
                    if st.button("Show Duplicate Rows"):
                        st.dataframe(df[df.duplicated(keep=False)].sort_values(by=df.columns[0]))
        
        # Tab 2: Smart Query
        with main_tabs[1]:
            st.header("Smart Query")
            
            # Query history section
            with st.expander("Query History", expanded=False):
                if st.session_state.history:
                    for i, (q, r) in enumerate(st.session_state.history):
                        st.subheader(f"Query {i+1}: {q}")
                        st.write(r)
                        st.divider()
                else:
                    st.info("No queries yet. Start asking questions below!")
            
            # Example queries
            st.subheader("Ask Questions About Your Data")
            example_categories = {
                "Statistical Analysis": [
                    "What is the average and standard deviation of [column]?",
                    "Find correlations between numeric columns",
                    "What are the minimum and maximum values for each numeric column?"
                ],
                "Data Visualization": [
                    "Create a scatter plot of [column_x] vs [column_y]",
                    "Show a bar chart of the top 10 values in [column]",
                    "Generate a pie chart showing the distribution of [column]"
                ],
                "Data Insights": [
                    "What interesting patterns can you find in this dataset?",
                    "Identify potential outliers in [column]",
                    "What are the main factors that relate to [column]?"
                ]
            }
            
            # Create tabs for example categories
            example_tabs = st.tabs(list(example_categories.keys()))
            
            for i, (category, examples) in enumerate(example_categories.items()):
                with example_tabs[i]:
                    for example in examples:
                        if st.button(example, key=f"ex_{category}_{example}"):
                            # Replace placeholders with actual column names
                            query = example
                            if "[column]" in query:
                                query = query.replace("[column]", df.columns[0])
                            if "[column_x]" in query:
                                query = query.replace("[column_x]", df.select_dtypes(include=['number']).columns[0])
                            if "[column_y]" in query:
                                query = query.replace("[column_y]", df.select_dtypes(include=['number']).columns[1] 
                                                     if len(df.select_dtypes(include=['number']).columns) > 1 
                                                     else df.select_dtypes(include=['number']).columns[0])
                            
                            st.session_state.query = query
            
            # Query input
            query = st.text_area(
                "Enter your question:", 
                height=100,
                key="query",
                help="Ask questions about your data in natural language"
            )
            
            col1, col2 = st.columns([3, 1])
            process_btn = col1.button('Process Query', use_container_width=True)
            save_btn = col2.button('Save to Report', use_container_width=True, disabled=not hasattr(st.session_state, 'last_result'))
            
            if process_btn and query:
                with st.spinner('Processing your query...'):
                    start_time = time.time()
                    result = process_data(df, query)
                    process_time = time.time() - start_time
                
                st.success(f"Query processed in {process_time:.2f} seconds")
                
                # Display results based on type
                result_container = st.container()
                with result_container:
                    st.subheader("Results")
                    
                    # Handle different result types
                    if isinstance(result, (int, float, np.int64, np.float64)):
                        st.metric("Result", result)
                        result_html = f"<p>Query: {query}</p><p>Result: {result}</p>"
                    elif isinstance(result, str):
                        if result.endswith('.png') and os.path.exists(result):
                            # It's a chart path
                            st.image(result, use_container_width=True)
                            result_html = f"<p>Query: {query}</p><img src='data:image/png;base64,{base64.b64encode(open(result, 'rb').read()).decode()}' width='100%'>"
                        else:
                            # It's a text response
                            st.write(result)
                            result_html = f"<p>Query: {query}</p><p>{result}</p>"
                    elif isinstance(result, pd.DataFrame):
                        st.dataframe(result, use_container_width=True)
                        result_html = f"<p>Query: {query}</p><p>Returned a dataframe with {len(result)} rows and {len(result.columns)} columns</p>"
                    else:
                        # Try to display as image
                        try:
                            st.image(result, use_container_width=True)
                            result_html = f"<p>Query: {query}</p><p>Generated visualization</p>"
                        except:
                            st.write(result)
                            result_html = f"<p>Query: {query}</p><p>{result}</p>"
                
                # Save to history
                st.session_state.history.append((query, result))
                st.session_state.last_result = result
                st.session_state.last_result_html = result_html
            
            if save_btn and hasattr(st.session_state, 'last_result'):
                report_title = st.text_input("Enter a title for this report item:", 
                                            value=f"Query: {query[:50]}..." if len(query) > 50 else f"Query: {query}")
                if report_title:
                    save_to_report(report_title, st.session_state.last_result_html)
        
        # Tab 3: Visualization Studio
        with main_tabs[2]:
            st.header("Visualization Studio")
            
            viz_col1, viz_col2 = st.columns([1, 3])
            
            with viz_col1:
                st.subheader("Chart Configuration")
                
                viz_type = st.selectbox(
                    "Chart Type",
                    ["Histogram", "Bar Chart", "Scatter Plot", "Box Plot", "Line Chart", "Pie Chart", "Heatmap"]
                )
                
                # Configure parameters based on chart type
                if viz_type != "Heatmap":
                    x_col = st.selectbox(
                        "X-axis Column",
                        df.columns,
                        index=0
                    )
                    
                    if viz_type not in ["Histogram", "Pie Chart"]:
                        y_col = st.selectbox(
                            "Y-axis Column",
                            df.select_dtypes(include=['number']).columns,
                            index=0 if len(df.select_dtypes(include=['number']).columns) > 0 else None
                        )
                    else:
                        y_col = None
                    
                    color_col = st.selectbox(
                        "Color By (Optional)",
                        ["None"] + list(df.columns),
                        index=0
                    )
                    color_col = None if color_col == "None" else color_col
                else:
                    x_col = y_col = color_col = None
                
                # Generate chart button
                generate_chart = st.button("Generate Chart", use_container_width=True)
                
                # Save to report button
                if 'current_chart' in st.session_state:
                    save_chart = st.button("Save to Report", use_container_width=True)
                    if save_chart:
                        title = f"{viz_type} of {x_col}" + (f" vs {y_col}" if y_col else "")
                        chart_html = f"<p>{title}</p><div>{st.session_state.current_chart_html}</div>"
                        save_to_report(title, chart_html)
            
            with viz_col2:
                st.subheader("Chart Preview")
                
                if generate_chart:
                    with st.spinner("Generating visualization..."):
                        fig = create_quick_viz(df, viz_type, x_col, y_col, color_col)
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Save current chart to session state
                            st.session_state.current_chart = fig
                            st.session_state.current_chart_html = fig.to_html(include_plotlyjs='cdn')
                        else:
                            st.error("Error generating chart. Please check your column selections.")
                elif 'current_chart' in st.session_state:
                    st.plotly_chart(st.session_state.current_chart, use_container_width=True)
            
            # Advanced visualization section
            with st.expander("Advanced Visualization Options"):
                st.subheader("Multi-Chart Dashboard")
                
                st.write("Create a dashboard with multiple charts")
                
                dash_cols = st.number_input("Number of columns", min_value=1, max_value=3, value=2)
                
                if st.button("Create Dashboard Template"):
                    charts = []
                    for i in range(4):  # Create 4 sample charts
                        if i % 2 == 0:
                            # Create a bar chart
                            sample_col = df.select_dtypes(include=['number']).columns[0]
                            chart = px.bar(df.head(50), y=sample_col, title=f"Sample Bar Chart {i+1}")
                        else:
                            # Create a scatter plot
                            num_cols = df.select_dtypes(include=['number']).columns
                            if len(num_cols) >= 2:
                                chart = px.scatter(df.head(50), x=num_cols[0], y=num_cols[1], 
                                                title=f"Sample Scatter Plot {i+1}")
                            else:
                                chart = px.bar(df.head(50), y=num_cols[0], title=f"Sample Chart {i+1}")
                        
                        charts.append(chart)
                    
                    # Display charts in a grid
                    dashboard_cols = st.columns(dash_cols)
                    for i, chart in enumerate(charts):
                        with dashboard_cols[i % dash_cols]:
                            st.plotly_chart(chart, use_container_width=True)
        
        # Tab 4: Data Processor
        with main_tabs[3]:
            st.header("Data Processor")
            
            processor_tabs = st.tabs(["Transform Data", "Column Operations", "Export Processed Data"])
            
            # Transform Data subtab
            with processor_tabs[0]:
                st.subheader("Transform Your Data")
                
                # Select transformation operation
                operation = st.selectbox(
                    "Select Operation",
                    [
                        "Drop NA Rows", "Drop NA Columns", "Fill NA with Mean", 
                        "Fill NA with Median", "Fill NA with Zero", "Convert to Datetime",
                        "Normalize Column", "Add Column", "Drop Column", "Filter Rows",
                        "Sort Values", "Group By", "One-Hot Encode", "Bin Values"
                    ]
                )
                
                # Apply transformation
                if st.button("Apply Transformation"):
                    with st.spinner("Transforming data..."):
                        transformed_df, message = transform_data(df, operation)
                        
                        # Display results
                        st.success(message)
                        
                        # Preview the transformed data
                        st.subheader("Transformed Data Preview")
                        st.dataframe(transformed_df.head(), use_container_width=True)
                        
                        # Option to save the transformed dataframe
                        if st.button("Save Transformed Data"):
                            new_df_name = f"{st.session_state.current_df_name}_transformed"
                            st.session_state.dataframes[new_df_name] = transformed_df
                            st.session_state.current_df_name = new_df_name
                            st.success(f"Saved as '{new_df_name}'")
                            st.experimental_rerun()
            
            # Column Operations subtab
            with processor_tabs[1]:
                st.subheader("Column Operations")
                
                # Select operation type
                op_type = st.radio(
                    "Operation Type",
                    ["Rename Columns", "Change Data Types", "Create New Column"]
                )
                
                if op_type == "Rename Columns":
                    st.write("Rename multiple columns")
                    
                    rename_cols = {}
                    for col in df.columns:
                        new_name = st.text_input(f"New name for '{col}'", value=col)
                        if new_name != col:
                            rename_cols[col] = new_name
                    
                    if rename_cols and st.button("Apply Rename"):
                        new_df = df.copy().rename(columns=rename_cols)
                        new_df_name = f"{st.session_state.current_df_name}_renamed"
                        st.session_state.dataframes[new_df_name] = new_df
                        st.session_state.current_df_name = new_df_name
                        st.success(f"Renamed columns and saved as '{new_df_name}'")
                        st.experimental_rerun()
                
                elif op_type == "Change Data Types":
                    st.write("Change column data types")
                    
                    col_to_change = st.selectbox("Select column", df.columns)
                    new_type = st.selectbox(
                        "New data type", 
                        ["float64", "int64", "string", "category", "datetime64", "boolean"]
                    )
                    
                    if st.button("Apply Type Change"):
                        try:
                            new_df = df.copy()
                            if new_type == "datetime64":
                                new_df[col_to_change] = pd.to_datetime(new_df[col_to_change])
                            else:
                                new_df[col_to_change] = new_df[col_to_change].astype(new_type)
                                
                            new_df_name = f"{st.session_state.current_df_name}_typed"
                            st.session_state.dataframes[new_df_name] = new_df
                            st.session_state.current_df_name = new_df_name
                            st.success(f"Changed data type and saved as '{new_df_name}'")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error changing data type: {str(e)}")
                
                elif op_type == "Create New Column":
                    st.write("Create a new calculated column")
                    
                    col_name = st.text_input("New column name")
                    formula_type = st.selectbox(
                        "Formula type", 
                        ["Basic Operation", "Custom Expression"]
                    )
                    
                    if formula_type == "Basic Operation":
                        col1 = st.selectbox("Column 1", df.columns)
                        operation = st.selectbox("Operation", ["+", "-", "*", "/", "%"])
                        col2_or_val = st.text_input("Column 2 or value")
                        
                        expression = f"df['{col1}'] {operation} "
                        if col2_or_val in df.columns:
                            expression += f"df['{col2_or_val}']"
                        else:
                            try:
                                # Try to convert to number
                                val = float(col2_or_val)
                                expression += f"{val}"
                            except:
                                expression += f"'{col2_or_val}'"
                    else:
                        expression = st.text_area(
                            "Custom expression (use 'df' to refer to dataframe)",
                            value="df['col1'] + df['col2']"
                        )
                    
                    if col_name and expression and st.button("Create Column"):
                        try:
                            new_df = df.copy()
                            # Safely evaluate the expression
                            local_dict = {"df": new_df, "np": np, "pd": pd}
                            new_df[col_name] = eval(expression, {"__builtins__": {}}, local_dict)
                            
                            new_df_name = f"{st.session_state.current_df_name}_new_col"
                            st.session_state.dataframes[new_df_name] = new_df
                            st.session_state.current_df_name = new_df_name
                            st.success(f"Added new column '{col_name}' and saved as '{new_df_name}'")
                            st.experimental_rerun()
                        except Exception as e:
                            st.error(f"Error creating column: {str(e)}")
            
            # Export Data subtab
            with processor_tabs[2]:
                st.subheader("Export Processed Data")
                
                export_format = st.selectbox(
                    "Export Format",
                    ["CSV", "Excel", "JSON", "Parquet"]
                )
                
                # Export options
                if export_format == "CSV":
                    delimiter = st.text_input("Delimiter", value=",")
                    index = st.checkbox("Include Index", value=False)
                
                elif export_format == "Excel":
                    sheet_name = st.text_input("Sheet Name", value="Data")
                    index = st.checkbox("Include Index", value=False)
                
                elif export_format == "JSON":
                    orient = st.selectbox(
                        "JSON Orientation",
                        ["records", "columns", "index", "split", "table"]
                    )
                
                # Generate export
                if st.button("Generate Export File"):
                    with st.spinner(f"Preparing {export_format} export..."):
                        try:
                            if export_format == "CSV":
                                csv = df.to_csv(index=index, sep=delimiter)
                                b64 = base64.b64encode(csv.encode()).decode()
                                href = f'<a href="data:file/csv;base64,{b64}" download="{st.session_state.current_df_name}.csv">Download CSV File</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                            elif export_format == "Excel":
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    df.to_excel(writer, index=index, sheet_name=sheet_name)
                                b64 = base64.b64encode(output.getvalue()).decode()
                                href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{st.session_state.current_df_name}.xlsx">Download Excel File</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                            elif export_format == "JSON":
                                json_str = df.to_json(orient=orient)
                                b64 = base64.b64encode(json_str.encode()).decode()
                                href = f'<a href="data:application/json;base64,{b64}" download="{st.session_state.current_df_name}.json">Download JSON File</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                            elif export_format == "Parquet":
                                output = BytesIO()
                                df.to_parquet(output)
                                b64 = base64.b64encode(output.getvalue()).decode()
                                href = f'<a href="data:application/octet-stream;base64,{b64}" download="{st.session_state.current_df_name}.parquet">Download Parquet File</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                            st.success(f"Export prepared! Click the link above to download.")
                        except Exception as e:
                            st.error(f"Error generating export: {str(e)}")
        
        # Tab 5: Insights & Report
        with main_tabs[4]:
            st.header("Insights & Report")
            
            insights_tabs = st.tabs(["Automated Insights", "Report Builder", "Export Report"])
            
            # Automated Insights subtab
            with insights_tabs[0]:
                st.subheader("Automated Insights")
                
                if st.button("Generate Automated Insights"):
                    with st.spinner("Analyzing data for insights..."):
                        insights = generate_automated_insights(df)
                        
                        st.markdown("### Key Findings")
                        st.markdown(insights)
                        
                        # Add option to save to report
                        if st.button("Add to Report"):
                            save_to_report("Automated Insights", f"<p><strong>Automated insights:</strong></p><p>{insights}</p>")
                
                st.subheader("Custom AI Analysis")
                ai_query = st.text_area(
                    "Ask for specific insights",
                    placeholder="E.g., Find key trends in this dataset or Analyze relationships between columns X and Y"
                )
                
                if ai_query and st.button("Generate Analysis"):
                    with st.spinner("Generating AI analysis..."):
                        result = process_data(df, ai_query)
                        
                        st.markdown("### AI Analysis")
                        if isinstance(result, str):
                            if result.endswith('.png') and os.path.exists(result):
                                st.image(result, use_container_width=True)
                                result_html = f"<p><strong>Query:</strong> {ai_query}</p><img src='data:image/png;base64,{base64.b64encode(open(result, 'rb').read()).decode()}' width='100%'>"
                            else:
                                st.write(result)
                                result_html = f"<p><strong>Query:</strong> {ai_query}</p><p>{result}</p>"
                        else:
                            st.write(result)
                            result_html = f"<p><strong>Query:</strong> {ai_query}</p><p>{str(result)}</p>"
                        
                        # Add to report button
                        if st.button("Add Analysis to Report"):
                            save_to_report(f"AI Analysis: {ai_query[:30]}...", result_html)
            
            # Report Builder subtab
            with insights_tabs[1]:
                st.subheader("Report Builder")
                
                if 'report' not in st.session_state or not st.session_state.report:
                    st.info("Your report is empty. Add items from other tabs to build your report.")
                else:
                    # Show report items with option to remove
                    for i, item in enumerate(st.session_state.report):
                        with st.expander(f"{i + 1}. {item['title']}"):
                            st.write(f"Added on: {item['timestamp']}")
                            st.markdown(item['content'], unsafe_allow_html=True)
                            
                            if st.button(f"Remove from Report", key=f"remove_{i}"):
                                st.session_state.report.pop(i)
                                st.experimental_rerun()
                    
                    # Clear report button
                    if st.button("Clear Report"):
                        st.session_state.report = []
                        st.experimental_rerun()
            
            # Export Report subtab
            with insights_tabs[2]:
                st.subheader("Export Report")
                
                if 'report' not in st.session_state or not st.session_state.report:
                    st.info("Your report is empty. Add items from other tabs to build your report.")
                else:
                    # Export options
                    export_format = st.selectbox("Export Format", ["HTML", "PDF"])
                    
                    report_title = st.text_input("Report Title", value="ClearVision Data Analysis Report")
                    author = st.text_input("Author", value="")
                    
                    if st.button("Generate Report"):
                        with st.spinner("Generating report..."):
                            if export_format == "HTML":
                                html_content = export_report_html()
                                
                                # Add custom title and author
                                html_content = html_content.replace("<h1>ClearVision Analytics Report</h1>", 
                                                                  f"<h1>{report_title}</h1>")
                                if author:
                                    date_str = f"<p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>"
                                    html_content = html_content.replace(date_str, 
                                                                      f"{date_str}<p>Author: {author}</p>")
                                
                                b64 = base64.b64encode(html_content.encode()).decode()
                                href = f'<a href="data:text/html;base64,{b64}" download="data_analysis_report.html">Download HTML Report</a>'
                                st.markdown(href, unsafe_allow_html=True)
                                
                                st.success("Report generated! Click the link above to download.")
                            
                            elif export_format == "PDF":
                                st.error("PDF export functionality requires additional setup. Please use HTML export for now.")
    else:
        # Welcome message when no data is loaded
        st.markdown("""
        # Welcome to ClearVision Analytics Platform
        
        ### Your all-in-one solution for data analysis and visualization
        
        This platform helps you:
        - Explore and clean your data
        - Ask natural language questions about your data
        - Create beautiful visualizations
        - Process and transform your datasets
        - Generate automated insights and reports
        
        To get started, please upload a dataset using the sidebar on the left.
        """)
        
        # Features showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔍 Smart Query")
            st.markdown("Ask questions in plain English and get answers from your data instantly.")
        
        with col2:
            st.markdown("### 📊 Visualization")
            st.markdown("Create publication-ready charts with just a few clicks.")
        
        with col3:
            st.markdown("### 📝 Reports")
            st.markdown("Build and export professional reports to share with your team.")

# Run the application
if __name__ == "__main__":
    main()
    
