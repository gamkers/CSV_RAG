import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
from io import StringIO, BytesIO

# Set page configuration
st.set_page_config(
    page_title="ClearVision Analytics",
    page_icon="📊",
    layout="wide"
)

# Custom CSS to improve UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .css-1d391kg {
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize ChatGroq with API key from environment
def init_llm():
    try:
        llm = ChatGroq(
            model_name='llama3-70b-8192',
            api_key="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"
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def read_file(uploaded_file):
    """
    Read the uploaded file based on its type (Excel or CSV).
    """
    try:
        if uploaded_file.name.endswith('.csv'):
            return pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            return pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None

def get_data_summary(df):
    """
    Generate a summary of the DataFrame.
    """
    summary = {
        "Total Rows": len(df),
        "Total Columns": len(df.columns),
        "Missing Values": df.isnull().sum().sum(),
        "Numeric Columns": len(df.select_dtypes(include=['int64', 'float64']).columns),
        "Categorical Columns": len(df.select_dtypes(include=['object', 'category']).columns)
    }
    return summary

def process_data(df, query):
    """
    Process the DataFrame using SmartDataframe and ChatGroq.
    """
    try:
        with st.spinner('Processing your query...'):
            llm = init_llm()
            if llm is None:
                return None
            
            smart_df = SmartDataframe(df, config={'llm': llm})
            result = smart_df.chat(query)
            return result
    except Exception as e:
        st.error(f"Error processing query: {str(e)}")
        return None

def display_result(result):
    """
    Display the result based on its type.
    """
    if result is None:
        return
    
    st.write("### Results")
    
    if isinstance(result, (int, np.int64)):
        st.metric("Value", result)
    elif isinstance(result, str):
        if result.endswith('.png'):
            st.image(result, use_column_width=True)
        else:
            st.write(result)
    else:
        try:
            st.image(result, use_column_width=True)
        except:
            st.write(result)

def main():
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Go to", ["Data Analysis", "Help & Guidelines"])
    
    if page == "Data Analysis":
        # Main content
        st.title('ClearVision Analytics Dashboard')
        st.subheader('Powered by AI - Created by Rajesh')
        
        uploaded_file = st.file_uploader("Upload your data file", type=["csv", "xlsx"])
        
        if uploaded_file is not None:
            df = read_file(uploaded_file)
            
            if df is not None:
                # Create two columns for layout
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write("### Data Preview")
                    st.dataframe(df.head(), use_container_width=True)
                
                with col2:
                    st.write("### Data Summary")
                    summary = get_data_summary(df)
                    for key, value in summary.items():
                        st.metric(key, value)
                
                # Query section
                st.write("### Ask Questions About Your Data")
                query = st.text_area(
                    "Enter your question",
                    placeholder="Example: 'Show a pie chart for the distribution of categories' or 'What is the average value?'"
                )
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col2:
                    if st.button('Analyze', use_container_width=True):
                        if query:
                            result = process_data(df, query)
                            display_result(result)
                        else:
                            st.warning("Please enter a query before submitting.")
                
                # Additional features
                with st.expander("📊 Available Columns"):
                    st.write(df.columns.tolist())
                
                with st.expander("📈 Data Types"):
                    st.write(df.dtypes)
    
    else:  # Help & Guidelines page
        st.title("Help & Guidelines")
        st.write("""
        ### 🎯 How to Use ClearVision
        1. Upload your CSV or Excel file using the file uploader
        2. Review the data preview and summary statistics
        3. Enter your question in natural language
        4. Click 'Analyze' to get insights
        
        ### 📝 Example Questions
        - "Show a bar chart of sales by category"
        - "What is the average revenue?"
        - "Create a scatter plot comparing price and quantity"
        - "Show the trend of sales over time"
        
        ### ⚠️ Tips
        - Make sure your data is clean and properly formatted
        - Be specific in your questions
        - Check the data types of your columns
        - Use clear and concise language
        """)

if __name__ == "__main__":
    main()
