import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

# Add the current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Import page modules
try:
    from pages import (
        dashboard,
        individual_prediction,
        batch_prediction,
        data_exploration,
        model_performance,
        shap_explainability,
        business_intelligence,
        compliance_report
    )
except ImportError as e:
    st.error(f"Error importing page modules: {e}")
    st.info("Please ensure all page modules are properly installed.")

# Import configuration
try:
    from config import APP_CONFIG, THEME_CONFIG
except ImportError:
    # Fallback configuration
    APP_CONFIG = {
        'app_name': 'CreditAnalyticsHub',
        'company_name': 'FinTech Solutions Inc.',
        'version': '2.0.0'
    }
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'background_color': '#0E1117',
        'secondary_background': '#1E1E1E'
    }

# Page configuration
st.set_page_config(
    page_title="CreditAnalyticsHub - Advanced Credit Risk Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced Black Theme CSS
def load_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0E1117 0%, #1E1E1E 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom Header */
    .main-header {
        background: linear-gradient(90deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        border: 1px solid #333;
        box-shadow: 0 8px 32px rgba(0, 212, 255, 0.1);
    }
    
    .header-content {
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .company-info h1 {
        color: #00D4FF;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 0 20px rgba(0, 212, 255, 0.3);
    }
    
    .company-info p {
        color: #B0B0B0;
        font-size: 1.1rem;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }
    
    .system-status {
        display: flex;
        gap: 1rem;
        align-items: center;
        flex-wrap: wrap;
    }
    
    .status-item {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(0, 212, 255, 0.1);
        padding: 0.5rem 1rem;
        border-radius: 25px;
        border: 1px solid rgba(0, 212, 255, 0.2);
    }
    
    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00FF88;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-text {
        color: #E0E0E0;
        font-size: 0.9rem;
        font-weight: 500;
    }
    
    /* Enhanced Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: #1E1E1E;
        border-radius: 15px;
        padding: 0.5rem;
        margin-bottom: 2rem;
        border: 1px solid #333;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        padding: 0 1.5rem;
        background: transparent;
        border-radius: 10px;
        color: #B0B0B0;
        font-weight: 500;
        font-size: 0.95rem;
        border: none;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(0, 212, 255, 0.1);
        color: #00D4FF;
        transform: translateY(-2px);
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: #000 !important;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stTabs [data-baseweb="tab-panel"] {
        background: rgba(30, 30, 30, 0.5);
        border-radius: 15px;
        padding: 2rem;
        border: 1px solid #333;
        backdrop-filter: blur(10px);
    }
    
    /* Custom Metrics */
    .metric-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        transition: transform 0.3s ease;
    }
    
    .metric-container:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #00D4FF;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        color: #B0B0B0;
        font-size: 1rem;
        font-weight: 500;
    }
    
    /* Enhanced Cards */
    .custom-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin-bottom: 1rem;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
        border-color: rgba(0, 212, 255, 0.3);
    }
    
    .card-title {
        color: #00D4FF;
        font-size: 1.2rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    
    .card-content {
        color: #E0E0E0;
        line-height: 1.6;
    }
    
    /* Custom Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: #000;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 212, 255, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 212, 255, 0.4);
    }
    
    /* Custom Selectbox */
    .stSelectbox > div > div {
        background: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
    }
    
    /* Custom Input Fields */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input {
        background: #1E1E1E;
        border: 1px solid #333;
        border-radius: 10px;
        color: #E0E0E0;
    }
    
    /* Footer */
    .custom-footer {
        background: #1E1E1E;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        border: 1px solid #333;
        text-align: center;
    }
    
    .footer-content {
        color: #B0B0B0;
        font-size: 0.9rem;
    }
    
    .footer-brand {
        color: #00D4FF;
        font-weight: 600;
    }
    
    /* Responsive Design */
    @media (max-width: 768px) {
        .header-content {
            flex-direction: column;
            gap: 1rem;
        }
        
        .company-info h1 {
            font-size: 2rem;
        }
        
        .system-status {
            justify-content: center;
        }
        
        .stTabs [data-baseweb="tab"] {
            padding: 0 1rem;
            font-size: 0.85rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize session state variables"""
    if 'user_data' not in st.session_state:
        st.session_state.user_data = {}
    
    if 'model_cache' not in st.session_state:
        st.session_state.model_cache = {}
    
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    
    if 'system_status' not in st.session_state:
        st.session_state.system_status = {
            'models_loaded': True,
            'data_connected': True,
            'api_status': True,
            'last_update': datetime.now()
        }

# System status component
def render_system_status():
    """Render system status indicators"""
    status = st.session_state.system_status
    
    status_html = f"""
    <div class="system-status">
        <div class="status-item">
            <div class="status-dot"></div>
            <span class="status-text">Models: {'Online' if status['models_loaded'] else 'Offline'}</span>
        </div>
        <div class="status-item">
            <div class="status-dot"></div>
            <span class="status-text">Data: {'Connected' if status['data_connected'] else 'Disconnected'}</span>
        </div>
        <div class="status-item">
            <div class="status-dot"></div>
            <span class="status-text">API: {'Active' if status['api_status'] else 'Inactive'}</span>
        </div>
        <div class="status-item">
            <span class="status-text">Updated: {status['last_update'].strftime('%H:%M')}</span>
        </div>
    </div>
    """
    return status_html

# Main header component
def render_header():
    """Render the main application header"""
    system_status_html = render_system_status()
    
    header_html = f"""
    <div class="main-header">
        <div class="header-content">
            <div class="company-info">
                <h1>🏦 {APP_CONFIG['app_name']}</h1>
                <p>Advanced Credit Risk Analytics Platform | {APP_CONFIG['company_name']} | v{APP_CONFIG['version']}</p>
            </div>
            {system_status_html} 
        </div>
    </div>
    """
    
    st.markdown(header_html, unsafe_allow_html=True)

# Footer component
def render_footer():
    """Render the application footer"""
    footer_html = f"""
    <div class="custom-footer">
        <div class="footer-content">
            <p>© 2025 <span class="footer-brand">{APP_CONFIG['company_name']}</span> | 
            Powered by <span class="footer-brand">{APP_CONFIG['app_name']}</span> | 
            Advanced Credit Risk Analytics & AI-Driven Insights</p>
            <p>🔒 Enterprise Security | 📊 Real-time Analytics | 🤖 AI-Powered Predictions</p>
        </div>
    </div>
    """
    
    st.markdown(footer_html, unsafe_allow_html=True)

# Error handling wrapper
def safe_import_and_run(module_name, function_name):
    """Safely import and run page functions with error handling"""
    try:
        if module_name == 'dashboard':
            from pages import dashboard
            dashboard.show()
        elif module_name == 'individual_prediction':
            from pages import individual_prediction
            individual_prediction.show()
        elif module_name == 'batch_prediction':
            from pages import batch_prediction
            batch_prediction.show()
        elif module_name == 'data_exploration':
            from pages import data_exploration
            data_exploration.show()
        elif module_name == 'model_performance':
            from pages import model_performance
            model_performance.show()
        elif module_name == 'shap_explainability':
            from pages import shap_explainability
            shap_explainability.show()
        elif module_name == 'business_intelligence':
            from pages import business_intelligence
            business_intelligence.show()
        elif module_name == 'compliance_report':
            from pages import compliance_report
            compliance_report.show()
        else:
            st.error(f"Unknown module: {module_name}")
    except ImportError as e:
        st.error(f"Module {module_name} not found: {str(e)}")
        st.info(f"The {module_name} module is currently under development.")
        
        # Show placeholder content
        st.markdown(f"""
        <div class="custom-card">
            <div class="card-title">🚧 {module_name.replace('_', ' ').title()} - Coming Soon</div>
            <div class="card-content">
                This advanced feature is currently being developed and will be available in the next release.
                <br><br>
                <strong>Planned Features:</strong>
                <ul>
                    <li>Real-time data processing</li>
                    <li>Advanced analytics and visualizations</li>
                    <li>AI-powered insights</li>
                    <li>Interactive dashboards</li>
                </ul>
            </div>
        </div>
        """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error in {module_name}: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")

# Main application
def main():
    """Main application function"""
    try:
        # Load CSS and initialize
        load_css()
        init_session_state()
        
        # Render header
        render_header()
        
        # Tab configuration
        tab_config = [
            ("📊 Dashboard", "dashboard"),
            ("👤 Individual", "individual_prediction"),
            ("📋 Batch", "batch_prediction"),
            ("🔍 Data Explorer", "data_exploration"),
            ("📈 Performance", "model_performance"),
            ("🧠 SHAP AI", "shap_explainability"),
            ("💼 Business Intel", "business_intelligence"),
            ("📋 Compliance", "compliance_report")
        ]
        
        # Create tabs
        tabs = st.tabs([config[0] for config in tab_config])
        
        # Render tab content
        for i, (tab_name, tab_key) in enumerate(tab_config):
            with tabs[i]:
                safe_import_and_run(tab_key, 'show')
        
        # Render footer
        render_footer()
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")
        
        # Show emergency contact info
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">🆘 Emergency Support</div>
            <div class="card-content">
                If you continue to experience issues, please contact our technical support team:
                <br><br>
                📧 Email: creditsolutions<br>
                📞 Phone: +1 (555) 123-4567<br>
                💬 Live Chat: Available 24/7
            </div>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()