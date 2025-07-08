"""
Data Exploration Page for CreditAnalyticsHub
==========================================
Interactive data analysis and visualization tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import io
from typing import Dict, List, Tuple, Any

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }

def generate_sample_data(n_samples: int = 1000) -> pd.DataFrame:
    """Generate sample credit data for exploration"""
    np.random.seed(42)

    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(1, n_samples + 1)],
        'age': np.random.normal(40, 12, n_samples).clip(18, 80).astype(int),
        'annual_income': np.random.lognormal(10.8, 0.6, n_samples).clip(25000, 500000),
        'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850).astype(int),
        'loan_amount': np.random.lognormal(10.1, 0.8, n_samples).clip(5000, 200000),
        'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
        'debt_to_income': np.random.beta(2, 5, n_samples).clip(0, 0.8),
        'credit_utilization': np.random.beta(2, 3, n_samples).clip(0, 1),
        'delinquencies_2yrs': np.random.poisson(0.5, n_samples).clip(0, 10),
        'inquiries_6mths': np.random.poisson(1, n_samples).clip(0, 15),
        'open_accounts': np.random.poisson(8, n_samples).clip(1, 30),
        'total_accounts': np.random.poisson(15, n_samples).clip(5, 50)
    }

    # Add categorical variables
    loan_purposes = ['debt_consolidation', 'home_improvement', 'major_purchase', 'medical', 'car', 'other']
    home_ownership = ['RENT', 'MORTGAGE', 'OWN', 'OTHER']
    employment_status = ['EMPLOYED', 'SELF_EMPLOYED', 'UNEMPLOYED', 'RETIRED']

    data['loan_purpose'] = np.random.choice(loan_purposes, n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    data['home_ownership'] = np.random.choice(home_ownership, n_samples, p=[0.4, 0.35, 0.2, 0.05])
    data['employment_status'] = np.random.choice(employment_status, n_samples, p=[0.7, 0.15, 0.1, 0.05])

    # Add target variable (default risk)
    risk_score = (
        (1 - data['credit_score'] / 850) * 0.3 +
        (data['debt_to_income']) * 0.25 +
        (data['credit_utilization']) * 0.2 +
        (data['delinquencies_2yrs'] / 10) * 0.15 +
        (data['inquiries_6mths'] / 15) * 0.1
    )
    data['default_risk'] = (risk_score > 0.5).astype(int)

    return pd.DataFrame(data)

def create_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Create interactive correlation heatmap"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmid=0,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False
    ))

    fig.update_layout(
        title="Feature Correlation Matrix",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=600
    )

    return fig

def create_distribution_plot(df: pd.DataFrame, column: str) -> go.Figure:
    """Create distribution plot for selected column"""
    if df[column].dtype in ['object', 'category']:
        # Categorical variable - bar chart
        value_counts = df[column].value_counts()
        fig = go.Figure(data=[go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker=dict(color=THEME_CONFIG['primary_color'])
        )])
        fig.update_layout(title=f"Distribution of {column}")
    else:
        # Numerical variable - histogram
        fig = go.Figure(data=[go.Histogram(
            x=df[column],
            nbinsx=30,
            marker=dict(color=THEME_CONFIG['primary_color'])
        )])
        fig.update_layout(title=f"Distribution of {column}")

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str, color_col: str = None) -> go.Figure:
    """Create interactive scatter plot"""
    if color_col and color_col in df.columns:
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{y_col} vs {x_col} (colored by {color_col})")
    else:
        fig = px.scatter(df, x=x_col, y=y_col,
                        title=f"{y_col} vs {x_col}")

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def assess_data_quality(df: pd.DataFrame) -> Dict:
    """Assess data quality metrics"""
    quality_metrics = {
        'total_records': len(df),
        'total_features': len(df.columns),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_records': df.duplicated().sum(),
        'numeric_features': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns)
    }

    # Missing value percentage by column
    missing_by_column = (df.isnull().sum() / len(df) * 100).round(2)
    quality_metrics['missing_by_column'] = missing_by_column[missing_by_column > 0]

    return quality_metrics

def show():
    """Main data exploration display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .exploration-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .metric-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem 0;
    }

    .quality-indicator {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        display: inline-block;
    }

    .quality-excellent { background: #00FF88; color: #000; }
    .quality-good { background: #90EE90; color: #000; }
    .quality-warning { background: #FFB800; color: #000; }
    .quality-poor { background: #FF4B4B; color: #FFF; }

    .filter-section {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("## 🔍 Data Exploration & Analysis")
    st.markdown("Interactive tools for comprehensive data analysis and visualization")

    # Data Loading Section
    st.markdown("### 📊 Data Source")

    col_data1, col_data2 = st.columns([2, 1])

    with col_data1:
        data_source = st.radio(
            "Select data source:",
            ["Sample Data", "Upload File"],
            horizontal=True
        )

        if data_source == "Upload File":
            uploaded_file = st.file_uploader("Choose CSV or Excel file", type=['csv', 'xlsx'])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                    st.success(f"✅ File loaded: {len(df)} records, {len(df.columns)} columns")
                except Exception as e:
                    st.error(f"Error loading file: {str(e)}")
                    df = generate_sample_data()
            else:
                df = generate_sample_data()
        else:
            sample_size = st.slider("Sample size", 100, 5000, 1000, 100)
            df = generate_sample_data(sample_size)

    with col_data2:
        if 'df' in locals():
            st.markdown("**📈 Dataset Overview**")
            st.metric("Records", f"{len(df):,}")
            st.metric("Features", len(df.columns))
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    if 'df' not in locals() or df is None:
        st.warning("Please load data to continue exploration")
        return

    # Data Quality Assessment
    st.markdown("---")
    st.markdown("### 🔍 Data Quality Assessment")

    quality_metrics = assess_data_quality(df)

    col_q1, col_q2, col_q3, col_q4 = st.columns(4)

    with col_q1:
        missing_pct = (quality_metrics['missing_values'] / (len(df) * len(df.columns))) * 100
        quality_class = "quality-excellent" if missing_pct < 1 else "quality-good" if missing_pct < 5 else "quality-warning" if missing_pct < 10 else "quality-poor"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #00D4FF; font-size: 1.5rem; font-weight: 700;">{missing_pct:.1f}%</div>
            <div style="color: #B0B0B0;">Missing Values</div>
            <div class="quality-indicator {quality_class}">
                {'Excellent' if missing_pct < 1 else 'Good' if missing_pct < 5 else 'Warning' if missing_pct < 10 else 'Poor'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_q2:
        dup_pct = (quality_metrics['duplicate_records'] / len(df)) * 100
        quality_class = "quality-excellent" if dup_pct < 1 else "quality-good" if dup_pct < 3 else "quality-warning" if dup_pct < 5 else "quality-poor"
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #00D4FF; font-size: 1.5rem; font-weight: 700;">{dup_pct:.1f}%</div>
            <div style="color: #B0B0B0;">Duplicates</div>
            <div class="quality-indicator {quality_class}">
                {'Excellent' if dup_pct < 1 else 'Good' if dup_pct < 3 else 'Warning' if dup_pct < 5 else 'Poor'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_q3:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #00D4FF; font-size: 1.5rem; font-weight: 700;">{quality_metrics['numeric_features']}</div>
            <div style="color: #B0B0B0;">Numeric Features</div>
        </div>
        """, unsafe_allow_html=True)

    with col_q4:
        st.markdown(f"""
        <div class="metric-card">
            <div style="color: #00D4FF; font-size: 1.5rem; font-weight: 700;">{quality_metrics['categorical_features']}</div>
            <div style="color: #B0B0B0;">Categorical Features</div>
        </div>
        """, unsafe_allow_html=True)

    # Summary Statistics
    st.markdown("### 📊 Summary Statistics")

    tab_numeric, tab_categorical = st.tabs(["📈 Numeric", "📋 Categorical"])

    with tab_numeric:
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            st.dataframe(numeric_df.describe().round(2), use_container_width=True)
        else:
            st.info("No numeric columns found")

    with tab_categorical:
        categorical_df = df.select_dtypes(include=['object', 'category'])
        if not categorical_df.empty:
            cat_summary = []
            for col in categorical_df.columns:
                cat_summary.append({
                    'Column': col,
                    'Unique Values': df[col].nunique(),
                    'Most Frequent': df[col].mode().iloc[0] if not df[col].mode().empty else 'N/A',
                    'Frequency': df[col].value_counts().iloc[0] if not df[col].empty else 0
                })
            st.dataframe(pd.DataFrame(cat_summary), use_container_width=True)
        else:
            st.info("No categorical columns found")

    # Interactive Visualizations
    st.markdown("---")
    st.markdown("### 📊 Interactive Visualizations")

    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["🔥 Correlation", "📊 Distributions", "🎯 Scatter Plot", "📈 Box Plots"])

    with viz_tab1:
        st.markdown("**Feature Correlation Analysis**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 1:
            fig_corr = create_correlation_matrix(df)
            st.plotly_chart(fig_corr, use_container_width=True)

            # Highlight strong correlations
            corr_matrix = df[numeric_cols].corr()
            strong_corr = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        strong_corr.append({
                            'Feature 1': corr_matrix.columns[i],
                            'Feature 2': corr_matrix.columns[j],
                            'Correlation': f"{corr_val:.3f}"
                        })

            if strong_corr:
                st.markdown("**🔍 Strong Correlations (|r| > 0.7):**")
                st.dataframe(pd.DataFrame(strong_corr), use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")

    with viz_tab2:
        st.markdown("**Feature Distributions**")
        col_dist1, col_dist2 = st.columns([1, 2])

        with col_dist1:
            selected_column = st.selectbox("Select column:", df.columns.tolist())

        with col_dist2:
            if selected_column:
                fig_dist = create_distribution_plot(df, selected_column)
                st.plotly_chart(fig_dist, use_container_width=True)

    with viz_tab3:
        st.markdown("**Scatter Plot Analysis**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) >= 2:
            col_scatter1, col_scatter2, col_scatter3 = st.columns(3)

            with col_scatter1:
                x_col = st.selectbox("X-axis:", numeric_cols, key="scatter_x")
            with col_scatter2:
                y_col = st.selectbox("Y-axis:", numeric_cols, index=1, key="scatter_y")
            with col_scatter3:
                color_col = st.selectbox("Color by:", ['None'] + df.columns.tolist(), key="scatter_color")

            if x_col and y_col:
                color_column = color_col if color_col != 'None' else None
                fig_scatter = create_scatter_plot(df, x_col, y_col, color_column)
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least 2 numeric columns for scatter plot")

    with viz_tab4:
        st.markdown("**Box Plot Analysis**")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        if numeric_cols and categorical_cols:
            col_box1, col_box2 = st.columns(2)

            with col_box1:
                box_numeric = st.selectbox("Numeric column:", numeric_cols, key="box_numeric")
            with col_box2:
                box_categorical = st.selectbox("Group by:", categorical_cols, key="box_categorical")

            if box_numeric and box_categorical:
                fig_box = px.box(df, x=box_categorical, y=box_numeric, 
                               title=f"{box_numeric} by {box_categorical}")
                fig_box.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Need both numeric and categorical columns for box plots")

    # Data Filtering and Segmentation
    st.markdown("---")
    st.markdown("### 🎛️ Data Filtering & Segmentation")

    with st.expander("🔧 Apply Filters", expanded=False):
        filter_cols = st.columns(3)
        filters = {}

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

        # Numeric filters
        if numeric_cols:
            with filter_cols[0]:
                st.markdown("**Numeric Filters**")
                for col in numeric_cols[:3]:  # Limit to first 3 for space
                    min_val, max_val = float(df[col].min()), float(df[col].max())
                    filters[col] = st.slider(f"{col}", min_val, max_val, (min_val, max_val), key=f"filter_{col}")

        # Categorical filters
        if categorical_cols:
            with filter_cols[1]:
                st.markdown("**Categorical Filters**")
                for col in categorical_cols[:3]:  # Limit to first 3 for space
                    unique_vals = df[col].unique().tolist()
                    filters[f"{col}_cat"] = st.multiselect(f"{col}", unique_vals, unique_vals, key=f"filter_cat_{col}")

        # Apply filters
        filtered_df = df.copy()
        for col, values in filters.items():
            if col.endswith('_cat'):
                original_col = col.replace('_cat', '')
                if original_col in df.columns:
                    filtered_df = filtered_df[filtered_df[original_col].isin(values)]
            else:
                if col in df.columns:
                    filtered_df = filtered_df[(filtered_df[col] >= values[0]) & (filtered_df[col] <= values[1])]

        with filter_cols[2]:
            st.markdown("**Filter Results**")
            st.metric("Original Records", f"{len(df):,}")
            st.metric("Filtered Records", f"{len(filtered_df):,}")
            st.metric("Reduction", f"{((len(df) - len(filtered_df)) / len(df) * 100):.1f}%")

    # Export Options
    st.markdown("---")
    st.markdown("### 📤 Export Data & Analysis")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        if st.button("📄 Export Filtered Data"):
            csv_buffer = io.StringIO()
            filtered_df.to_csv(csv_buffer, index=False)
            st.download_button(
                "Download CSV",
                csv_buffer.getvalue(),
                f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

    with col_export2:
        if st.button("📊 Export Summary Report"):
            report = f"""
Data Exploration Report
======================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Dataset Overview:
- Total Records: {len(df):,}
- Total Features: {len(df.columns)}
- Numeric Features: {quality_metrics['numeric_features']}
- Categorical Features: {quality_metrics['categorical_features']}

Data Quality:
- Missing Values: {quality_metrics['missing_values']:,} ({(quality_metrics['missing_values'] / (len(df) * len(df.columns))) * 100:.1f}%)
- Duplicate Records: {quality_metrics['duplicate_records']:,} ({(quality_metrics['duplicate_records'] / len(df)) * 100:.1f}%)

Summary Statistics:
{df.describe().to_string()}
            """
            st.download_button(
                "Download Report",
                report,
                f"exploration_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

    with col_export3:
        if st.button("🔄 Reset Analysis"):
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"🔍 Exploring {len(df):,} records across {len(df.columns)} features | "
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
