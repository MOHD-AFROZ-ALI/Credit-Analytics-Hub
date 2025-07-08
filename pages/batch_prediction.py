"""
Batch Prediction Page for CreditAnalyticsHub
==========================================
Upload and process multiple credit applications for batch risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import json
from typing import Dict, List, Tuple, Any
import time

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
    RISK_CONFIG = get_config('risk')
    VALIDATION_CONFIG = get_config('validation')
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }
    RISK_CONFIG = {
        'risk_categories': {
            'very_low': {'min': 0.0, 'max': 0.2, 'color': '#00FF88', 'label': 'Very Low Risk'},
            'low': {'min': 0.2, 'max': 0.4, 'color': '#90EE90', 'label': 'Low Risk'},
            'medium': {'min': 0.4, 'max': 0.6, 'color': '#FFB800', 'label': 'Medium Risk'},
            'high': {'min': 0.6, 'max': 0.8, 'color': '#FF8C00', 'label': 'High Risk'},
            'very_high': {'min': 0.8, 'max': 1.0, 'color': '#FF4B4B', 'label': 'Very High Risk'}
        }
    }

class BatchProcessor:
    """Batch processing engine for credit risk assessment"""

    def __init__(self):
        self.required_columns = [
            'annual_income', 'credit_score', 'loan_amount', 
            'employment_length', 'debt_to_income', 'credit_utilization'
        ]
        self.feature_weights = {
            'credit_score': 0.25,
            'annual_income': 0.20,
            'debt_to_income': 0.20,
            'employment_length': 0.15,
            'loan_amount': 0.10,
            'credit_utilization': 0.10
        }

    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate uploaded data"""
        errors = []

        # Check required columns
        missing_cols = [col for col in self.required_columns if col not in df.columns]
        if missing_cols:
            errors.append(f"Missing required columns: {', '.join(missing_cols)}")

        # Check data types and ranges
        if 'credit_score' in df.columns:
            invalid_scores = df[(df['credit_score'] < 300) | (df['credit_score'] > 850)]
            if not invalid_scores.empty:
                errors.append(f"Invalid credit scores found in {len(invalid_scores)} rows")

        if 'annual_income' in df.columns:
            invalid_income = df[df['annual_income'] <= 0]
            if not invalid_income.empty:
                errors.append(f"Invalid income values found in {len(invalid_income)} rows")

        return len(errors) == 0, errors

    def calculate_batch_risk_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk scores for batch data"""
        results = df.copy()
        risk_scores = []
        risk_categories = []
        decisions = []

        for _, row in df.iterrows():
            # Calculate risk score using same logic as individual prediction
            score = 0.0

            # Credit Score (higher is better)
            credit_score_norm = min(row.get('credit_score', 600) / 850, 1.0)
            score += (1 - credit_score_norm) * self.feature_weights['credit_score']

            # Annual Income (higher is better)
            income_risk = max(0, (100000 - row.get('annual_income', 50000)) / 100000)
            score += income_risk * self.feature_weights['annual_income']

            # Debt to Income (lower is better)
            score += row.get('debt_to_income', 0.3) * self.feature_weights['debt_to_income']

            # Employment Length (higher is better)
            emp_risk = max(0, (10 - row.get('employment_length', 5)) / 10)
            score += emp_risk * self.feature_weights['employment_length']

            # Loan Amount (higher amount = higher risk)
            loan_risk = min(row.get('loan_amount', 25000) / 100000, 1.0)
            score += loan_risk * self.feature_weights['loan_amount']

            # Credit Utilization (lower is better)
            score += row.get('credit_utilization', 0.3) * self.feature_weights['credit_utilization']

            final_score = min(score, 1.0)
            risk_scores.append(final_score)

            # Determine risk category
            risk_cat = self.get_risk_category(final_score)
            risk_categories.append(risk_cat['label'])

            # Make decision
            if final_score <= 0.3:
                decisions.append('APPROVED')
            elif final_score <= 0.6:
                decisions.append('REVIEW')
            else:
                decisions.append('REJECTED')

        results['risk_score'] = risk_scores
        results['risk_category'] = risk_categories
        results['decision'] = decisions

        return results

    def get_risk_category(self, score: float) -> Dict:
        """Get risk category based on score"""
        for category, config in RISK_CONFIG['risk_categories'].items():
            if config['min'] <= score <= config['max']:
                return config
        return RISK_CONFIG['risk_categories']['medium']

def create_sample_template() -> pd.DataFrame:
    """Create sample data template for download"""
    sample_data = {
        'customer_id': [f'CUST_{i:04d}' for i in range(1, 11)],
        'annual_income': [45000, 75000, 120000, 35000, 95000, 60000, 85000, 55000, 110000, 40000],
        'credit_score': [720, 680, 780, 620, 750, 690, 740, 650, 800, 580],
        'loan_amount': [25000, 45000, 80000, 15000, 60000, 30000, 50000, 20000, 75000, 12000],
        'employment_length': [5.0, 3.5, 8.0, 1.5, 6.0, 4.0, 7.5, 2.0, 10.0, 0.5],
        'debt_to_income': [0.25, 0.35, 0.20, 0.45, 0.30, 0.40, 0.25, 0.50, 0.15, 0.55],
        'credit_utilization': [0.20, 0.45, 0.15, 0.60, 0.25, 0.35, 0.20, 0.50, 0.10, 0.70],
        'loan_purpose': ['debt_consolidation', 'home_improvement', 'major_purchase', 'car', 
                        'medical', 'vacation', 'other', 'debt_consolidation', 'home_improvement', 'car'],
        'home_ownership': ['RENT', 'MORTGAGE', 'OWN', 'RENT', 'MORTGAGE', 'RENT', 'OWN', 'RENT', 'OWN', 'RENT']
    }
    return pd.DataFrame(sample_data)

def create_results_summary_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create summary chart of batch results"""
    decision_counts = results_df['decision'].value_counts()

    colors = {
        'APPROVED': THEME_CONFIG['success_color'],
        'REVIEW': THEME_CONFIG['warning_color'],
        'REJECTED': THEME_CONFIG['error_color']
    }

    fig = go.Figure(data=[go.Pie(
        labels=decision_counts.index,
        values=decision_counts.values,
        hole=0.4,
        marker=dict(
            colors=[colors.get(label, '#666') for label in decision_counts.index],
            line=dict(color='#000000', width=2)
        )
    )])

    fig.update_layout(
        title="Batch Processing Results",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )

    return fig

def create_risk_distribution_chart(results_df: pd.DataFrame) -> go.Figure:
    """Create risk score distribution histogram"""
    fig = go.Figure(data=[go.Histogram(
        x=results_df['risk_score'],
        nbinsx=20,
        marker=dict(
            color=results_df['risk_score'],
            colorscale='RdYlBu_r',
            showscale=True,
            colorbar=dict(title="Risk Score")
        )
    )])

    fig.update_layout(
        title="Risk Score Distribution",
        xaxis_title="Risk Score",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    return fig

def show():
    """Main batch prediction display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .upload-area {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 2px dashed #00D4FF;
        text-align: center;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }

    .upload-area:hover {
        border-color: #0099CC;
        background: linear-gradient(135deg, #2D2D2D 0%, #3D3D3D 100%);
    }

    .processing-card {
        background: rgba(0, 212, 255, 0.1);
        padding: 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.3);
        margin: 1rem 0;
    }

    .results-summary {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .metric-row {
        display: flex;
        justify-content: space-around;
        margin: 1rem 0;
    }

    .metric-item {
        text-align: center;
        padding: 1rem;
        background: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
        border: 1px solid #333;
        min-width: 120px;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00D4FF;
    }

    .metric-label {
        color: #B0B0B0;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }

    .error-message {
        background: rgba(255, 75, 75, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FF4B4B;
        color: #FF4B4B;
        margin: 0.5rem 0;
    }

    .success-message {
        background: rgba(0, 255, 136, 0.1);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #00FF88;
        color: #00FF88;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize batch processor
    processor = BatchProcessor()

    # Header
    st.markdown("## 📋 Batch Credit Risk Assessment")
    st.markdown("Upload multiple applications for efficient batch processing and analysis")

    # File Upload Section
    st.markdown("### 📤 Upload Data")

    col_upload, col_template = st.columns([2, 1])

    with col_upload:
        uploaded_file = st.file_uploader(
            "Choose CSV or Excel file",
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing customer data for batch processing"
        )

        if uploaded_file is not None:
            try:
                # Read uploaded file
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)

                st.success(f"✅ File uploaded successfully! {len(df)} records found.")

                # Display file info
                st.markdown(f"""
                <div class="processing-card">
                    <strong>📊 File Information:</strong><br>
                    • Filename: {uploaded_file.name}<br>
                    • Records: {len(df):,}<br>
                    • Columns: {len(df.columns)}<br>
                    • Size: {uploaded_file.size / 1024:.1f} KB
                </div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error reading file: {str(e)}")
                df = None
        else:
            df = None

    with col_template:
        st.markdown("**📋 Need a template?**")
        sample_df = create_sample_template()

        # Convert to CSV for download
        csv_buffer = io.StringIO()
        sample_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()

        st.download_button(
            label="📥 Download Template",
            data=csv_data,
            file_name="credit_batch_template.csv",
            mime="text/csv",
            help="Download a sample template with required columns"
        )

        # Show template preview
        with st.expander("👀 Preview Template"):
            st.dataframe(sample_df.head(3), use_container_width=True)

    # Data Processing Section
    if df is not None:
        st.markdown("---")
        st.markdown("### 🔍 Data Validation & Processing")

        # Validate data
        is_valid, errors = processor.validate_data(df)

        if not is_valid:
            st.markdown("**❌ Validation Errors:**")
            for error in errors:
                st.markdown(f"""
                <div class="error-message">
                    {error}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-message">
                ✅ Data validation passed! Ready for processing.
            </div>
            """, unsafe_allow_html=True)

        # Data preview
        st.markdown("**📊 Data Preview:**")
        st.dataframe(df.head(10), use_container_width=True)

        # Processing button
        if st.button("🚀 Process Batch", disabled=not is_valid, type="primary"):
            with st.spinner("Processing batch data..."):
                # Simulate processing time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

                # Process the data
                results_df = processor.calculate_batch_risk_scores(df)

                # Store results in session state
                st.session_state.batch_results = results_df

                st.success("✅ Batch processing completed!")

        # Results Section
        if 'batch_results' in st.session_state:
            results_df = st.session_state.batch_results

            st.markdown("---")
            st.markdown("### 📊 Processing Results")

            # Summary metrics
            total_records = len(results_df)
            approved = len(results_df[results_df['decision'] == 'APPROVED'])
            review = len(results_df[results_df['decision'] == 'REVIEW'])
            rejected = len(results_df[results_df['decision'] == 'REJECTED'])
            avg_risk_score = results_df['risk_score'].mean()

            # Metrics display
            col1, col2, col3, col4, col5 = st.columns(5)

            with col1:
                st.metric("Total Records", f"{total_records:,}")
            with col2:
                st.metric("Approved", f"{approved:,}", f"{approved/total_records:.1%}")
            with col3:
                st.metric("Review Required", f"{review:,}", f"{review/total_records:.1%}")
            with col4:
                st.metric("Rejected", f"{rejected:,}", f"{rejected/total_records:.1%}")
            with col5:
                st.metric("Avg Risk Score", f"{avg_risk_score:.1%}")

            # Charts
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                fig_summary = create_results_summary_chart(results_df)
                st.plotly_chart(fig_summary, use_container_width=True)

            with col_chart2:
                fig_distribution = create_risk_distribution_chart(results_df)
                st.plotly_chart(fig_distribution, use_container_width=True)

            # Detailed results table
            st.markdown("### 📋 Detailed Results")

            # Filter options
            col_filter1, col_filter2 = st.columns(2)

            with col_filter1:
                decision_filter = st.multiselect(
                    "Filter by Decision",
                    options=['APPROVED', 'REVIEW', 'REJECTED'],
                    default=['APPROVED', 'REVIEW', 'REJECTED']
                )

            with col_filter2:
                risk_threshold = st.slider(
                    "Risk Score Threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=(0.0, 1.0),
                    step=0.01
                )

            # Apply filters
            filtered_df = results_df[
                (results_df['decision'].isin(decision_filter)) &
                (results_df['risk_score'] >= risk_threshold[0]) &
                (results_df['risk_score'] <= risk_threshold[1])
            ]

            st.dataframe(
                filtered_df.style.format({
                    'risk_score': '{:.1%}',
                    'annual_income': '${:,.0f}',
                    'loan_amount': '${:,.0f}',
                    'debt_to_income': '{:.1%}',
                    'credit_utilization': '{:.1%}'
                }),
                use_container_width=True
            )

            # Export options
            st.markdown("### 📤 Export Results")

            col_export1, col_export2, col_export3 = st.columns(3)

            with col_export1:
                # Export to CSV
                csv_buffer = io.StringIO()
                results_df.to_csv(csv_buffer, index=False)
                csv_data = csv_buffer.getvalue()

                st.download_button(
                    label="📄 Download CSV",
                    data=csv_data,
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

            with col_export2:
                # Export to Excel
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='xlsxwriter') as writer:
                    results_df.to_excel(writer, sheet_name='Results', index=False)

                    # Add summary sheet
                    summary_data = {
                        'Metric': ['Total Records', 'Approved', 'Review Required', 'Rejected', 'Average Risk Score'],
                        'Value': [total_records, approved, review, rejected, f"{avg_risk_score:.1%}"]
                    }
                    pd.DataFrame(summary_data).to_excel(writer, sheet_name='Summary', index=False)

                st.download_button(
                    label="📊 Download Excel",
                    data=excel_buffer.getvalue(),
                    file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

            with col_export3:
                # Export summary report
                summary_report = f"""
Batch Credit Risk Assessment Report
==================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Summary Statistics:
- Total Records Processed: {total_records:,}
- Approved Applications: {approved:,} ({approved/total_records:.1%})
- Applications Requiring Review: {review:,} ({review/total_records:.1%})
- Rejected Applications: {rejected:,} ({rejected/total_records:.1%})
- Average Risk Score: {avg_risk_score:.1%}

Risk Distribution:
{results_df['risk_category'].value_counts().to_string()}

Processing Details:
- File: {uploaded_file.name if uploaded_file else 'N/A'}
- Processing Time: <1 minute
- Model Version: 2.0.0
                """

                st.download_button(
                    label="📋 Download Report",
                    data=summary_report,
                    file_name=f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"💡 Tip: Use the template for best results | Maximum file size: 50MB | Supported formats: CSV, Excel"
        f"</div>", 
        unsafe_allow_html=True
    )
