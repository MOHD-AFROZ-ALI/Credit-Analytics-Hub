"""
Individual Prediction Page for CreditAnalyticsHub
===============================================
Interactive form for individual credit risk assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
    RISK_CONFIG = get_config('risk')
    BUSINESS_RULES = get_config('business_rules')
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
    BUSINESS_RULES = {'minimum_age': 18, 'maximum_age': 75, 'minimum_income': 25000}

class RiskCalculator:
    """Credit risk calculation engine"""

    def __init__(self):
        self.feature_weights = {
            'credit_score': 0.25,
            'annual_income': 0.20,
            'debt_to_income': 0.20,
            'employment_length': 0.15,
            'loan_amount': 0.10,
            'credit_utilization': 0.10
        }

    def calculate_risk_score(self, data: Dict) -> Tuple[float, Dict]:
        """Calculate risk score and feature contributions"""
        score = 0.0
        contributions = {}

        # Credit Score (higher is better)
        credit_score_norm = min(data.get('credit_score', 600) / 850, 1.0)
        credit_contrib = (1 - credit_score_norm) * self.feature_weights['credit_score']
        contributions['Credit Score'] = credit_contrib
        score += credit_contrib

        # Annual Income (higher is better)
        income_risk = max(0, (100000 - data.get('annual_income', 50000)) / 100000)
        income_contrib = income_risk * self.feature_weights['annual_income']
        contributions['Annual Income'] = income_contrib
        score += income_contrib

        # Debt to Income (lower is better)
        dti_contrib = data.get('debt_to_income', 0.3) * self.feature_weights['debt_to_income']
        contributions['Debt-to-Income'] = dti_contrib
        score += dti_contrib

        # Employment Length (higher is better)
        emp_risk = max(0, (10 - data.get('employment_length', 5)) / 10)
        emp_contrib = emp_risk * self.feature_weights['employment_length']
        contributions['Employment Length'] = emp_contrib
        score += emp_contrib

        # Loan Amount (higher amount = higher risk)
        loan_risk = min(data.get('loan_amount', 25000) / 100000, 1.0)
        loan_contrib = loan_risk * self.feature_weights['loan_amount']
        contributions['Loan Amount'] = loan_contrib
        score += loan_contrib

        # Credit Utilization (lower is better)
        util_contrib = data.get('credit_utilization', 0.3) * self.feature_weights['credit_utilization']
        contributions['Credit Utilization'] = util_contrib
        score += util_contrib

        return min(score, 1.0), contributions

def get_risk_category(score: float) -> Dict:
    """Get risk category based on score"""
    for category, config in RISK_CONFIG['risk_categories'].items():
        if config['min'] <= score <= config['max']:
            return config
    return RISK_CONFIG['risk_categories']['medium']

def create_risk_gauge(score: float) -> go.Figure:
    """Create risk score gauge chart"""
    risk_cat = get_risk_category(score)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'color': 'white', 'size': 20}},
        number={'font': {'color': 'white', 'size': 30}},
        gauge={
            'axis': {'range': [None, 100], 'tickcolor': 'white'},
            'bar': {'color': risk_cat['color'], 'thickness': 0.8},
            'steps': [
                {'range': [0, 20], 'color': 'rgba(0, 255, 136, 0.3)'},
                {'range': [20, 40], 'color': 'rgba(144, 238, 144, 0.3)'},
                {'range': [40, 60], 'color': 'rgba(255, 184, 0, 0.3)'},
                {'range': [60, 80], 'color': 'rgba(255, 140, 0, 0.3)'},
                {'range': [80, 100], 'color': 'rgba(255, 75, 75, 0.3)'}
            ],
            'borderwidth': 2,
            'bordercolor': "white"
        }
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=300
    )

    return fig

def create_feature_importance_chart(contributions: Dict) -> go.Figure:
    """Create feature importance horizontal bar chart"""
    features = list(contributions.keys())
    values = [v * 100 for v in contributions.values()]

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation='h',
        marker=dict(
            color=values,
            colorscale='RdYlBu_r',
            showscale=True,
            # colorbar=dict(title="Risk Contribution (%)", titlefont=dict(color='white'))
            colorbar=dict(title="Risk Contribution (%)", title=dict(font=dict(color='white')))
        ),
        text=[f'{v:.1f}%' for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title="Feature Risk Contributions",
        xaxis_title="Risk Contribution (%)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    return fig

def generate_recommendations(data: Dict, risk_score: float) -> List[str]:
    """Generate personalized recommendations"""
    recommendations = []

    if data.get('credit_score', 600) < 650:
        recommendations.append("🔧 Improve credit score by paying bills on time and reducing credit utilization")

    if data.get('debt_to_income', 0.3) > 0.4:
        recommendations.append("💰 Reduce debt-to-income ratio by paying down existing debts")

    if data.get('credit_utilization', 0.3) > 0.3:
        recommendations.append("💳 Lower credit utilization below 30% for better risk profile")

    if data.get('employment_length', 5) < 2:
        recommendations.append("👔 Longer employment history would improve creditworthiness")

    if risk_score > 0.6:
        recommendations.append("⚠️ Consider a co-signer or collateral to reduce risk")
        recommendations.append("📋 Provide additional documentation to support application")

    if not recommendations:
        recommendations.append("✅ Excellent credit profile! You qualify for our best rates")

    return recommendations

def export_results(data: Dict, risk_score: float, contributions: Dict, recommendations: List[str]):
    """Export analysis results to JSON"""
    results = {
        'timestamp': datetime.now().isoformat(),
        'customer_data': data,
        'risk_score': risk_score,
        'risk_category': get_risk_category(risk_score)['label'],
        'feature_contributions': contributions,
        'recommendations': recommendations,
        'analysis_version': '2.0.0'
    }

    return json.dumps(results, indent=2)

def show():
    """Main individual prediction display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .prediction-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin-bottom: 2rem;
    }

    .risk-display {
        text-align: center;
        padding: 2rem;
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .risk-score {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
    }

    .risk-label {
        font-size: 1.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }

    .recommendation-card {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00D4FF;
        margin: 0.5rem 0;
        color: #E0E0E0;
    }

    .breakdown-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem;
        background: rgba(30, 30, 30, 0.5);
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #333;
    }

    .breakdown-feature {
        color: #FFFFFF;
        font-weight: 500;
    }

    .breakdown-value {
        color: #00D4FF;
        font-weight: 600;
    }

    .export-button {
        background: linear-gradient(135deg, #00D4FF 0%, #0099CC 100%);
        color: #000;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize risk calculator
    calculator = RiskCalculator()

    # Header
    st.markdown("## 👤 Individual Credit Risk Assessment")
    st.markdown("Enter customer information to get real-time risk analysis and recommendations")

    # Input Form
    with st.container():
        st.markdown("### 📝 Customer Information")

        col1, col2 = st.columns(2)

        with col1:
            # Personal Information
            st.markdown("**Personal Details**")
            age = st.number_input("Age", min_value=18, max_value=100, value=35, step=1)
            annual_income = st.number_input("Annual Income ($)", min_value=0, value=75000, step=1000)
            employment_length = st.number_input("Employment Length (years)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)

            # Loan Information
            st.markdown("**Loan Details**")
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=1000, value=25000, step=1000)
            loan_purpose = st.selectbox("Loan Purpose", [
                "debt_consolidation", "home_improvement", "major_purchase", 
                "medical", "car", "vacation", "other"
            ])

        with col2:
            # Credit Information
            st.markdown("**Credit Profile**")
            credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=720, step=1)
            credit_utilization = st.slider("Credit Utilization (%)", min_value=0, max_value=100, value=25, step=1) / 100
            debt_to_income = st.slider("Debt-to-Income Ratio (%)", min_value=0, max_value=100, value=30, step=1) / 100

            # Additional Information
            st.markdown("**Additional Details**")
            home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
            employment_status = st.selectbox("Employment Status", ["EMPLOYED", "SELF_EMPLOYED", "UNEMPLOYED", "RETIRED"])

    # Collect input data
    input_data = {
        'age': age,
        'annual_income': annual_income,
        'employment_length': employment_length,
        'loan_amount': loan_amount,
        'loan_purpose': loan_purpose,
        'credit_score': credit_score,
        'credit_utilization': credit_utilization,
        'debt_to_income': debt_to_income,
        'home_ownership': home_ownership,
        'employment_status': employment_status
    }

    # Calculate risk score
    risk_score, contributions = calculator.calculate_risk_score(input_data)
    risk_category = get_risk_category(risk_score)

    # Real-time Results
    st.markdown("---")
    st.markdown("### 🎯 Risk Assessment Results")

    # Risk Score Display
    col_gauge, col_details = st.columns([1, 1])

    with col_gauge:
        # Risk gauge
        fig_gauge = create_risk_gauge(risk_score)
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Risk category display
        st.markdown(f"""
        <div class="risk-display">
            <div class="risk-label" style="color: {risk_category['color']}">
                {risk_category['label']}
            </div>
            <div style="color: #B0B0B0; font-size: 1rem;">
                Risk Score: {risk_score:.1%}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_details:
        # Feature importance chart
        fig_importance = create_feature_importance_chart(contributions)
        st.plotly_chart(fig_importance, use_container_width=True)

    # Detailed Breakdown
    st.markdown("### 📊 Risk Factor Breakdown")

    col_breakdown1, col_breakdown2 = st.columns(2)

    with col_breakdown1:
        for i, (feature, contribution) in enumerate(list(contributions.items())[:3]):
            st.markdown(f"""
            <div class="breakdown-item">
                <span class="breakdown-feature">{feature}</span>
                <span class="breakdown-value">{contribution:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    with col_breakdown2:
        for feature, contribution in list(contributions.items())[3:]:
            st.markdown(f"""
            <div class="breakdown-item">
                <span class="breakdown-feature">{feature}</span>
                <span class="breakdown-value">{contribution:.1%}</span>
            </div>
            """, unsafe_allow_html=True)

    # Recommendations
    st.markdown("### 💡 Personalized Recommendations")
    recommendations = generate_recommendations(input_data, risk_score)

    for rec in recommendations:
        st.markdown(f"""
        <div class="recommendation-card">
            {rec}
        </div>
        """, unsafe_allow_html=True)

    # Decision Summary
    st.markdown("### 📋 Decision Summary")

    if risk_score <= 0.3:
        decision = "✅ **APPROVED** - Excellent credit profile"
        decision_color = THEME_CONFIG['success_color']
    elif risk_score <= 0.6:
        decision = "⚠️ **REVIEW REQUIRED** - Moderate risk detected"
        decision_color = THEME_CONFIG['warning_color']
    else:
        decision = "❌ **HIGH RISK** - Additional documentation required"
        decision_color = THEME_CONFIG['error_color']

    st.markdown(f"""
    <div style="background: rgba(30, 30, 30, 0.8); padding: 1.5rem; border-radius: 10px; 
                border-left: 4px solid {decision_color}; margin: 1rem 0;">
        <div style="color: {decision_color}; font-size: 1.2rem; font-weight: 600;">
            {decision}
        </div>
        <div style="color: #B0B0B0; margin-top: 0.5rem;">
            Based on risk score of {risk_score:.1%} and current business rules
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Export Functionality
    st.markdown("### 📤 Export Results")

    col_export1, col_export2, col_export3 = st.columns(3)

    with col_export1:
        if st.button("📄 Export to JSON", key="export_json"):
            json_data = export_results(input_data, risk_score, contributions, recommendations)
            st.download_button(
                label="Download JSON Report",
                data=json_data,
                file_name=f"credit_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

    with col_export2:
        if st.button("📊 Export Summary", key="export_summary"):
            summary = f"""
Credit Risk Analysis Summary
===========================
Customer: {employment_status} | Age: {age}
Risk Score: {risk_score:.1%} ({risk_category['label']})
Decision: {decision.split('**')[1]}

Key Factors:
{chr(10).join([f"• {k}: {v:.1%}" for k, v in contributions.items()])}

Recommendations:
{chr(10).join([f"• {rec}" for rec in recommendations])}

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            """
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"credit_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

    with col_export3:
        if st.button("🔄 Reset Form", key="reset_form"):
            st.experimental_rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"⚡ Analysis completed in real-time | Last updated: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
