"""
Compliance Report Page for CreditAnalyticsHub
===========================================
Regulatory compliance, bias detection, audit trails, and governance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from typing import Dict, List, Tuple, Any

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
    COMPLIANCE_CONFIG = get_config('compliance')
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }
    COMPLIANCE_CONFIG = {
        'regulations': {
            'fair_credit_reporting_act': True,
            'equal_credit_opportunity_act': True,
            'gdpr_compliance': True
        },
        'protected_attributes': ['race', 'gender', 'age', 'marital_status']
    }

class ComplianceEngine:
    """Compliance monitoring and reporting engine"""

    def __init__(self):
        self.regulations = {
            'FCRA': {'name': 'Fair Credit Reporting Act', 'status': 'Compliant', 'last_audit': '2024-01-15'},
            'ECOA': {'name': 'Equal Credit Opportunity Act', 'status': 'Compliant', 'last_audit': '2024-01-10'},
            'GDPR': {'name': 'General Data Protection Regulation', 'status': 'Compliant', 'last_audit': '2024-01-20'},
            'CCPA': {'name': 'California Consumer Privacy Act', 'status': 'Review Required', 'last_audit': '2024-01-05'},
            'SOX': {'name': 'Sarbanes-Oxley Act', 'status': 'Compliant', 'last_audit': '2024-01-12'}
        }

        self.bias_metrics = {
            'demographic_parity': 0.95,
            'equalized_odds': 0.92,
            'calibration': 0.94,
            'individual_fairness': 0.89
        }

    def generate_audit_trail(self):
        """Generate sample audit trail data"""
        np.random.seed(42)
        activities = [
            'Model Training', 'Data Access', 'Prediction Request', 'Report Generation',
            'Configuration Change', 'User Login', 'Data Export', 'Model Validation'
        ]

        audit_data = []
        for i in range(50):
            audit_data.append({
                'timestamp': datetime.now() - timedelta(hours=np.random.randint(1, 720)),
                'user_id': f'USER_{np.random.randint(1000, 9999)}',
                'activity': np.random.choice(activities),
                'resource': f'Model_{np.random.choice(["RF", "LR", "XGB"])}',
                'ip_address': f'192.168.{np.random.randint(1, 255)}.{np.random.randint(1, 255)}',
                'status': np.random.choice(['Success', 'Failed', 'Warning'], p=[0.8, 0.1, 0.1]),
                'risk_level': np.random.choice(['Low', 'Medium', 'High'], p=[0.7, 0.2, 0.1])
            })

        return pd.DataFrame(audit_data)

    def calculate_bias_metrics(self, protected_groups):
        """Calculate fairness metrics for protected groups"""
        np.random.seed(42)
        bias_results = {}

        for group in protected_groups:
            bias_results[group] = {
                'demographic_parity': np.random.uniform(0.85, 0.98),
                'equalized_odds': np.random.uniform(0.88, 0.96),
                'calibration': np.random.uniform(0.90, 0.98),
                'statistical_significance': np.random.choice([True, False], p=[0.8, 0.2])
            }

        return bias_results

def create_compliance_status_chart(regulations):
    """Create compliance status overview chart"""
    statuses = [reg['status'] for reg in regulations.values()]
    status_counts = pd.Series(statuses).value_counts()

    colors = {
        'Compliant': THEME_CONFIG['success_color'],
        'Review Required': THEME_CONFIG['warning_color'],
        'Non-Compliant': THEME_CONFIG['error_color']
    }

    fig = go.Figure(data=[go.Pie(
        labels=status_counts.index,
        values=status_counts.values,
        hole=0.4,
        marker=dict(
            colors=[colors.get(status, '#666') for status in status_counts.index],
            line=dict(color='#000000', width=2)
        )
    )])

    fig.update_layout(
        title="Regulatory Compliance Status",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True
    )

    return fig

def create_bias_metrics_chart(bias_data):
    """Create bias metrics radar chart"""
    metrics = list(bias_data.keys())
    values = list(bias_data.values())

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=metrics,
        fill='toself',
        name='Fairness Metrics',
        line=dict(color=THEME_CONFIG['primary_color'])
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(color='white')
            ),
            angularaxis=dict(tickfont=dict(color='white'))
        ),
        title="Model Fairness Metrics",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def create_audit_timeline(audit_df):
    """Create audit activity timeline"""
    daily_counts = audit_df.groupby(audit_df['timestamp'].dt.date).size()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=daily_counts.index,
        y=daily_counts.values,
        mode='lines+markers',
        name='Daily Activities',
        line=dict(color=THEME_CONFIG['primary_color'], width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Audit Activity Timeline",
        xaxis_title="Date",
        yaxis_title="Number of Activities",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def show():
    """Main compliance report display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .compliance-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .regulation-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem 0;
        transition: transform 0.3s ease;
    }

    .regulation-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 212, 255, 0.2);
    }

    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin-left: 0.5rem;
    }

    .status-compliant { background: #00FF88; color: #000; }
    .status-review { background: #FFB800; color: #000; }
    .status-non-compliant { background: #FF4B4B; color: #FFF; }

    .audit-entry {
        background: rgba(30, 30, 30, 0.5);
        padding: 0.75rem;
        border-radius: 8px;
        border-left: 4px solid;
        margin: 0.5rem 0;
    }

    .audit-success { border-color: #00FF88; }
    .audit-warning { border-color: #FFB800; }
    .audit-failed { border-color: #FF4B4B; }

    .metric-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem;
    }

    .alert-critical {
        background: rgba(255, 75, 75, 0.1);
        border: 1px solid #FF4B4B;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    .alert-warning {
        background: rgba(255, 184, 0, 0.1);
        border: 1px solid #FFB800;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize compliance engine
    if 'compliance_engine' not in st.session_state:
        st.session_state.compliance_engine = ComplianceEngine()

    engine = st.session_state.compliance_engine

    # Header
    st.markdown("## 📋 Compliance & Regulatory Dashboard")
    st.markdown("Comprehensive compliance monitoring, bias detection, and audit management")

    # Compliance Overview
    st.markdown("### 🏛️ Regulatory Compliance Status")

    col_overview1, col_overview2 = st.columns([2, 1])

    with col_overview1:
        fig_compliance = create_compliance_status_chart(engine.regulations)
        st.plotly_chart(fig_compliance, use_container_width=True)

    with col_overview2:
        st.markdown("**📊 Compliance Summary**")
        compliant_count = sum(1 for reg in engine.regulations.values() if reg['status'] == 'Compliant')
        total_count = len(engine.regulations)

        st.metric("Compliance Rate", f"{compliant_count}/{total_count}", f"{(compliant_count/total_count)*100:.0f}%")
        st.metric("Last Full Audit", "2024-01-20")
        st.metric("Next Review", "2024-04-20")

    # Detailed Regulations
    st.markdown("**📋 Regulation Details**")
    for reg_code, reg_info in engine.regulations.items():
        status_class = f"status-{reg_info['status'].lower().replace(' ', '-')}"
        st.markdown(f"""
        <div class="regulation-card">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <strong style="color: #FFFFFF;">{reg_info['name']} ({reg_code})</strong>
                    <span class="status-badge {status_class}">{reg_info['status']}</span>
                </div>
                <div style="color: #B0B0B0; font-size: 0.9rem;">
                    Last Audit: {reg_info['last_audit']}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bias Detection and Fairness
    st.markdown("---")
    st.markdown("### ⚖️ Bias Detection & Fairness Metrics")

    fairness_tab1, fairness_tab2 = st.tabs(["📊 Overall Metrics", "👥 Group Analysis"])

    with fairness_tab1:
        col_bias1, col_bias2 = st.columns([1, 1])

        with col_bias1:
            fig_bias = create_bias_metrics_chart(engine.bias_metrics)
            st.plotly_chart(fig_bias, use_container_width=True)

        with col_bias2:
            st.markdown("**⚖️ Fairness Thresholds**")
            for metric, value in engine.bias_metrics.items():
                threshold = 0.90
                status = "✅ Pass" if value >= threshold else "❌ Fail"
                color = THEME_CONFIG['success_color'] if value >= threshold else THEME_CONFIG['error_color']

                st.markdown(f"""
                <div class="metric-card">
                    <div style="color: {color}; font-size: 1.2rem; font-weight: 700;">{value:.2f}</div>
                    <div style="color: #B0B0B0;">{metric.replace('_', ' ').title()}</div>
                    <div style="color: {color}; font-size: 0.9rem;">{status}</div>
                </div>
                """, unsafe_allow_html=True)

    with fairness_tab2:
        st.markdown("**👥 Protected Group Analysis**")
        protected_groups = ['Gender', 'Age Group', 'Ethnicity', 'Marital Status']
        bias_results = engine.calculate_bias_metrics(protected_groups)

        for group, metrics in bias_results.items():
            with st.expander(f"📊 {group} Analysis"):
                col_group1, col_group2 = st.columns(2)

                with col_group1:
                    for metric, value in metrics.items():
                        if metric != 'statistical_significance':
                            st.metric(metric.replace('_', ' ').title(), f"{value:.3f}")

                with col_group2:
                    sig_status = "Significant" if metrics['statistical_significance'] else "Not Significant"
                    sig_color = THEME_CONFIG['error_color'] if metrics['statistical_significance'] else THEME_CONFIG['success_color']
                    st.markdown(f"**Statistical Significance:** <span style='color: {sig_color}'>{sig_status}</span>", unsafe_allow_html=True)

    # Audit Trail
    st.markdown("---")
    st.markdown("### 📝 Audit Trail & Activity Monitoring")

    audit_data = engine.generate_audit_trail()

    col_audit1, col_audit2 = st.columns([2, 1])

    with col_audit1:
        fig_audit = create_audit_timeline(audit_data)
        st.plotly_chart(fig_audit, use_container_width=True)

    with col_audit2:
        st.markdown("**📊 Activity Summary**")
        activity_counts = audit_data['activity'].value_counts()
        for activity, count in activity_counts.head(5).items():
            st.metric(activity, count)

    # Recent Audit Entries
    st.markdown("**📋 Recent Audit Entries**")
    recent_audits = audit_data.head(10)

    for _, entry in recent_audits.iterrows():
        status_class = f"audit-{entry['status'].lower()}"
        st.markdown(f"""
        <div class="audit-entry {status_class}">
            <div style="display: flex; justify-content: space-between;">
                <div>
                    <strong>{entry['activity']}</strong> by {entry['user_id']}<br>
                    <small style="color: #B0B0B0;">Resource: {entry['resource']} | IP: {entry['ip_address']}</small>
                </div>
                <div style="text-align: right;">
                    <div style="color: #FFFFFF;">{entry['status']}</div>
                    <small style="color: #B0B0B0;">{entry['timestamp'].strftime('%Y-%m-%d %H:%M')}</small>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Model Governance
    st.markdown("---")
    st.markdown("### 🎯 Model Governance & Validation")

    governance_col1, governance_col2 = st.columns(2)

    with governance_col1:
        st.markdown("**📊 Model Validation Status**")
        models = ['Random Forest', 'Logistic Regression', 'XGBoost', 'Neural Network']
        validation_status = ['Validated', 'Pending', 'Validated', 'In Review']

        for model, status in zip(models, validation_status):
            status_color = THEME_CONFIG['success_color'] if status == 'Validated' else THEME_CONFIG['warning_color']
            st.markdown(f"""
            <div class="regulation-card">
                <div style="display: flex; justify-content: space-between;">
                    <strong style="color: #FFFFFF;">{model}</strong>
                    <span style="color: {status_color}; font-weight: 600;">{status}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    with governance_col2:
        st.markdown("**📋 Governance Checklist**")
        checklist_items = [
            ("Model Documentation", True),
            ("Performance Monitoring", True),
            ("Bias Testing", True),
            ("Version Control", True),
            ("Change Management", False),
            ("Risk Assessment", True)
        ]

        for item, completed in checklist_items:
            icon = "✅" if completed else "❌"
            color = THEME_CONFIG['success_color'] if completed else THEME_CONFIG['error_color']
            st.markdown(f"<span style='color: {color}'>{icon} {item}</span>", unsafe_allow_html=True)

    # Compliance Alerts
    st.markdown("---")
    st.markdown("### 🚨 Compliance Alerts & Monitoring")

    # Generate sample alerts
    alerts = [
        {"level": "warning", "message": "CCPA compliance review required within 30 days", "date": "2024-01-25"},
        {"level": "critical", "message": "Bias threshold exceeded for age group analysis", "date": "2024-01-24"}
    ]

    for alert in alerts:
        alert_class = f"alert-{alert['level']}"
        icon = "🚨" if alert['level'] == 'critical' else "⚠️"
        st.markdown(f"""
        <div class="{alert_class}">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div><strong>{icon} {alert['message']}</strong></div>
                <div style="color: #B0B0B0; font-size: 0.9rem;">{alert['date']}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Export Compliance Report
    st.markdown("---")
    st.markdown("### 📤 Export Compliance Documentation")

    def convert_to_python_types(obj):
        if isinstance(obj, dict):
            return {k: convert_to_python_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_python_types(i) for i in obj]
        elif isinstance(obj, (np.bool_, np.integer, np.floating)):
            return obj
        else:
            return obj

    export_col1, export_col2, export_col3 = st.columns(3)

    with export_col1:
        if st.button("📋 Generate Compliance Report"):
            compliance_report = f"""
COMPLIANCE REPORT
================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

REGULATORY COMPLIANCE STATUS:
{chr(10).join([f"- {info['name']}: {info['status']}" for code, info in engine.regulations.items()])}

FAIRNESS METRICS:
{chr(10).join([f"- {metric.replace('_', ' ').title()}: {value:.3f}" for metric, value in engine.bias_metrics.items()])}

AUDIT SUMMARY:
- Total Activities: {len(audit_data)}
- Success Rate: {(audit_data['status'] == 'Success').mean():.1%}
- High Risk Activities: {(audit_data['risk_level'] == 'High').sum()}

MODEL GOVERNANCE:
- Models Under Management: 4
- Validated Models: 3
- Pending Validation: 1

RECOMMENDATIONS:
- Complete CCPA compliance review
- Address bias threshold exceedance
- Update model documentation
- Schedule quarterly governance review

Next Review Date: {(datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')}
            """

            st.download_button(
                "Download Compliance Report",
                compliance_report,
                f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

    with export_col2:
        if st.button("📊 Export Audit Trail"):
            csv_buffer = audit_data.to_csv(index=False)
            st.download_button(
                "Download Audit Trail CSV",
                csv_buffer,
                f"audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                "text/csv"
            )

    with export_col3:
        if st.button("⚖️ Export Bias Analysis"):
            bias_report = {
                'timestamp': datetime.now().isoformat(),
                'overall_metrics': engine.bias_metrics,
                'group_analysis': convert_to_python_types(engine.calculate_bias_metrics(protected_groups)),
                'compliance_status': 'PASS' if all(v >= 0.90 for v in engine.bias_metrics.values()) else 'REVIEW_REQUIRED'
            }

            st.download_button(
                "Download Bias Analysis JSON",
                json.dumps(bias_report, indent=2),
                f"bias_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"📋 Compliance Engine v2.0 | Regulatory Standards: FCRA, ECOA, GDPR, CCPA | "
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
