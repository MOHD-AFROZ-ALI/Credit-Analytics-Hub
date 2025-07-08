"""
Dashboard Page for CreditAnalyticsHub
====================================
Main dashboard with system overview, metrics, and quick access
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import random
from typing import Dict, List, Tuple

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
    RISK_CONFIG = get_config('risk')
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }
    RISK_CONFIG = {
        'risk_categories': {
            'very_low': {'color': '#00FF88', 'label': 'Very Low Risk'},
            'low': {'color': '#90EE90', 'label': 'Low Risk'},
            'medium': {'color': '#FFB800', 'label': 'Medium Risk'},
            'high': {'color': '#FF8C00', 'label': 'High Risk'},
            'very_high': {'color': '#FF4B4B', 'label': 'Very High Risk'}
        }
    }

def generate_sample_data():
    """Generate sample data for dashboard metrics"""
    np.random.seed(42)

    # System metrics
    system_metrics = {
        'total_applications': 15847,
        'approved_today': 234,
        'pending_review': 89,
        'rejected_today': 45,
        'approval_rate': 78.5,
        'avg_processing_time': 2.3,
        'system_uptime': 99.8,
        'active_models': 4
    }

    # Recent activity
    activities = [
        {'time': '2 min ago', 'action': 'New application processed', 'status': 'approved', 'amount': '$45,000'},
        {'time': '5 min ago', 'action': 'Batch prediction completed', 'status': 'completed', 'amount': '500 records'},
        {'time': '12 min ago', 'action': 'Model retrained', 'status': 'success', 'amount': 'XGBoost'},
        {'time': '18 min ago', 'action': 'Risk alert triggered', 'status': 'warning', 'amount': 'High risk detected'},
        {'time': '25 min ago', 'action': 'Compliance report generated', 'status': 'completed', 'amount': 'Monthly report'}
    ]

    # Performance data
    dates = pd.date_range(start='2025-01-01', end='2025-07-10', freq='W')
    performance_data = pd.DataFrame({
        'date': dates,
        'applications': np.random.poisson(500, len(dates)),
        'approvals': np.random.poisson(380, len(dates)),
        'accuracy': np.random.normal(0.85, 0.02, len(dates)).clip(0.75, 0.95)
    })

    # Risk distribution
    risk_dist = {
        'Very Low': 25,
        'Low': 35,
        'Medium': 25,
        'High': 12,
        'Very High': 3
    }

    return system_metrics, activities, performance_data, risk_dist

def create_metric_card(title: str, value: str, delta: str = None, delta_color: str = "normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta:
        color = THEME_CONFIG['success_color'] if delta_color == "normal" else THEME_CONFIG['error_color']
        delta_html = f'<div style="color: {color}; font-size: 0.9rem; margin-top: 0.5rem;">📈 {delta}</div>'

    card_html = f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """
    return card_html

def create_status_indicator(status: str, label: str):
    """Create status indicator"""
    colors = {
        'online': THEME_CONFIG['success_color'],
        'warning': THEME_CONFIG['warning_color'],
        'offline': THEME_CONFIG['error_color']
    }

    return f"""
    <div class="status-indicator">
        <div class="status-dot" style="background: {colors.get(status, '#666')}"></div>
        <span>{label}</span>
    </div>
    """

def create_quick_access_card(title: str, description: str, icon: str, action: str):
    """Create quick access card"""
    return f"""
    <div class="quick-access-card" onclick="alert('Navigate to {action}')">
        <div class="card-icon">{icon}</div>
        <div class="card-content">
            <h3>{title}</h3>
            <p>{description}</p>
        </div>
        <div class="card-arrow">→</div>
    </div>
    """

def create_performance_chart(data: pd.DataFrame):
    """Create performance trend chart"""
    fig = go.Figure()

    # Applications line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['applications'],
        mode='lines+markers',
        name='Applications',
        line=dict(color=THEME_CONFIG['primary_color'], width=3),
        marker=dict(size=6)
    ))

    # Approvals line
    fig.add_trace(go.Scatter(
        x=data['date'],
        y=data['approvals'],
        mode='lines+markers',
        name='Approvals',
        line=dict(color=THEME_CONFIG['success_color'], width=3),
        marker=dict(size=6)
    ))

    fig.update_layout(
        title="Daily Application Trends",
        xaxis_title="Date",
        yaxis_title="Count",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')

    return fig

def create_risk_distribution_chart(risk_data: Dict):
    """Create risk distribution pie chart"""
    colors = [
        THEME_CONFIG['success_color'],
        '#90EE90',
        THEME_CONFIG['warning_color'],
        '#FF8C00',
        THEME_CONFIG['error_color']
    ]

    fig = go.Figure(data=[go.Pie(
        labels=list(risk_data.keys()),
        values=list(risk_data.values()),
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='#000000', width=2))
    )])

    fig.update_layout(
        title="Risk Distribution",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="middle",
            y=0.5,
            xanchor="left",
            x=1.05
        )
    )

    return fig

def create_accuracy_gauge(accuracy: float):
    """Create model accuracy gauge"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=accuracy * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Model Accuracy (%)"},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [None, 100]},
            'bar': {'color': THEME_CONFIG['primary_color']},
            'steps': [
                {'range': [0, 50], 'color': THEME_CONFIG['error_color']},
                {'range': [50, 80], 'color': THEME_CONFIG['warning_color']},
                {'range': [80, 100], 'color': THEME_CONFIG['success_color']}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white', size=12),
        height=300
    )

    return fig

def show():
    """Main dashboard display function"""
    # Load custom CSS
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        text-align: center;
        transition: transform 0.3s ease;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 212, 255, 0.2);
    }

    .metric-title {
        color: #B0B0B0;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
    }

    .metric-value {
        color: #00D4FF;
        font-size: 2rem;
        font-weight: 700;
    }

    .status-indicator {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 1rem;
        background: rgba(0, 212, 255, 0.1);
        border-radius: 25px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 0.25rem;
    }

    .status-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        animation: pulse 2s infinite;
    }

    .quick-access-card {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        cursor: pointer;
        transition: all 0.3s ease;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1rem;
    }

    .quick-access-card:hover {
        transform: translateX(10px);
        border-color: #00D4FF;
        box-shadow: 0 5px 20px rgba(0, 212, 255, 0.2);
    }

    .card-icon {
        font-size: 2rem;
        color: #00D4FF;
    }

    .card-content h3 {
        color: #FFFFFF;
        margin: 0 0 0.5rem 0;
        font-size: 1.1rem;
    }

    .card-content p {
        color: #B0B0B0;
        margin: 0;
        font-size: 0.9rem;
    }

    .card-arrow {
        color: #00D4FF;
        font-size: 1.5rem;
        margin-left: auto;
    }

    .activity-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background: rgba(30, 30, 30, 0.5);
        border-radius: 10px;
        border: 1px solid #333;
        margin-bottom: 0.5rem;
    }

    .activity-time {
        color: #B0B0B0;
        font-size: 0.8rem;
    }

    .activity-action {
        color: #FFFFFF;
        font-weight: 500;
    }

    .activity-status {
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.8rem;
        font-weight: 600;
    }

    .status-approved { background: #00FF88; color: #000; }
    .status-completed { background: #00D4FF; color: #000; }
    .status-warning { background: #FFB800; color: #000; }
    .status-success { background: #00FF88; color: #000; }

    .alert-card {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF6B6B 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 1rem;
        border-left: 4px solid #FF0000;
    }

    .alert-title {
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Generate sample data
    metrics, activities, performance_data, risk_dist = generate_sample_data()

    # Header
    st.markdown("## 📊 System Dashboard")
    st.markdown("Real-time overview of your credit risk analytics platform")

    # System Status Row
    st.markdown("### 🔄 System Status")
    status_col1, status_col2, status_col3, status_col4 = st.columns(4)

    with status_col1:
        st.markdown(create_status_indicator('online', 'Models Online'), unsafe_allow_html=True)
    with status_col2:
        st.markdown(create_status_indicator('online', 'Data Connected'), unsafe_allow_html=True)
    with status_col3:
        st.markdown(create_status_indicator('online', 'API Active'), unsafe_allow_html=True)
    with status_col4:
        st.markdown(create_status_indicator('online', f"Uptime: {metrics['system_uptime']}%"), unsafe_allow_html=True)

    st.markdown("---")

    # Key Metrics Row
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(create_metric_card(
            "Total Applications", 
            f"{metrics['total_applications']:,}",
            "+12% vs last month"
        ), unsafe_allow_html=True)

    with col2:
        st.markdown(create_metric_card(
            "Approval Rate", 
            f"{metrics['approval_rate']}%",
            "+2.3% vs yesterday"
        ), unsafe_allow_html=True)

    with col3:
        st.markdown(create_metric_card(
            "Avg Processing Time", 
            f"{metrics['avg_processing_time']} min",
            "-0.5 min vs yesterday"
        ), unsafe_allow_html=True)

    with col4:
        st.markdown(create_metric_card(
            "Active Models", 
            str(metrics['active_models']),
            "All operational"
        ), unsafe_allow_html=True)

    st.markdown("---")

    # Main Content Row
    col_left, col_right = st.columns([2, 1])

    with col_left:
        # Performance Charts
        st.markdown("### 📊 Performance Trends")

        # Daily trends
        fig_performance = create_performance_chart(performance_data)
        st.plotly_chart(fig_performance, use_container_width=True)

        # Model accuracy gauge
        st.markdown("### 🎯 Model Accuracy")
        col_gauge1, col_gauge2 = st.columns(2)

        with col_gauge1:
            fig_accuracy = create_accuracy_gauge(0.847)
            st.plotly_chart(fig_accuracy, use_container_width=True)

        with col_gauge2:
            # Risk distribution
            fig_risk = create_risk_distribution_chart(risk_dist)
            st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        # Quick Access Cards
        st.markdown("### 🚀 Quick Access")

        quick_access_cards = [
            ("👤 New Prediction", "Analyze individual credit risk", "🔍", "individual"),
            ("📋 Batch Process", "Upload and process multiple applications", "⚡", "batch"),
            ("📊 Data Explorer", "Explore and analyze your data", "🔬", "explorer"),
            ("🧠 SHAP Analysis", "AI model explanations", "🤖", "shap"),
            ("📈 Performance", "Model performance metrics", "📊", "performance")
        ]

        for title, desc, icon, action in quick_access_cards:
            st.markdown(create_quick_access_card(title, desc, icon, action), unsafe_allow_html=True)

        # Recent Activity
        st.markdown("### 📝 Recent Activity")

        for activity in activities[:5]:
            status_class = f"status-{activity['status']}"
            activity_html = f"""
            <div class="activity-item">
                <div>
                    <div class="activity-action">{activity['action']}</div>
                    <div class="activity-time">{activity['time']} • {activity['amount']}</div>
                </div>
                <div class="activity-status {status_class}">{activity['status'].title()}</div>
            </div>
            """
            st.markdown(activity_html, unsafe_allow_html=True)

        # Alerts Section
        st.markdown("### 🚨 Active Alerts")

        # Sample alerts
        alerts = [
            {"type": "warning", "title": "Model Drift Detected", "message": "XGBoost model showing 5% performance degradation"},
            {"type": "info", "title": "Scheduled Maintenance", "message": "System maintenance scheduled for tonight 2-4 AM"}
        ]

        for alert in alerts:
            if alert["type"] == "warning":
                st.warning(f"**{alert['title']}**: {alert['message']}")
            else:
                st.info(f"**{alert['title']}**: {alert['message']}")

    # Footer Stats
    st.markdown("---")
    st.markdown("### 📊 Today's Summary")

    summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)

    with summary_col1:
        st.metric("Applications Processed", metrics['approved_today'] + metrics['rejected_today'], 
                 delta=f"+{random.randint(5, 15)} vs yesterday")

    with summary_col2:
        st.metric("Approved", metrics['approved_today'], 
                 delta=f"+{random.randint(2, 8)} vs yesterday")

    with summary_col3:
        st.metric("Pending Review", metrics['pending_review'], 
                 delta=f"-{random.randint(1, 5)} vs yesterday")

    with summary_col4:
        st.metric("Rejected", metrics['rejected_today'], 
                 delta=f"+{random.randint(1, 3)} vs yesterday")

    # Auto-refresh indicator
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"🔄 Last updated: {datetime.now().strftime('%H:%M:%S')} | Auto-refresh: Every 30 seconds"
        f"</div>", 
        unsafe_allow_html=True
    )
