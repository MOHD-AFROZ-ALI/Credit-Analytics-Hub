"""
Business Intelligence Page for CreditAnalyticsHub
===============================================
Comprehensive business analytics, KPI tracking, and strategic insights
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
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }

class BusinessIntelligenceEngine:
    """Business intelligence and analytics engine"""

    def __init__(self):
        self.kpi_thresholds = {
            'approval_rate': {'target': 0.75, 'warning': 0.70, 'critical': 0.65},
            'default_rate': {'target': 0.05, 'warning': 0.08, 'critical': 0.12},
            'profit_margin': {'target': 0.15, 'warning': 0.12, 'critical': 0.08},
            'portfolio_growth': {'target': 0.10, 'warning': 0.05, 'critical': 0.00}
        }

    def generate_financial_metrics(self):
        """Generate sample financial metrics"""
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='D')

        return {
            'total_revenue': np.random.normal(2500000, 200000, len(dates)).cumsum(),
            'total_loans': np.random.normal(50000000, 5000000, len(dates)).cumsum(),
            'default_losses': np.random.normal(125000, 15000, len(dates)).cumsum(),
            'operating_costs': np.random.normal(800000, 50000, len(dates)).cumsum(),
            'net_profit': np.random.normal(400000, 30000, len(dates)).cumsum(),
            'dates': dates
        }

    def generate_portfolio_data(self):
        """Generate portfolio analysis data"""
        np.random.seed(42)
        segments = ['Prime', 'Near-Prime', 'Subprime', 'Deep Subprime']

        portfolio_data = []
        for segment in segments:
            base_default = {'Prime': 0.02, 'Near-Prime': 0.05, 'Subprime': 0.12, 'Deep Subprime': 0.25}[segment]
            base_yield = {'Prime': 0.08, 'Near-Prime': 0.12, 'Subprime': 0.18, 'Deep Subprime': 0.28}[segment]

            portfolio_data.append({
                'segment': segment,
                'loan_count': np.random.randint(1000, 5000),
                'total_amount': np.random.uniform(10000000, 50000000),
                'avg_loan_size': np.random.uniform(15000, 45000),
                'default_rate': np.random.normal(base_default, base_default * 0.2),
                'yield_rate': np.random.normal(base_yield, base_yield * 0.1),
                'profit_margin': np.random.uniform(0.05, 0.25)
            })

        return pd.DataFrame(portfolio_data)

def create_financial_dashboard(metrics_data):
    """Create financial metrics dashboard"""
    fig = go.Figure()

    # Revenue trend
    fig.add_trace(go.Scatter(
        x=metrics_data['dates'],
        y=metrics_data['total_revenue'],
        mode='lines',
        name='Total Revenue',
        line=dict(color=THEME_CONFIG['success_color'], width=3)
    ))

    # Profit trend
    fig.add_trace(go.Scatter(
        x=metrics_data['dates'],
        y=metrics_data['net_profit'],
        mode='lines',
        name='Net Profit',
        line=dict(color=THEME_CONFIG['primary_color'], width=3)
    ))

    fig.update_layout(
        title="Financial Performance Trends",
        xaxis_title="Date",
        yaxis_title="Amount ($)",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_portfolio_analysis_chart(portfolio_df):
    """Create portfolio risk-return analysis"""
    fig = go.Figure()

    # Bubble chart: Risk vs Return
    fig.add_trace(go.Scatter(
        x=portfolio_df['default_rate'],
        y=portfolio_df['yield_rate'],
        mode='markers+text',
        text=portfolio_df['segment'],
        textposition='middle center',
        marker=dict(
            size=portfolio_df['total_amount'] / 1000000,  # Size by portfolio amount
            color=portfolio_df['profit_margin'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Profit Margin"),
            sizemode='diameter',
            sizeref=2,
            line=dict(width=2, color='white')
        ),
        name='Portfolio Segments'
    ))

    fig.update_layout(
        title="Portfolio Risk-Return Analysis",
        xaxis_title="Default Rate",
        yaxis_title="Yield Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def create_kpi_gauge(value, target, title, format_type='percent'):
    """Create KPI gauge chart"""
    if format_type == 'percent':
        display_value = value * 100
        target_value = target * 100
        suffix = '%'
    else:
        display_value = value
        target_value = target
        suffix = ''

    color = THEME_CONFIG['success_color'] if value >= target else THEME_CONFIG['warning_color'] if value >= target * 0.8 else THEME_CONFIG['error_color']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=display_value,
        delta={'reference': target_value},
        title={'text': title},
        gauge={
            'axis': {'range': [None, target_value * 1.5]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, target_value * 0.8], 'color': 'rgba(255, 75, 75, 0.3)'},
                {'range': [target_value * 0.8, target_value], 'color': 'rgba(255, 184, 0, 0.3)'},
                {'range': [target_value, target_value * 1.5], 'color': 'rgba(0, 255, 136, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': target_value
            }
        },
        number={'suffix': suffix}
    ))

    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=250
    )

    return fig

def show():
    """Main business intelligence display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .bi-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .kpi-card {
        background: rgba(30, 30, 30, 0.8);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #333;
        text-align: center;
        margin: 0.5rem;
        transition: transform 0.3s ease;
    }

    .kpi-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0, 212, 255, 0.15);
    }

    .alert-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }

    .alert-success { background: rgba(0, 255, 136, 0.1); border-color: #00FF88; }
    .alert-warning { background: rgba(255, 184, 0, 0.1); border-color: #FFB800; }
    .alert-error { background: rgba(255, 75, 75, 0.1); border-color: #FF4B4B; }

    .recommendation-card {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 0.5rem 0;
    }

    .metric-trend {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize BI engine
    if 'bi_engine' not in st.session_state:
        st.session_state.bi_engine = BusinessIntelligenceEngine()

    bi_engine = st.session_state.bi_engine

    # Header
    st.markdown("## 💼 Business Intelligence Dashboard")
    st.markdown("Strategic insights, KPI tracking, and business performance analytics")

    # Generate sample data
    financial_data = bi_engine.generate_financial_metrics()
    portfolio_data = bi_engine.generate_portfolio_data()

    # KPI Overview
    st.markdown("### 📊 Key Performance Indicators")

    # Calculate current KPIs
    current_kpis = {
        'approval_rate': 0.78,
        'default_rate': 0.045,
        'profit_margin': 0.16,
        'portfolio_growth': 0.12
    }

    kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)

    with kpi_col1:
        fig_approval = create_kpi_gauge(
            current_kpis['approval_rate'], 
            bi_engine.kpi_thresholds['approval_rate']['target'],
            "Approval Rate"
        )
        st.plotly_chart(fig_approval, use_container_width=True)

    with kpi_col2:
        fig_default = create_kpi_gauge(
            current_kpis['default_rate'], 
            bi_engine.kpi_thresholds['default_rate']['target'],
            "Default Rate"
        )
        st.plotly_chart(fig_default, use_container_width=True)

    with kpi_col3:
        fig_profit = create_kpi_gauge(
            current_kpis['profit_margin'], 
            bi_engine.kpi_thresholds['profit_margin']['target'],
            "Profit Margin"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

    with kpi_col4:
        fig_growth = create_kpi_gauge(
            current_kpis['portfolio_growth'], 
            bi_engine.kpi_thresholds['portfolio_growth']['target'],
            "Portfolio Growth"
        )
        st.plotly_chart(fig_growth, use_container_width=True)

    # Financial Dashboard
    st.markdown("---")
    st.markdown("### 💰 Financial Performance")

    col_financial1, col_financial2 = st.columns([2, 1])

    with col_financial1:
        fig_financial = create_financial_dashboard(financial_data)
        st.plotly_chart(fig_financial, use_container_width=True)

    with col_financial2:
        st.markdown("**📈 Financial Metrics**")

        # Current month metrics
        current_revenue = financial_data['total_revenue'][-1]
        current_profit = financial_data['net_profit'][-1]
        current_costs = financial_data['operating_costs'][-1]

        st.metric("Total Revenue", f"${current_revenue:,.0f}", "+12.5%")
        st.metric("Net Profit", f"${current_profit:,.0f}", "+8.3%")
        st.metric("Operating Costs", f"${current_costs:,.0f}", "+3.2%")
        st.metric("ROI", "18.4%", "+2.1%")

    # Portfolio Analysis
    st.markdown("---")
    st.markdown("### 📊 Portfolio Analysis & Segmentation")

    portfolio_tab1, portfolio_tab2 = st.tabs(["🎯 Risk-Return Analysis", "📋 Segment Performance"])

    with portfolio_tab1:
        fig_portfolio = create_portfolio_analysis_chart(portfolio_data)
        st.plotly_chart(fig_portfolio, use_container_width=True)

        st.markdown("**💡 Portfolio Insights:**")
        insights = [
            "Prime segment shows optimal risk-return balance with 8% yield and 2% default rate",
            "Subprime segment offers highest profitability but requires careful risk management",
            "Near-Prime segment presents growth opportunity with moderate risk profile"
        ]

        for insight in insights:
            st.markdown(f"""
            <div class="recommendation-card">
                💡 {insight}
            </div>
            """, unsafe_allow_html=True)

    with portfolio_tab2:
        st.markdown("**📊 Segment Performance Breakdown**")

        # Enhanced portfolio table
        portfolio_display = portfolio_data.copy()
        portfolio_display['total_amount'] = portfolio_display['total_amount'].apply(lambda x: f"${x:,.0f}")
        portfolio_display['avg_loan_size'] = portfolio_display['avg_loan_size'].apply(lambda x: f"${x:,.0f}")
        portfolio_display['default_rate'] = portfolio_display['default_rate'].apply(lambda x: f"{x:.1%}")
        portfolio_display['yield_rate'] = portfolio_display['yield_rate'].apply(lambda x: f"{x:.1%}")
        portfolio_display['profit_margin'] = portfolio_display['profit_margin'].apply(lambda x: f"{x:.1%}")

        st.dataframe(portfolio_display, use_container_width=True)

        # Segment recommendations
        st.markdown("**🎯 Segment Recommendations:**")
        recommendations = {
            'Prime': "Maintain current pricing strategy, focus on volume growth",
            'Near-Prime': "Increase marketing spend, optimize approval criteria",
            'Subprime': "Implement enhanced monitoring, consider rate adjustments",
            'Deep Subprime': "Review risk appetite, strengthen collection processes"
        }

        for segment, rec in recommendations.items():
            st.markdown(f"**{segment}**: {rec}")

    # Market Trends & Alerts
    st.markdown("---")
    st.markdown("### 📈 Market Trends & KPI Alerts")

    trends_col1, trends_col2 = st.columns(2)

    with trends_col1:
        st.markdown("**📊 Market Trends**")

        # Mock market data
        market_trends = [
            {"metric": "Interest Rates", "current": "5.25%", "trend": "↑", "impact": "Positive"},
            {"metric": "Unemployment", "current": "3.8%", "trend": "↓", "impact": "Positive"},
            {"metric": "Consumer Confidence", "current": "102.3", "trend": "↑", "impact": "Positive"},
            {"metric": "Credit Demand", "current": "High", "trend": "↑", "impact": "Positive"}
        ]

        for trend in market_trends:
            trend_color = THEME_CONFIG['success_color'] if trend['impact'] == 'Positive' else THEME_CONFIG['error_color']
            st.markdown(f"""
            <div class="kpi-card">
                <div style="color: {trend_color}; font-weight: 600;">{trend['metric']}</div>
                <div style="color: #FFFFFF; font-size: 1.2rem;">{trend['current']} {trend['trend']}</div>
                <div style="color: #B0B0B0; font-size: 0.9rem;">{trend['impact']} Impact</div>
            </div>
            """, unsafe_allow_html=True)

    with trends_col2:
        st.markdown("**🚨 KPI Alerts**")

        # Generate alerts based on KPI performance
        alerts = []
        for kpi, value in current_kpis.items():
            thresholds = bi_engine.kpi_thresholds[kpi]
            if kpi == 'default_rate':  # Lower is better for default rate
                if value > thresholds['critical']:
                    alerts.append({"kpi": kpi, "level": "error", "message": f"Critical: {kpi.replace('_', ' ').title()} at {value:.1%}"})
                elif value > thresholds['warning']:
                    alerts.append({"kpi": kpi, "level": "warning", "message": f"Warning: {kpi.replace('_', ' ').title()} at {value:.1%}"})
            else:  # Higher is better for other KPIs
                if value < thresholds['critical']:
                    alerts.append({"kpi": kpi, "level": "error", "message": f"Critical: {kpi.replace('_', ' ').title()} at {value:.1%}"})
                elif value < thresholds['warning']:
                    alerts.append({"kpi": kpi, "level": "warning", "message": f"Warning: {kpi.replace('_', ' ').title()} at {value:.1%}"})

        if not alerts:
            st.markdown("""
            <div class="alert-card alert-success">
                ✅ All KPIs are performing within target ranges
            </div>
            """, unsafe_allow_html=True)
        else:
            for alert in alerts:
                alert_class = f"alert-{alert['level']}"
                icon = "🚨" if alert['level'] == 'error' else "⚠️"
                st.markdown(f"""
                <div class="alert-card {alert_class}">
                    {icon} {alert['message']}
                </div>
                """, unsafe_allow_html=True)

    # Business Recommendations
    st.markdown("---")
    st.markdown("### 💡 Strategic Business Recommendations")

    rec_col1, rec_col2 = st.columns(2)

    with rec_col1:
        st.markdown("**🎯 Short-term Actions (1-3 months)**")
        short_term = [
            "Optimize pricing for Near-Prime segment to capture market share",
            "Implement automated decision rules for Prime applications",
            "Launch targeted marketing campaign for debt consolidation loans",
            "Enhance fraud detection for online applications"
        ]

        for i, rec in enumerate(short_term, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

    with rec_col2:
        st.markdown("**🚀 Long-term Strategy (6-12 months)**")
        long_term = [
            "Develop AI-powered risk assessment for emerging credit segments",
            "Expand into adjacent financial products (insurance, investments)",
            "Build strategic partnerships with fintech companies",
            "Implement ESG scoring for sustainable lending practices"
        ]

        for i, rec in enumerate(long_term, 1):
            st.markdown(f"""
            <div class="recommendation-card">
                <strong>{i}.</strong> {rec}
            </div>
            """, unsafe_allow_html=True)

    # Export Business Intelligence Report
    st.markdown("---")
    st.markdown("### 📤 Export Business Intelligence Report")

    if st.button("📊 Generate BI Report", type="primary"):
        bi_report = f"""
Business Intelligence Report
===========================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Executive Summary:
- Portfolio performing above target with {current_kpis['approval_rate']:.1%} approval rate
- Profit margin of {current_kpis['profit_margin']:.1%} exceeds target by {(current_kpis['profit_margin'] - bi_engine.kpi_thresholds['profit_margin']['target']):.1%}
- Portfolio growth of {current_kpis['portfolio_growth']:.1%} indicates strong market position

Key Performance Indicators:
{chr(10).join([f"- {k.replace('_', ' ').title()}: {v:.1%}" for k, v in current_kpis.items()])}

Portfolio Analysis:
{portfolio_data.to_string(index=False)}

Strategic Recommendations:
- Focus on Near-Prime segment expansion
- Optimize operational efficiency in Prime segment
- Monitor Subprime portfolio closely for risk management
- Leverage market trends for competitive advantage

Risk Factors:
- Monitor default rate trends in Subprime segment
- Watch for interest rate impact on demand
- Ensure adequate capital reserves for growth

Next Review: {(datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')}
        """

        st.download_button(
            "Download BI Report",
            bi_report,
            f"business_intelligence_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            "text/plain"
        )

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"💼 Business Intelligence Engine | Real-time Analytics | "
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
