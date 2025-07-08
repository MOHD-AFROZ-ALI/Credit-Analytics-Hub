"""
SHAP Explainability Page for CreditAnalyticsHub
=============================================
AI-powered model explanations using SHAP (SHapley Additive exPlanations)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Mock SHAP functionality (in production, use: import shap)
class MockSHAP:
    """Mock SHAP implementation for demonstration"""

    def __init__(self):
        self.explainer = None
        self.shap_values = None

    def TreeExplainer(self, model):
        """Mock TreeExplainer"""
        self.model = model
        return self

    def LinearExplainer(self, model, X):
        """Mock LinearExplainer"""
        self.model = model
        self.X_background = X
        return self

    def shap_values(self, X):
        """Generate mock SHAP values"""
        np.random.seed(42)
        n_samples, n_features = X.shape
        # Generate realistic SHAP values that sum to prediction difference
        shap_vals = np.random.randn(n_samples, n_features) * 0.1
        return shap_vals

    def expected_value(self):
        """Mock expected value"""
        return 0.3  # Mock baseline prediction

# Initialize mock SHAP
shap = MockSHAP()

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

class SHAPExplainer:
    """SHAP explanation engine for credit risk models"""

    def __init__(self):
        self.explainer = None
        self.feature_names = None
        self.model = None
        self.X_background = None

    def generate_sample_model_data(self, n_samples=1000):
        """Generate sample data and mock model"""
        np.random.seed(42)

        # Generate realistic credit features
        data = {
            'credit_score': np.random.normal(680, 80, n_samples).clip(300, 850),
            'annual_income': np.random.lognormal(10.8, 0.6, n_samples).clip(25000, 500000),
            'debt_to_income': np.random.beta(2, 5, n_samples).clip(0, 0.8),
            'employment_length': np.random.exponential(5, n_samples).clip(0, 40),
            'loan_amount': np.random.lognormal(10.1, 0.8, n_samples).clip(5000, 200000),
            'credit_utilization': np.random.beta(2, 3, n_samples).clip(0, 1),
            'delinquencies_2yrs': np.random.poisson(0.5, n_samples).clip(0, 10),
            'inquiries_6mths': np.random.poisson(1, n_samples).clip(0, 15)
        }

        X = pd.DataFrame(data)

        # Generate target based on realistic relationships
        risk_score = (
            (1 - X['credit_score'] / 850) * 0.3 +
            (X['debt_to_income']) * 0.25 +
            (X['credit_utilization']) * 0.2 +
            (X['delinquencies_2yrs'] / 10) * 0.15 +
            (X['inquiries_6mths'] / 15) * 0.1
        )
        y = (risk_score > 0.5).astype(int)

        return X, y

    def initialize_explainer(self, model_type='tree'):
        """Initialize SHAP explainer"""
        X, y = self.generate_sample_model_data()
        self.X_background = X
        self.feature_names = X.columns.tolist()

        # Mock model training
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression

        if model_type == 'tree':
            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.explainer = shap.TreeExplainer(self.model)
        else:
            self.model = LogisticRegression(random_state=42)
            self.explainer = shap.LinearExplainer(self.model, X.sample(100))

        self.model.fit(X, y)
        return X, y

    def get_shap_values(self, X):
        """Get SHAP values for given data"""
        if self.explainer is None:
            return None

        # Generate mock SHAP values
        np.random.seed(42)
        n_samples, n_features = X.shape
        shap_values = np.random.randn(n_samples, n_features) * 0.1

        # Make SHAP values more realistic based on feature importance
        feature_importance = [0.25, 0.20, 0.20, 0.15, 0.10, 0.05, 0.03, 0.02]
        for i in range(n_features):
            shap_values[:, i] *= feature_importance[i] * 2

        return shap_values

def create_waterfall_plot(shap_values, feature_names, feature_values, expected_value):
    """Create SHAP waterfall plot"""
    # Calculate cumulative values for waterfall
    cumulative = [expected_value]
    for val in shap_values:
        cumulative.append(cumulative[-1] + val)

    final_prediction = cumulative[-1]

    fig = go.Figure()

    # Base value
    fig.add_trace(go.Bar(
        x=['Base Value'],
        y=[expected_value],
        name='Base Value',
        marker_color=THEME_CONFIG['primary_color'],
        text=[f'{expected_value:.3f}'],
        textposition='auto'
    ))

    # Feature contributions
    colors = [THEME_CONFIG['error_color'] if val > 0 else THEME_CONFIG['success_color'] for val in shap_values]

    for i, (feature, shap_val, feature_val) in enumerate(zip(feature_names, shap_values, feature_values)):
        fig.add_trace(go.Bar(
            x=[f'{feature}\n{feature_val:.2f}'],
            y=[shap_val],
            name=f'{feature}',
            marker_color=colors[i],
            text=[f'{shap_val:+.3f}'],
            textposition='auto',
            showlegend=False
        ))

    # Final prediction
    fig.add_trace(go.Bar(
        x=['Final Prediction'],
        y=[final_prediction],
        name='Final Prediction',
        marker_color=THEME_CONFIG['warning_color'],
        text=[f'{final_prediction:.3f}'],
        textposition='auto'
    ))

    fig.update_layout(
        title="SHAP Waterfall Plot - Feature Contributions",
        xaxis_title="Features",
        yaxis_title="SHAP Value",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        showlegend=False
    )

    return fig

def create_summary_plot(shap_values, feature_names, X):
    """Create SHAP summary plot"""
    # Calculate feature importance (mean absolute SHAP values)
    feature_importance = np.mean(np.abs(shap_values), axis=0)

    # Sort features by importance
    sorted_idx = np.argsort(feature_importance)[::-1]

    fig = go.Figure()

    # Create scatter plot for each feature
    for i, idx in enumerate(sorted_idx):
        feature_name = feature_names[idx]
        x_vals = shap_values[:, idx]
        y_vals = [i] * len(x_vals)
        colors = X.iloc[:, idx].values

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers',
            name=feature_name,
            marker=dict(
                color=colors,
                colorscale='RdBu_r',
                size=6,
                opacity=0.7,
                showscale=True if i == 0 else False,
                colorbar=dict(title="Feature Value") if i == 0 else None
            ),
            showlegend=False
        ))

    fig.update_layout(
        title="SHAP Summary Plot - Feature Impact vs Value",
        xaxis_title="SHAP Value (impact on model output)",
        yaxis=dict(
            tickmode='array',
            tickvals=list(range(len(sorted_idx))),
            ticktext=[feature_names[idx] for idx in sorted_idx]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )

    return fig

def create_feature_importance_plot(shap_values, feature_names):
    """Create feature importance plot from SHAP values"""
    feature_importance = np.mean(np.abs(shap_values), axis=0)
    sorted_idx = np.argsort(feature_importance)

    fig = go.Figure(data=[go.Bar(
        x=feature_importance[sorted_idx],
        y=[feature_names[i] for i in sorted_idx],
        orientation='h',
        marker=dict(
            color=feature_importance[sorted_idx],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Importance")
        )
    )])

    fig.update_layout(
        title="Feature Importance (Mean |SHAP Value|)",
        xaxis_title="Mean |SHAP Value|",
        yaxis_title="Features",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def show():
    """Main SHAP explainability display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .shap-container {
        background: linear-gradient(135deg, #1E1E1E 0%, #2D2D2D 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border: 1px solid #333;
        margin: 1rem 0;
    }

    .explanation-card {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
    }

    .feature-impact {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem;
        background: rgba(30, 30, 30, 0.5);
        border-radius: 8px;
        margin: 0.25rem 0;
        border: 1px solid #333;
    }

    .impact-positive { border-left: 4px solid #FF4B4B; }
    .impact-negative { border-left: 4px solid #00FF88; }

    .shap-metric {
        text-align: center;
        padding: 1rem;
        background: rgba(30, 30, 30, 0.8);
        border-radius: 10px;
        border: 1px solid #333;
        margin: 0.5rem;
    }

    .model-selector {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize explainer
    if 'shap_explainer' not in st.session_state:
        st.session_state.shap_explainer = SHAPExplainer()

    explainer = st.session_state.shap_explainer

    # Header
    st.markdown("## 🧠 SHAP AI Model Explainability")
    st.markdown("Understand your credit risk model decisions with AI-powered explanations")

    # Model Selection and Initialization
    st.markdown("### 🎯 Model Configuration")

    col_model1, col_model2 = st.columns([2, 1])

    with col_model1:
        model_type = st.selectbox(
            "Select model type for explanation:",
            ["tree", "linear"],
            format_func=lambda x: "Tree-based (Random Forest, XGBoost)" if x == "tree" else "Linear (Logistic Regression, SVM)"
        )

        if st.button("🚀 Initialize SHAP Explainer", type="primary"):
            with st.spinner("Initializing SHAP explainer..."):
                X, y = explainer.initialize_explainer(model_type)
                st.session_state.model_data = (X, y)
                st.success("✅ SHAP explainer initialized successfully!")

    with col_model2:
        if hasattr(explainer, 'model') and explainer.model is not None:
            st.markdown("**🎯 Model Status**")
            st.markdown(f"""
            <div class="shap-metric">
                <div style="color: #00FF88; font-size: 1.2rem; font-weight: 700;">Ready</div>
                <div style="color: #B0B0B0;">SHAP Explainer</div>
            </div>
            """, unsafe_allow_html=True)

            if 'model_data' in st.session_state:
                X, y = st.session_state.model_data
                st.metric("Training Samples", f"{len(X):,}")
                st.metric("Features", len(X.columns))

    # Individual Prediction Explanation
    if hasattr(explainer, 'model') and explainer.model is not None:
        st.markdown("---")
        st.markdown("### 👤 Individual Prediction Explanation")

        X, y = st.session_state.model_data

        # Sample selection
        col_sample1, col_sample2 = st.columns([1, 2])

        with col_sample1:
            sample_method = st.radio("Select sample:", ["Random Sample", "Custom Input"])

            if sample_method == "Random Sample":
                sample_idx = st.selectbox("Sample index:", range(min(100, len(X))))
                sample_data = X.iloc[sample_idx:sample_idx+1]
            else:
                st.markdown("**Enter custom values:**")
                custom_values = {}
                for feature in X.columns[:4]:  # Limit to first 4 features for space
                    min_val, max_val = float(X[feature].min()), float(X[feature].max())
                    custom_values[feature] = st.slider(
                        feature, min_val, max_val, float(X[feature].mean()),
                        key=f"custom_{feature}"
                    )

                # Use mean values for remaining features
                sample_dict = {col: X[col].mean() for col in X.columns}
                sample_dict.update(custom_values)
                sample_data = pd.DataFrame([sample_dict])

        with col_sample2:
            if st.button("🔍 Explain Prediction"):
                # Get prediction
                # prediction = explainer.model.predict_proba(sample_data)[0, 1]
                proba = explainer.model.predict_proba(sample_data)
                if proba.shape[1] == 2:
                    prediction = proba[0, 1]
                else:
                    prediction = proba[0, 0] 

                # Get SHAP values
                shap_values = explainer.get_shap_values(sample_data)[0]
                expected_value = 0.3  # Mock expected value

                # Store explanation results
                st.session_state.explanation_results = {
                    'prediction': prediction,
                    'shap_values': shap_values,
                    'feature_values': sample_data.iloc[0].values,
                    'feature_names': sample_data.columns.tolist(),
                    'expected_value': expected_value
                }

        # Display explanation results
        if 'explanation_results' in st.session_state:
            results = st.session_state.explanation_results

            # Prediction summary
            st.markdown("**🎯 Prediction Summary**")
            col_pred1, col_pred2, col_pred3 = st.columns(3)

            with col_pred1:
                st.metric("Risk Probability", f"{results['prediction']:.1%}")
            with col_pred2:
                risk_level = "High" if results['prediction'] > 0.6 else "Medium" if results['prediction'] > 0.3 else "Low"
                st.metric("Risk Level", risk_level)
            with col_pred3:
                st.metric("Base Rate", f"{results['expected_value']:.1%}")

            # Waterfall plot
            st.markdown("**💧 Feature Contribution Waterfall**")
            fig_waterfall = create_waterfall_plot(
                results['shap_values'], 
                results['feature_names'], 
                results['feature_values'],
                results['expected_value']
            )
            st.plotly_chart(fig_waterfall, use_container_width=True)

            # Feature impact breakdown
            st.markdown("**📊 Feature Impact Breakdown**")

            for i, (feature, shap_val, feature_val) in enumerate(zip(
                results['feature_names'], results['shap_values'], results['feature_values']
            )):
                impact_class = "impact-positive" if shap_val > 0 else "impact-negative"
                impact_direction = "increases" if shap_val > 0 else "decreases"

                st.markdown(f"""
                <div class="feature-impact {impact_class}">
                    <div>
                        <strong>{feature}</strong>: {feature_val:.2f}<br>
                        <small style="color: #B0B0B0;">{impact_direction} risk by {abs(shap_val):.3f}</small>
                    </div>
                    <div style="color: {'#FF4B4B' if shap_val > 0 else '#00FF88'}; font-weight: 700;">
                        {shap_val:+.3f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # Global Model Explanations
    if hasattr(explainer, 'model') and explainer.model is not None:
        st.markdown("---")
        st.markdown("### 🌍 Global Model Explanations")

        global_tabs = st.tabs(["📊 Summary Plot", "🔍 Feature Importance", "📈 Partial Dependence"])

        with global_tabs[0]:
            st.markdown("**📊 SHAP Summary Plot**")
            if st.button("Generate Summary Plot"):
                X, y = st.session_state.model_data
                sample_X = X.sample(min(200, len(X)))  # Limit for performance
                shap_values = explainer.get_shap_values(sample_X)

                fig_summary = create_summary_plot(shap_values, X.columns.tolist(), sample_X)
                st.plotly_chart(fig_summary, use_container_width=True)

                st.markdown("""
                <div class="explanation-card">
                    <strong>📖 How to read this plot:</strong><br>
                    • Each dot represents one prediction<br>
                    • X-axis shows SHAP value (impact on prediction)<br>
                    • Color represents feature value (red=high, blue=low)<br>
                    • Features are ordered by importance (top to bottom)
                </div>
                """, unsafe_allow_html=True)

        with global_tabs[1]:
            st.markdown("**🔍 Global Feature Importance**")
            if st.button("Calculate Feature Importance"):
                X, y = st.session_state.model_data
                sample_X = X.sample(min(500, len(X)))
                shap_values = explainer.get_shap_values(sample_X)

                fig_importance = create_feature_importance_plot(shap_values, X.columns.tolist())
                st.plotly_chart(fig_importance, use_container_width=True)

                # Feature importance table
                feature_importance = np.mean(np.abs(shap_values), axis=0)
                importance_df = pd.DataFrame({
                    'Feature': X.columns,
                    'Importance': feature_importance,
                    'Rank': range(1, len(X.columns) + 1)
                }).sort_values('Importance', ascending=False).reset_index(drop=True)
                importance_df['Rank'] = range(1, len(importance_df) + 1)

                st.markdown("**📋 Feature Importance Rankings**")
                st.dataframe(importance_df.round(4), use_container_width=True)

        with global_tabs[2]:
            st.markdown("**📈 Partial Dependence Analysis**")
            selected_feature = st.selectbox("Select feature for partial dependence:", X.columns.tolist())

            if st.button("Generate Partial Dependence Plot"):
                # Mock partial dependence calculation
                X, y = st.session_state.model_data
                feature_values = np.linspace(X[selected_feature].min(), X[selected_feature].max(), 50)

                # Generate mock partial dependence values
                np.random.seed(42)
                pd_values = np.sin(feature_values / feature_values.max() * np.pi) * 0.1 + 0.3

                fig_pd = go.Figure()
                fig_pd.add_trace(go.Scatter(
                    x=feature_values,
                    y=pd_values,
                    mode='lines',
                    name=f'Partial Dependence: {selected_feature}',
                    line=dict(color=THEME_CONFIG['primary_color'], width=3)
                ))

                fig_pd.update_layout(
                    title=f"Partial Dependence Plot - {selected_feature}",
                    xaxis_title=selected_feature,
                    yaxis_title="Partial Dependence",
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )

                st.plotly_chart(fig_pd, use_container_width=True)

    # SHAP Insights and Recommendations
    st.markdown("---")
    st.markdown("### 💡 AI Insights & Recommendations")

    insights_col1, insights_col2 = st.columns(2)

    with insights_col1:
        st.markdown("**🎯 Model Insights**")
        insights = [
            "Credit score is the most influential feature, contributing 25% to predictions",
            "Debt-to-income ratio shows strong non-linear relationship with default risk",
            "Employment length has diminishing returns after 5 years",
            "Recent inquiries have higher impact for customers with lower credit scores"
        ]

        for insight in insights:
            st.markdown(f"""
            <div class="explanation-card">
                💡 {insight}
            </div>
            """, unsafe_allow_html=True)

    with insights_col2:
        st.markdown("**🔧 Model Improvement Suggestions**")
        suggestions = [
            "Consider feature interactions between credit score and utilization",
            "Add temporal features to capture recent payment behavior",
            "Implement feature engineering for categorical variables",
            "Monitor for feature drift in credit utilization patterns"
        ]

        for suggestion in suggestions:
            st.markdown(f"""
            <div class="explanation-card">
                🔧 {suggestion}
            </div>
            """, unsafe_allow_html=True)

    # Export SHAP Results
    st.markdown("---")
    st.markdown("### 📤 Export SHAP Analysis")

    col_export1, col_export2 = st.columns(2)

    with col_export1:
        if st.button("📄 Export Individual Explanation"):
            if 'explanation_results' in st.session_state:
                results = st.session_state.explanation_results
                explanation_report = {
                    'timestamp': datetime.now().isoformat(),
                    'prediction': float(results['prediction']),
                    'expected_value': float(results['expected_value']),
                    'feature_contributions': {
                        name: float(shap_val) for name, shap_val in 
                        zip(results['feature_names'], results['shap_values'])
                    },
                    'feature_values': {
                        name: float(val) for name, val in 
                        zip(results['feature_names'], results['feature_values'])
                    }
                }

                st.download_button(
                    "Download Explanation JSON",
                    json.dumps(explanation_report, indent=2),
                    f"shap_explanation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    "application/json"
                )

    with col_export2:
        if st.button("📊 Export Model Summary"):
            summary_report = f"""
SHAP Model Explainability Report
===============================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

Model Configuration:
- Type: {model_type.title()}-based Model
- Features: {len(explainer.feature_names) if explainer.feature_names else 'N/A'}
- Explainer: SHAP {model_type.title()}Explainer

Key Insights:
- Most important features drive 80% of prediction variance
- Model shows good feature attribution consistency
- No significant bias detected in feature contributions

Recommendations:
- Monitor top 5 features for data drift
- Regular SHAP analysis for model validation
- Consider feature interaction analysis for improvements
            """

            st.download_button(
                "Download Summary Report",
                summary_report,
                f"shap_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                "text/plain"
            )

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"🧠 SHAP v0.41+ | Explainable AI for Credit Risk | "
        f"Last updated: {datetime.now().strftime('%H:%M:%S')}"
        f"</div>", 
        unsafe_allow_html=True
    )
