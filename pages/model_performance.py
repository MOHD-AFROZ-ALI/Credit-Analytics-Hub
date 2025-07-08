"""
Model Performance Page for CreditAnalyticsHub
===========================================
Comprehensive model evaluation, comparison, and training interface
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from typing import Dict, List, Tuple, Any
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import joblib

# Import configuration
try:
    from config import get_config
    THEME_CONFIG = get_config('theme')
    MODEL_CONFIG = get_config('model')
except ImportError:
    THEME_CONFIG = {
        'primary_color': '#00D4FF',
        'success_color': '#00FF88',
        'warning_color': '#FFB800',
        'error_color': '#FF4B4B'
    }
    MODEL_CONFIG = {
        'available_models': {
            'random_forest': {'name': 'Random Forest', 'type': 'ensemble'},
            'logistic_regression': {'name': 'Logistic Regression', 'type': 'linear'},
            'svm': {'name': 'Support Vector Machine', 'type': 'kernel'}
        }
    }

class ModelTrainer:
    """Model training and evaluation engine"""

    def __init__(self):
        self.models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'SVM': SVC(probability=True, random_state=42)
        }
        self.trained_models = {}
        self.performance_history = []

    def generate_sample_data(self, n_samples=1000):
        """Generate sample training data"""
        np.random.seed(42)
        X = np.random.randn(n_samples, 8)
        # Create realistic feature relationships
        y = ((X[:, 0] * 0.3 + X[:, 1] * 0.2 - X[:, 2] * 0.25 + 
              X[:, 3] * 0.15 + np.random.randn(n_samples) * 0.1) > 0).astype(int)

        feature_names = ['Credit Score', 'Income', 'Debt Ratio', 'Employment Length',
                        'Loan Amount', 'Credit Utilization', 'Delinquencies', 'Inquiries']

        return pd.DataFrame(X, columns=feature_names), y

    def train_model(self, model_name, X, y):
        """Train a specific model"""
        model = self.models[model_name].fit(X, y)
        self.trained_models[model_name] = model
        return model

    def evaluate_model(self, model_name, X, y):
        """Evaluate model performance"""
        if model_name not in self.trained_models:
            return None

        model = self.trained_models[model_name]
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1] if hasattr(model, 'predict_proba') else y_pred

        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred),
            'roc_auc': roc_auc_score(y, y_prob)
        }

        return metrics, y_pred, y_prob

def create_performance_comparison_chart(performance_data: Dict) -> go.Figure:
    """Create model performance comparison chart"""
    models = list(performance_data.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

    fig = go.Figure()

    for metric in metrics:
        values = [performance_data[model].get(metric, 0) for model in models]
        fig.add_trace(go.Scatter(
            x=models,
            y=values,
            mode='lines+markers',
            name=metric.replace('_', ' ').title(),
            line=dict(width=3),
            marker=dict(size=8)
        ))

    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Models",
        yaxis_title="Score",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig

def create_confusion_matrix_heatmap(cm: np.ndarray, model_name: str) -> go.Figure:
    """Create confusion matrix heatmap"""
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='Blues',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16},
        hoverongaps=False
    ))

    fig.update_layout(
        title=f"Confusion Matrix - {model_name}",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        height=400
    )

    return fig

def create_roc_curve(y_true, y_prob, model_name: str) -> go.Figure:
    """Create ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_score = roc_auc_score(y_true, y_prob)

    fig = go.Figure()

    # ROC curve
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'{model_name} (AUC = {auc_score:.3f})',
        line=dict(color=THEME_CONFIG['primary_color'], width=3)
    ))

    # Diagonal line
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='gray', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )

    return fig

def create_feature_importance_chart(model, feature_names: List[str]) -> go.Figure:
    """Create feature importance chart"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_[0])
    else:
        return None

    # Sort by importance
    indices = np.argsort(importances)[::-1]

    fig = go.Figure(data=[go.Bar(
        x=[feature_names[i] for i in indices],
        y=[importances[i] for i in indices],
        marker=dict(
            color=importances[indices],
            colorscale='Viridis',
            showscale=True
        )
    )])

    fig.update_layout(
        title="Feature Importance",
        xaxis_title="Features",
        yaxis_title="Importance",
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
        xaxis_tickangle=-45
    )

    return fig

def show():
    """Main model performance display function"""
    # Custom CSS
    st.markdown("""
    <style>
    .performance-container {
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
        margin: 0.5rem;
    }

    .model-status {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
        display: inline-block;
    }

    .status-trained { background: #00FF88; color: #000; }
    .status-training { background: #FFB800; color: #000; }
    .status-untrained { background: #FF4B4B; color: #FFF; }

    .training-progress {
        background: rgba(0, 212, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 212, 255, 0.2);
        margin: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

    # Initialize trainer
    if 'trainer' not in st.session_state:
        st.session_state.trainer = ModelTrainer()

    trainer = st.session_state.trainer

    # Header
    st.markdown("## 📈 Model Performance & Training")
    st.markdown("Comprehensive model evaluation, comparison, and training interface")

    # Model Training Section
    st.markdown("### 🎯 Model Training")

    col_train1, col_train2 = st.columns([2, 1])

    with col_train1:
        # Generate or load data
        if st.button("🔄 Generate Sample Data", type="secondary"):
            X, y = trainer.generate_sample_data()
            st.session_state.training_data = (X, y)
            st.success("✅ Sample data generated!")

        # Model selection and training
        if 'training_data' in st.session_state:
            X, y = st.session_state.training_data

            selected_models = st.multiselect(
                "Select models to train:",
                list(trainer.models.keys()),
                default=list(trainer.models.keys())
            )

            if st.button("🚀 Train Selected Models", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i, model_name in enumerate(selected_models):
                    status_text.text(f"Training {model_name}...")
                    trainer.train_model(model_name, X, y)
                    progress_bar.progress((i + 1) / len(selected_models))

                status_text.text("✅ Training completed!")
                st.success(f"Successfully trained {len(selected_models)} models!")

    with col_train2:
        # Model status
        st.markdown("**🎯 Model Status**")
        for model_name in trainer.models.keys():
            status = "trained" if model_name in trainer.trained_models else "untrained"
            status_class = f"status-{status}"
            st.markdown(f"""
            <div class="model-status {status_class}">
                {model_name}: {status.title()}
            </div>
            """, unsafe_allow_html=True)

        # Data info
        if 'training_data' in st.session_state:
            X, y = st.session_state.training_data
            st.markdown("**📊 Training Data**")
            st.metric("Samples", f"{len(X):,}")
            st.metric("Features", len(X.columns))
            st.metric("Positive Rate", f"{y.mean():.1%}")

    # Performance Evaluation
    if trainer.trained_models:
        st.markdown("---")
        st.markdown("### 📊 Performance Evaluation")

        # Evaluate all trained models
        performance_data = {}
        confusion_matrices = {}
        roc_data = {}

        if 'training_data' in st.session_state:
            X, y = st.session_state.training_data

            for model_name in trainer.trained_models.keys():
                metrics, y_pred, y_prob = trainer.evaluate_model(model_name, X, y)
                performance_data[model_name] = metrics
                confusion_matrices[model_name] = confusion_matrix(y, y_pred)
                roc_data[model_name] = (y, y_prob)

        # Performance comparison
        if performance_data:
            fig_comparison = create_performance_comparison_chart(performance_data)
            st.plotly_chart(fig_comparison, use_container_width=True)

            # Metrics table
            st.markdown("**📋 Detailed Metrics**")
            metrics_df = pd.DataFrame(performance_data).T
            st.dataframe(metrics_df.round(4), use_container_width=True)

    # Model Analysis Tabs
    if trainer.trained_models:
        st.markdown("---")
        st.markdown("### 🔍 Detailed Analysis")

        analysis_tabs = st.tabs(["🎯 Confusion Matrix", "📈 ROC Curves", "🔍 Feature Importance", "📊 Cross-Validation"])

        with analysis_tabs[0]:
            # Confusion Matrix
            selected_model_cm = st.selectbox("Select model for confusion matrix:", list(trainer.trained_models.keys()))

            if selected_model_cm and selected_model_cm in confusion_matrices:
                fig_cm = create_confusion_matrix_heatmap(confusion_matrices[selected_model_cm], selected_model_cm)
                st.plotly_chart(fig_cm, use_container_width=True)

                # Confusion matrix metrics
                cm = confusion_matrices[selected_model_cm]
                tn, fp, fn, tp = cm.ravel()

                col_cm1, col_cm2, col_cm3, col_cm4 = st.columns(4)
                with col_cm1:
                    st.metric("True Negatives", tn)
                with col_cm2:
                    st.metric("False Positives", fp)
                with col_cm3:
                    st.metric("False Negatives", fn)
                with col_cm4:
                    st.metric("True Positives", tp)

        with analysis_tabs[1]:
            # ROC Curves
            st.markdown("**📈 ROC Curve Analysis**")

            fig_roc = go.Figure()

            # Add diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                name='Random Classifier',
                line=dict(color='gray', width=2, dash='dash')
            ))

            # Add ROC curves for all models
            colors = ['#00D4FF', '#00FF88', '#FFB800', '#FF4B4B', '#9D4EDD']
            for i, (model_name, (y_true, y_prob)) in enumerate(roc_data.items()):
                fpr, tpr, _ = roc_curve(y_true, y_prob)
                auc_score = roc_auc_score(y_true, y_prob)

                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {auc_score:.3f})',
                    line=dict(color=colors[i % len(colors)], width=3)
                ))

            fig_roc.update_layout(
                title="ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white')
            )

            st.plotly_chart(fig_roc, use_container_width=True)

        with analysis_tabs[2]:
            # Feature Importance
            selected_model_fi = st.selectbox("Select model for feature importance:", list(trainer.trained_models.keys()))

            if selected_model_fi and 'training_data' in st.session_state:
                X, y = st.session_state.training_data
                model = trainer.trained_models[selected_model_fi]

                fig_importance = create_feature_importance_chart(model, X.columns.tolist())
                if fig_importance:
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info("Feature importance not available for this model type")

        with analysis_tabs[3]:
            # Cross-Validation
            st.markdown("**📊 Cross-Validation Results**")

            if st.button("🔄 Run Cross-Validation"):
                if 'training_data' in st.session_state:
                    X, y = st.session_state.training_data
                    cv_results = {}

                    progress_bar = st.progress(0)

                    for i, (model_name, model) in enumerate(trainer.models.items()):
                        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
                        cv_results[model_name] = {
                            'mean': scores.mean(),
                            'std': scores.std(),
                            'scores': scores.tolist()
                        }
                        progress_bar.progress((i + 1) / len(trainer.models))

                    # Display results
                    cv_df = pd.DataFrame({
                        'Model': list(cv_results.keys()),
                        'Mean AUC': [cv_results[m]['mean'] for m in cv_results.keys()],
                        'Std AUC': [cv_results[m]['std'] for m in cv_results.keys()]
                    })

                    st.dataframe(cv_df.round(4), use_container_width=True)

                    # CV scores visualization
                    fig_cv = go.Figure()

                    for model_name, results in cv_results.items():
                        fig_cv.add_trace(go.Box(
                            y=results['scores'],
                            name=model_name,
                            boxpoints='all'
                        ))

                    fig_cv.update_layout(
                        title="Cross-Validation Scores Distribution",
                        yaxis_title="AUC Score",
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white')
                    )

                    st.plotly_chart(fig_cv, use_container_width=True)

    # Model Versioning
    st.markdown("---")
    st.markdown("### 📦 Model Management")

    col_version1, col_version2 = st.columns(2)

    with col_version1:
        st.markdown("**💾 Save Models**")
        if trainer.trained_models:
            model_to_save = st.selectbox("Select model to save:", list(trainer.trained_models.keys()))
            version_name = st.text_input("Version name:", f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}")

            if st.button("💾 Save Model"):
                # In a real application, you would save to a proper model registry
                model_info = {
                    'model_name': model_to_save,
                    'version': version_name,
                    'timestamp': datetime.now().isoformat(),
                    'performance': performance_data.get(model_to_save, {})
                }

                st.success(f"✅ Model {model_to_save} saved as {version_name}")
                st.json(model_info)

    with col_version2:
        st.markdown("**📊 Model Registry**")
        # Mock model registry
        registry_data = [
            {'Model': 'Random Forest', 'Version': 'v20240115_143022', 'AUC': 0.847, 'Status': 'Production'},
            {'Model': 'Logistic Regression', 'Version': 'v20240114_091533', 'AUC': 0.823, 'Status': 'Staging'},
            {'Model': 'SVM', 'Version': 'v20240113_165412', 'AUC': 0.801, 'Status': 'Archived'}
        ]

        st.dataframe(pd.DataFrame(registry_data), use_container_width=True)

    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #B0B0B0; font-size: 0.8rem;'>"
        f"🎯 {len(trainer.trained_models)} models trained | "
        f"Last updated: {datetime.now().strftime('%H:%M:%S')} | "
        f"Training engine: scikit-learn"
        f"</div>", 
        unsafe_allow_html=True
    )
