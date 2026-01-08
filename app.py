import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Wine Quality Prediction",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #8B0000;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and related files
@st.cache_resource
def load_model():
    """Load the trained model and related files"""
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('model_metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
        return model, feature_names, metadata
    except FileNotFoundError as e:
        st.error(f"Model files not found. Please ensure model.pkl, feature_names.pkl, and model_metadata.pkl exist in the root directory.")
        st.stop()

@st.cache_data
def load_data():
    """Load the dataset"""
    try:
        df = pd.read_csv('data/winequality.csv')
        return df
    except FileNotFoundError:
        st.warning("Dataset not found. Some features may not work. Please ensure data/winequality.csv exists.")
        return None

@st.cache_data
def load_model_comparison():
    """Load model comparison results"""
    try:
        with open('model_comparison.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

@st.cache_data
def load_confusion_matrix():
    """Load confusion matrix"""
    try:
        with open('confusion_matrix.pkl', 'rb') as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None

# Load resources
model, feature_names, metadata = load_model()
df = load_data()
model_comparison = load_model_comparison()
confusion_matrix_data = load_confusion_matrix()

# Sidebar Navigation
st.sidebar.title("🍷 Navigation")
page = st.sidebar.selectbox(
    "Choose a page",
    ["Home", "Data Exploration", "Visualizations", "Model Prediction", "Model Performance"]
)

# Home Page
if page == "Home":
    st.markdown('<h1 class="main-header">🍷 Wine Quality Prediction</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Machine Learning Model Deployment with Streamlit</p>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", metadata.get('model_type', 'Classification').title())
    
    with col2:
        st.metric("Best Model", metadata.get('best_model', 'Random Forest'))
    
    with col3:
        st.metric("Accuracy", f"{metadata.get('accuracy', 0):.2%}")
    
    st.markdown("---")
    
    st.markdown("""
    ## 📋 Project Overview
    
    This application demonstrates a complete machine learning pipeline for predicting wine quality:
    
    ### 🎯 Objectives
    - **Data Analysis**: Explore and understand the wine quality dataset
    - **Model Training**: Train and compare multiple ML algorithms
    - **Interactive Prediction**: Make real-time quality predictions
    - **Performance Analysis**: Visualize model metrics and comparisons
    
    ### 📊 Dataset
    The Wine Quality dataset contains various physicochemical properties of wines and their quality ratings.
    The model predicts whether a wine is of good quality (quality score ≥ 7) or not.
    
    ### 🔧 Features
    - **Data Exploration**: Interactive dataset overview and filtering
    - **Visualizations**: Multiple charts and plots for data analysis
    - **Real-time Prediction**: Input wine characteristics and get instant predictions
    - **Performance Metrics**: View model evaluation results and comparisons
    
    ### 🚀 How to Use
    1. Navigate through different sections using the sidebar
    2. Explore the dataset in the **Data Exploration** section
    3. View visualizations in the **Visualizations** section
    4. Make predictions in the **Model Prediction** section
    5. Check model performance in the **Model Performance** section
    """)
    
    if df is not None:
        st.markdown("### 📈 Quick Dataset Stats")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Samples", len(df))
        with col2:
            st.metric("Features", len(df.columns) - 1)
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        with col4:
            st.metric("Data Types", len(df.dtypes.unique()))

# Data Exploration Page
elif page == "Data Exploration":
    st.title("📊 Data Exploration")
    
    if df is None:
        st.error("Dataset not available. Please ensure data/winequality.csv exists.")
    else:
        st.markdown("### Dataset Overview")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Dataset Shape", f"{df.shape[0]} rows × {df.shape[1]} columns")
        with col2:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.2f} KB")
        
        st.markdown("---")
        
        # Dataset Info
        st.markdown("### Dataset Information")
        with st.expander("View Dataset Info"):
            st.text("Data Types and Non-Null Counts:")
            st.dataframe(pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes,
                'Non-Null Count': df.count(),
                'Null Count': df.isnull().sum()
            }))
        
        # Sample Data
        st.markdown("### Sample Data")
        num_rows = st.slider("Number of rows to display", 5, 100, 10)
        st.dataframe(df.head(num_rows), use_container_width=True)
        
        # Statistical Summary
        st.markdown("### Statistical Summary")
        with st.expander("View Statistical Summary"):
            st.dataframe(df.describe(), use_container_width=True)
        
        # Missing Values
        st.markdown("### Missing Values Analysis")
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.bar_chart(missing_data[missing_data > 0])
            st.dataframe(pd.DataFrame({
                'Column': missing_data[missing_data > 0].index,
                'Missing Count': missing_data[missing_data > 0].values,
                'Percentage': (missing_data[missing_data > 0] / len(df) * 100).values
            }))
        else:
            st.success("✅ No missing values in the dataset!")
        
        # Data Filtering
        st.markdown("### Interactive Data Filtering")
        st.markdown("Filter the dataset based on feature values:")
        
        filter_cols = st.columns(min(3, len(df.columns)))
        filters = {}
        
        for idx, col in enumerate(df.select_dtypes(include=[np.number]).columns):
            with filter_cols[idx % len(filter_cols)]:
                min_val = float(df[col].min())
                max_val = float(df[col].max())
                filters[col] = st.slider(
                    col,
                    min_value=min_val,
                    max_value=max_val,
                    value=(min_val, max_val),
                    key=f"filter_{col}"
                )
        
        # Apply filters
        if st.button("Apply Filters"):
            filtered_df = df.copy()
            for col, (min_val, max_val) in filters.items():
                filtered_df = filtered_df[(filtered_df[col] >= min_val) & (filtered_df[col] <= max_val)]
            
            st.markdown(f"**Filtered Dataset: {len(filtered_df)} rows**")
            st.dataframe(filtered_df, use_container_width=True)
            
            if len(filtered_df) > 0:
                st.download_button(
                    label="Download Filtered Data",
                    data=filtered_df.to_csv(index=False),
                    file_name="filtered_wine_data.csv",
                    mime="text/csv"
                )

# Visualizations Page
elif page == "Visualizations":
    st.title("📈 Visualizations")
    
    if df is None:
        st.error("Dataset not available. Please ensure data/winequality.csv exists.")
    else:
        # Visualization 1: Quality Distribution
        st.markdown("### 1. Wine Quality Distribution")
        if 'quality' in df.columns:
            quality_counts = df['quality'].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(10, 6))
            quality_counts.plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title('Distribution of Wine Quality Scores', fontsize=14, fontweight='bold')
            ax.set_xlabel('Quality Score', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
            
            st.markdown(f"**Quality Range**: {df['quality'].min()} - {df['quality'].max()}")
            st.markdown(f"**Average Quality**: {df['quality'].mean():.2f}")
        
        st.markdown("---")
        
        # Visualization 2: Correlation Heatmap
        st.markdown("### 2. Feature Correlation Matrix")
        show_corr = st.checkbox("Show Correlation Heatmap", value=True)
        if show_corr:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = df.corr()
            sns.heatmap(
                correlation_matrix,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                linewidths=1,
                cbar_kws={"shrink": 0.8},
                ax=ax
            )
            ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=20)
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show top correlations
            st.markdown("**Top Feature Correlations with Quality:**")
            if 'quality' in df.columns:
                quality_corr = df.corr()['quality'].drop('quality').sort_values(ascending=False)
                top_corr = pd.DataFrame({
                    'Feature': quality_corr.index,
                    'Correlation': quality_corr.values
                })
                st.dataframe(top_corr, use_container_width=True)
        
        st.markdown("---")
        
        # Visualization 3: Feature Distribution
        st.markdown("### 3. Feature Distribution Analysis")
        selected_feature = st.selectbox(
            "Select a feature to visualize",
            df.select_dtypes(include=[np.number]).columns.tolist()
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig, ax = plt.subplots(figsize=(8, 6))
            df[selected_feature].hist(bins=30, ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(f'Distribution of {selected_feature}', fontsize=12, fontweight='bold')
            ax.set_xlabel(selected_feature, fontsize=10)
            ax.set_ylabel('Frequency', fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Box plot
            fig, ax = plt.subplots(figsize=(8, 6))
            df.boxplot(column=selected_feature, ax=ax)
            ax.set_title(f'Box Plot of {selected_feature}', fontsize=12, fontweight='bold')
            ax.set_ylabel(selected_feature, fontsize=10)
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.markdown(f"**Statistics for {selected_feature}:**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Mean", f"{df[selected_feature].mean():.2f}")
        with col2:
            st.metric("Median", f"{df[selected_feature].median():.2f}")
        with col3:
            st.metric("Std Dev", f"{df[selected_feature].std():.2f}")
        with col4:
            st.metric("Range", f"{df[selected_feature].max() - df[selected_feature].min():.2f}")
        
        st.markdown("---")
        
        # Visualization 4: Quality vs Features
        st.markdown("### 4. Quality vs Feature Relationships")
        if 'quality' in df.columns:
            feature_for_comparison = st.selectbox(
                "Select a feature to compare with quality",
                [col for col in df.select_dtypes(include=[np.number]).columns if col != 'quality']
            )
            
            fig, ax = plt.subplots(figsize=(10, 6))
            df.boxplot(column=feature_for_comparison, by='quality', ax=ax)
            ax.set_title(f'{feature_for_comparison} by Quality Score', fontsize=12, fontweight='bold')
            ax.set_xlabel('Quality Score', fontsize=10)
            ax.set_ylabel(feature_for_comparison, fontsize=10)
            plt.suptitle('')  # Remove default title
            plt.tight_layout()
            st.pyplot(fig)

# Model Prediction Page
elif page == "Model Prediction":
    st.title("🔮 Model Prediction")
    
    st.markdown("""
    Enter the wine characteristics below to predict its quality.
    The model will classify the wine as **Good Quality** (score ≥ 7) or **Poor Quality** (score < 7).
    """)
    
    st.markdown("---")
    
    # Input widgets organized in columns
    st.markdown("### Input Features")
    
    # Get feature ranges from dataset if available, otherwise use defaults
    if df is not None:
        feature_ranges = {col: (float(df[col].min()), float(df[col].max()), float(df[col].mean())) 
                         for col in feature_names if col in df.columns}
    else:
        # Default ranges based on typical wine quality dataset
        feature_ranges = {
            'fixed acidity': (4.0, 16.0, 8.0),
            'volatile acidity': (0.1, 1.6, 0.5),
            'citric acid': (0.0, 1.0, 0.3),
            'residual sugar': (0.5, 15.0, 2.5),
            'chlorides': (0.01, 0.6, 0.09),
            'free sulfur dioxide': (1.0, 72.0, 15.0),
            'total sulfur dioxide': (6.0, 289.0, 46.0),
            'density': (0.99, 1.0, 0.997),
            'pH': (2.7, 4.0, 3.3),
            'sulphates': (0.3, 2.0, 0.65),
            'alcohol': (8.0, 15.0, 10.5)
        }
    
    # Organize inputs in columns
    num_cols = 3
    cols = st.columns(num_cols)
    
    input_values = {}
    for idx, feature in enumerate(feature_names):
        with cols[idx % num_cols]:
            if feature in feature_ranges:
                min_val, max_val, default_val = feature_ranges[feature]
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=(max_val - min_val) / 100,
                    help=f"Range: {min_val:.2f} - {max_val:.2f}"
                )
            else:
                input_values[feature] = st.number_input(
                    feature.replace('_', ' ').title(),
                    value=0.0
                )
    
    st.markdown("---")
    
    # Prediction button
    if st.button("🔮 Predict Wine Quality", type="primary", use_container_width=True):
        try:
            # Prepare input data
            input_array = np.array([input_values[feature] for feature in feature_names]).reshape(1, -1)
            
            # Make prediction
            with st.spinner("Making prediction..."):
                prediction = model.predict(input_array)[0]
                prediction_proba = model.predict_proba(input_array)[0] if hasattr(model, 'predict_proba') else None
            
            # Display results
            st.markdown("### Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **Good Quality Wine**")
                    st.markdown("Quality Score: **≥ 7**")
                else:
                    st.warning("⚠️ **Poor Quality Wine**")
                    st.markdown("Quality Score: **< 7**")
            
            with col2:
                if prediction_proba is not None:
                    confidence = prediction_proba[prediction] * 100
                    st.metric("Confidence", f"{confidence:.2f}%")
                else:
                    st.metric("Prediction", "Good" if prediction == 1 else "Poor")
            
            with col3:
                if prediction_proba is not None:
                    proba_good = prediction_proba[1] * 100
                    proba_poor = prediction_proba[0] * 100
                    st.metric("Probability (Good)", f"{proba_good:.2f}%")
            
            # Show probability distribution
            if prediction_proba is not None:
                st.markdown("### Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Class': ['Poor Quality (< 7)', 'Good Quality (≥ 7)'],
                    'Probability': [prediction_proba[0] * 100, prediction_proba[1] * 100]
                })
                
                fig, ax = plt.subplots(figsize=(8, 4))
                bars = ax.barh(prob_df['Class'], prob_df['Probability'], 
                               color=['#ff6b6b', '#51cf66'])
                ax.set_xlabel('Probability (%)', fontsize=10)
                ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='bold')
                ax.set_xlim(0, 100)
                
                # Add value labels on bars
                for i, (bar, prob) in enumerate(zip(bars, prob_df['Probability'])):
                    ax.text(prob + 1, i, f'{prob:.2f}%', 
                           va='center', fontsize=10, fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
            
            # Show input summary
            with st.expander("View Input Summary"):
                input_summary = pd.DataFrame({
                    'Feature': feature_names,
                    'Value': [input_values[f] for f in feature_names]
                })
                st.dataframe(input_summary, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            st.info("Please ensure all input values are within valid ranges.")

# Model Performance Page
elif page == "Model Performance":
    st.title("📊 Model Performance")
    
    # Model Information
    st.markdown("### Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Model Type", metadata.get('model_type', 'Classification').title())
    with col2:
        st.metric("Algorithm", metadata.get('best_model', 'Random Forest'))
    with col3:
        st.metric("Test Accuracy", f"{metadata.get('accuracy', 0):.2%}")
    with col4:
        st.metric("CV Accuracy", f"{metadata.get('cv_accuracy', 0):.2%}")
    
    st.markdown("---")
    
    # Model Comparison
    if model_comparison:
        st.markdown("### Model Comparison Results")
        
        # Classification models
        if 'classification' in model_comparison:
            st.markdown("#### Classification Models")
            clf_results = model_comparison['classification']
            
            # Create comparison chart
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(clf_results.keys())
            accuracies = [clf_results[m] * 100 for m in models]
            
            bars = ax.barh(models, accuracies, color=['#4ecdc4', '#45b7d1', '#96ceb4'])
            ax.set_xlabel('Accuracy (%)', fontsize=12)
            ax.set_title('Classification Model Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)
            
            # Add value labels
            for i, (bar, acc) in enumerate(zip(bars, accuracies)):
                ax.text(acc + 1, i, f'{acc:.2f}%', 
                       va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
            
            # Comparison table
            comparison_df = pd.DataFrame({
                'Model': models,
                'Accuracy': [f"{acc:.4f}" for acc in clf_results.values()],
                'Accuracy (%)': [f"{acc*100:.2f}%" for acc in clf_results.values()]
            }).sort_values('Accuracy', ascending=False, key=lambda x: x.str.replace('%', '').astype(float))
            
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Regression models
        if 'regression' in model_comparison:
            st.markdown("#### Regression Models")
            reg_results = model_comparison['regression']
            
            fig, ax = plt.subplots(figsize=(10, 6))
            models = list(reg_results.keys())
            r2_scores = [reg_results[m] * 100 for m in models]
            
            bars = ax.barh(models, r2_scores, color=['#4ecdc4', '#45b7d1'])
            ax.set_xlabel('R² Score (%)', fontsize=12)
            ax.set_title('Regression Model Comparison', fontsize=14, fontweight='bold')
            ax.set_xlim(0, 100)
            
            for i, (bar, score) in enumerate(zip(bars, r2_scores)):
                ax.text(score + 1, i, f'{score:.2f}%', 
                       va='center', fontsize=10, fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
    
    st.markdown("---")
    
    # Confusion Matrix
    if confusion_matrix_data is not None:
        st.markdown("### Confusion Matrix")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            confusion_matrix_data,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=['Poor Quality', 'Good Quality'],
            yticklabels=['Poor Quality', 'Good Quality'],
            ax=ax
        )
        ax.set_xlabel('Predicted', fontsize=12)
        ax.set_ylabel('Actual', fontsize=12)
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = confusion_matrix_data.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        st.markdown("#### Performance Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy:.2%}")
        with col2:
            st.metric("Precision", f"{precision:.2%}")
        with col3:
            st.metric("Recall", f"{recall:.2%}")
        with col4:
            st.metric("F1-Score", f"{f1_score:.2%}")
        
        # Detailed metrics table
        metrics_df = pd.DataFrame({
            'Metric': ['True Negatives', 'False Positives', 'False Negatives', 'True Positives'],
            'Count': [tn, fp, fn, tp]
        })
        st.dataframe(metrics_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Feature Importance (if available)
    if df is not None and hasattr(model, 'feature_importances_'):
        st.markdown("### Feature Importance")
        
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax, palette='viridis')
        ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importance', fontsize=12)
        plt.tight_layout()
        st.pyplot(fig)
        
        st.dataframe(feature_importance, use_container_width=True, hide_index=True)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📚 About
**Wine Quality Prediction App**

Built with Streamlit for ML model deployment.

### 🔗 Links
- [Streamlit Documentation](https://docs.streamlit.io)
- [Scikit-learn Documentation](https://scikit-learn.org)
""")

