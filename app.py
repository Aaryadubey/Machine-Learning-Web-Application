import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- Helper Function for Preprocessing ---
def preprocess_data(df):
    """
    A simple preprocessing function that handles missing values and scales numerical data.
    """
    # Drop rows with any missing values for simplicity
    df_processed = df.dropna()
    numeric_cols = df_processed.select_dtypes(include=np.number).columns.tolist()
    
    if not numeric_cols:
        return df_processed, [] # Return if no numeric columns
        
    # Scale numerical features
    scaler = StandardScaler()
    df_processed[numeric_cols] = scaler.fit_transform(df_processed[numeric_cols])
    
    return df_processed, numeric_cols

# --- App Title and Description ---
st.title("🤖 Machine Learning Analysis Web App")
st.write(
    "This modular app provides three core analysis functionalities: "
    "Clustering, Classification, and Time-Series Activity Analysis. "
    "Upload your data and select an analysis to begin!"
)

# --- Sidebar for Navigation and Upload ---
st.sidebar.header("⚙️ Controls")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type", 
    ["Clustering (K-Means)", "Classification (Random Forest/Decision Tree)", "Time-Series Activity Analysis"]
)

st.sidebar.header("Upload your CSV Data")
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])

# --- Main Panel Logic ---
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("Uploaded Data Preview")
    st.write(df.head())

    # --- Clustering Module ---
    if analysis_type == "Clustering (K-Means)":
        st.header("K-Means Clustering Analysis")
        
        df_processed, numeric_cols = preprocess_data(df.copy())
        
        if len(numeric_cols) < 2:
            st.warning("Clustering requires at least two numerical columns in your data after handling missing values.")
        else:
            st.write("Data has been preprocessed (scaled). Select the number of clusters.")
            k = st.slider("Select number of clusters (K)", 2, 10, 3, key="k_slider")
            
            # Run K-Means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
            df_processed['cluster'] = kmeans.fit_predict(df_processed[numeric_cols])
            
            st.write("Clustering Results:")
            st.write(df_processed.head())

            # Visualize clusters using the first two numeric columns
            fig, ax = plt.subplots(figsize=(10, 6))
            scatter = ax.scatter(
                df_processed[numeric_cols[0]], 
                df_processed[numeric_cols[1]], 
                c=df_processed['cluster'], 
                cmap='viridis',
                alpha=0.7
            )
            ax.set_title(f'K-Means Clustering (K={k}) on {numeric_cols[0]} vs {numeric_cols[1]}')
            ax.set_xlabel(f"Scaled {numeric_cols[0]}")
            ax.set_ylabel(f"Scaled {numeric_cols[1]}")
            legend1 = ax.legend(*scatter.legend_elements(), title="Clusters")
            ax.add_artist(legend1)
            st.pyplot(fig)

    # --- Classification Module ---
    elif analysis_type in ["Classification (Random Forest/Decision Tree)", "Time-Series Activity Analysis"]:
        if analysis_type == "Time-Series Activity Analysis":
            st.header("Time-Series Activity Analysis (as Classification)")
            st.info("This section treats your time-series data as a classification problem. Each row is a time-step with features and an activity label.")
        else:
            st.header("Classification Analysis")

        df_processed, _ = preprocess_data(df.copy())
        
        all_cols = df_processed.columns.tolist()
        # Ensure target is not pre-selected in features
        target_col = st.selectbox("1. Select Target Variable", all_cols, key="target")
        
        available_features = [col for col in all_cols if col != target_col]
        feature_cols = st.multiselect(
            "2. Select Feature Variables", 
            available_features, 
            default=available_features[:min(3, len(available_features))], # Default to first 3 features
            key="features"
        )
        
        classifier_name = st.selectbox("3. Select Classifier", ["Random Forest", "Decision Tree"], key="classifier")

        if st.button("Run Classification Analysis", key="run_clf"):
            if not feature_cols:
                st.warning("Please select at least one feature variable.")
            else:
                X = df_processed[feature_cols]
                y = df_processed[target_col]
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y if pd.api.types.is_categorical_dtype(y) or y.nunique() > 1 else None)
                
                # Select and train model
                if classifier_name == "Random Forest":
                    model = RandomForestClassifier(random_state=42)
                else:
                    model = DecisionTreeClassifier(random_state=42)
                    
                model.fit(X_train, y_train)
                
                # Evaluate and display performance
                y_train_pred = model.predict(X_train)
                y_test_pred = model.predict(X_test)
                
                training_accuracy = accuracy_score(y_train, y_train_pred)
                test_accuracy = accuracy_score(y_test, y_test_pred)
                
                st.subheader("Model Performance")
                st.metric(label=f"**Training Accuracy** ({classifier_name})", value=f"{training_accuracy:.4f}")
                st.metric(label=f"**Test Accuracy** ({classifier_name})", value=f"{test_accuracy:.4f}")
                st.info("Test accuracy is the primary indicator of the model's performance on new, unseen data.")

else:
    st.info("Awaiting for CSV file to be uploaded. Please use the sidebar to upload your data.")