# Copyright 2025 Arshvir
# Licensed under the Apache License, Version 2.0
# See LICENSE file in the project root for full license information.

#---------------------------------------------------------------------#

import streamlit as st
import webbrowser
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle
import base64
import io
import plotly.graph_objects as go
#import viz_report
import viz_ai_img
import word_cloud
import notepad_lite
import calculator


# Import ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB # For classification
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np # For numerical operations, especially with metrics

st.set_page_config("Visio AI", page_icon="images/favicon.png", layout='wide')

st.markdown("<h1 style='text-align: center; color: #4A90E2;'>📊 VISIO AI</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: orange;'>Machine Learning and Data Analysis Platform</h4>", unsafe_allow_html=True)
st.markdown("<hr>", unsafe_allow_html=True)
#-------------------------------------------------#

# --- Session State Initialization ---
if 'updated_df' not in st.session_state:
    st.session_state.updated_df = None
if 'original_df_uploaded' not in st.session_state:
    st.session_state.original_df_uploaded = False
if 'last_uploaded_file_name' not in st.session_state:
    st.session_state.last_uploaded_file_name = None
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None
if 'feature_columns' not in st.session_state:
    st.session_state.feature_columns = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None # 'classification' or 'regression'
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'model_metrics' not in st.session_state:
    st.session_state.model_metrics = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None


# Navigation Bar
col1, col2, col3, col4, col5 = st.columns((1, 1, 1, 1, 1))
with col1:
    about_url = "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_about.html"
    if st.button('About'):
        st.markdown("check out this [link](%s)" % about_url)
        
        #webbrowser.open_new_tab(about_url)

with col2:
    guide_url = "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_helper.html"
    if st.button('Guide'):
        st.markdown("check out this [link](%s)" % guide_url)
        
with col3:
    docs_url = "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_docs.html"
    if st.button('Docs'):
        st.markdown("check out this [link](%s)" % docs_url)

with col4:
    joinus_url = "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/visio_join.html"
    if st.button('Join Us'):
        st.markdown("check out this [link](%s)" % joinus_url)

with col5:
    elite_access = "https://jaiho-labs.onrender.com/pages/products_resources/docs/visio_ai_docs/get_elite_access.html"
    if st.button('Get Elite Access'):
        st.markdown("check out this [link](%s)" % elite_access)

#-------------------------------------------------#

# Top Expander Columns (Data Operations & Algorithms, Select Plot Type, Pre Analysis)
col11, col12, col13 = st.columns([1, 1, 1])

# --- Data Operations & Algorithms Expander ---
with col11:
    with st.expander("⚙️ Data Operations & Algorithms", expanded=False):
        if st.session_state.updated_df is not None:
            st.markdown("#### 1. Define Target Variable and Problem Type")
            all_columns = st.session_state.updated_df.columns.tolist()
            target_column = st.selectbox("Select your **Target Column (Y)**:", ["--- Select ---"] + all_columns, key="target_col_select")

            if target_column != "--- Select ---":
                st.session_state.target_column = target_column
                # Heuristic to guess problem type
                if st.session_state.updated_df[target_column].dtype in ['int64', 'float64']:
                    if st.session_state.updated_df[target_column].nunique() < 20 and st.session_state.updated_df[target_column].dtype == 'int64':
                        st.session_state.problem_type = 'classification'
                        st.info(f"Detected **Classification** problem based on target column '{target_column}' (integer with few unique values).")
                    else:
                        st.session_state.problem_type = 'regression'
                        st.info(f"Detected **Regression** problem based on target column '{target_column}' (numerical).")
                elif st.session_state.updated_df[target_column].dtype == 'object' or st.session_state.updated_df[target_column].dtype == 'bool':
                    st.session_state.problem_type = 'classification'
                    st.info(f"Detected **Classification** problem based on target column '{target_column}' (categorical).")
                else:
                    st.session_state.problem_type = None
                    st.warning("Could not definitively determine problem type. Please proceed with caution.")

                st.markdown("---")
                st.markdown("#### 2. Select Independent Variables (Features)")
                
                available_features = [col for col in all_columns if col != target_column]
                feature_columns = st.multiselect("Select your **Independent Variables (X)**:", available_features, default=available_features, key="feature_select")
                
                if feature_columns:
                    st.session_state.feature_columns = feature_columns
                    st.markdown("---")
                    st.markdown("#### 3. Split Data into Train and Test Sets")

                    test_size = st.slider("Select Test Set Size:", min_value=0.1, max_value=0.5, value=0.2, step=0.05, key="test_size_slider")
                    random_state = st.number_input("Random State (for reproducibility):", value=42, step=1, key="random_state_input")
                    
                    # Use only selected features
                    features = st.session_state.updated_df[feature_columns]
                    target = st.session_state.updated_df[target_column]

                    # Handle categorical features by encoding
                    for col in features.select_dtypes(include=['object', 'bool']).columns:
                        le = LabelEncoder()
                        features[col] = le.fit_transform(features[col].astype(str))

                    # Handle numerical features by scaling
                    numerical_cols = features.select_dtypes(include=['number']).columns
                    if not numerical_cols.empty:
                        scaler = StandardScaler()
                        features[numerical_cols] = scaler.fit_transform(features[numerical_cols])
                        st.session_state.scaler = scaler # Save the scaler

                    try:
                        X_train, X_test, y_train, y_test = train_test_split(
                            features, target, test_size=test_size, random_state=random_state,
                            stratify=target if st.session_state.problem_type == 'classification' else None
                        )
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.success(f"Data split successfully! Training: {len(X_train)} samples, Testing: {len(X_test)} samples.")

                        st.markdown("---")
                        st.markdown("#### 4. Select Machine Learning Algorithm")

                        if st.session_state.problem_type == 'classification':
                            algo_options = {
                                "Logistic Regression": LogisticRegression(random_state=random_state),
                                "Decision Tree Classifier": DecisionTreeClassifier(random_state=random_state),
                                "Random Forest Classifier": RandomForestClassifier(random_state=random_state),
                                "Support Vector Classifier (SVC)": SVC(random_state=random_state),
                                "K-Nearest Neighbors Classifier": KNeighborsClassifier(),
                                "Gaussian Naive Bayes": GaussianNB()
                            }
                            algo_name = st.selectbox("Choose a Classification Algorithm:", list(algo_options.keys()), key="classification_algo_select")
                            selected_algo = algo_options.get(algo_name)

                        elif st.session_state.problem_type == 'regression':
                            algo_options = {
                                "Linear Regression": LinearRegression(),
                                "Decision Tree Regressor": DecisionTreeRegressor(random_state=random_state),
                                "Random Forest Regressor": RandomForestRegressor(random_state=random_state),
                                "Support Vector Regressor (SVR)": SVR(),
                                "K-Nearest Neighbors Regressor": KNeighborsRegressor()
                            }
                            algo_name = st.selectbox("Choose a Regression Algorithm:", list(algo_options.keys()), key="regression_algo_select")
                            selected_algo = algo_options.get(algo_name)
                        else:
                            st.warning("Please define target column and problem type to select an algorithm.")
                            selected_algo = None

                        if selected_algo:
                            st.info(f"Selected Algorithm: **{algo_name}**")
                            st.session_state.selected_algo = selected_algo
                            st.session_state.selected_algo_name = algo_name
                            st.markdown("---")
                            if st.button("🚀 Train Model"):
                                if st.session_state.X_train is not None and st.session_state.y_train is not None:
                                    try:
                                        with st.spinner(f"Training {st.session_state.selected_algo_name}..."):
                                            st.session_state.selected_algo.fit(st.session_state.X_train, st.session_state.y_train)
                                        st.session_state.trained_model = st.session_state.selected_algo
                                        st.success(f"Model **{st.session_state.selected_algo_name}** trained successfully!")

                                        y_pred = st.session_state.trained_model.predict(st.session_state.X_test)
                                        metrics = {}
                                        if st.session_state.problem_type == 'classification':
                                            metrics['Accuracy'] = accuracy_score(st.session_state.y_test, y_pred)
                                            metrics['Precision'] = precision_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                                            metrics['Recall'] = recall_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                                            metrics['F1 Score'] = f1_score(st.session_state.y_test, y_pred, average='weighted', zero_division=0)
                                            metrics['Confusion Matrix'] = confusion_matrix(st.session_state.y_test, y_pred)
                                        elif st.session_state.problem_type == 'regression':
                                            metrics['Mean Squared Error'] = mean_squared_error(st.session_state.y_test, y_pred)
                                            metrics['R2 Score'] = r2_score(st.session_state.y_test, y_pred)
                                        st.session_state.model_metrics = metrics
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error training model: {e}")
                                else:
                                    st.warning("Please split the data first before training the model.")
                        else:
                             st.warning("Please select a target column and problem type to enable algorithm selection.")
                    except Exception as e:
                        st.error(f"Error splitting data or preparing features: {e}")
                        st.info("Ensure your data is clean and suitable for splitting (e.g., no remaining NaN values after imputation).")
                else:
                    st.warning("Please select at least one independent variable.")
            else:
                st.info("Please select a target column to proceed with data operations.")
        else:
            st.info("Please upload a dataset first to access Data Operations & Algorithms.")


with col12:
    with st.expander("🎨 Select Plot Type", expanded=False):
        if st.session_state.updated_df is not None:
            df = st.session_state.updated_df
            numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
            categorical_cols = df.select_dtypes(include='object').columns.tolist()
            
            plot_type = st.selectbox("Select a plot type", ["---Select---", "Bar Chart", "Histogram", "Scatter Plot", "Box Plot", "Heatmap", 
                                                            "Line Chart", "Pie Chart", "Violin Plot", "Pair Plot", 
                                                             "3D Scatter Plot", "3D Surface Plot"])

            if plot_type == "Bar Chart":
                st.info("A bar chart shows counts of categories within a column.")
                selected_col = st.selectbox("Select a categorical column", categorical_cols)
                if st.button("Generate Bar Chart"):
                    if selected_col:
                        fig = px.bar(df, x=selected_col, title=f'Bar Chart of {selected_col}', color=selected_col)
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Histogram":
                st.info("A histogram shows the distribution of a numerical column.")
                selected_col = st.selectbox("Select a numerical column", numerical_cols)
                if st.button("Generate Histogram"):
                    if selected_col:
                        fig = px.histogram(df, x=selected_col, title=f'Histogram of {selected_col}')
                        st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Scatter Plot":
                st.info("A scatter plot shows the relationship between two numerical columns.")
                x_col = st.selectbox("Select X-axis column", numerical_cols, key='scatter_x')
                y_col = st.selectbox("Select Y-axis column", numerical_cols, key='scatter_y')
                if st.button("Generate Scatter Plot"):
                    if x_col and y_col:
                        fig = px.scatter(df, x=x_col, y=y_col, title=f'Scatter Plot of {x_col} vs {y_col}')
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Box Plot":
                st.info("A box plot shows the distribution of a numerical column grouped by a categorical column.")
                num_col = st.selectbox("Select a numerical column", numerical_cols, key='box_num')
                cat_col = st.selectbox("Select a categorical column for grouping", categorical_cols, key='box_cat')
                if st.button("Generate Box Plot"):
                    if num_col and cat_col:
                        fig = px.box(df, x=cat_col, y=num_col, title=f'Box Plot of {num_col} by {cat_col}', color=cat_col)
                        st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Heatmap":
                st.info("A heatmap shows the correlation between all numerical columns.")
                if st.button("Generate Heatmap"):
                    corr = df[numerical_cols].corr()
                    fig = px.imshow(corr, text_auto=True, title='Correlation Heatmap')
                    st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Line Chart":
                st.info("A line chart shows trends over time or ordered categories.")
                x_col = st.selectbox("Select X-axis column", df.columns, key='line_x')
                y_col = st.selectbox("Select Y-axis (numerical) column", numerical_cols, key='line_y')
                if st.button("Generate Line Chart"):
                    if x_col and y_col:
                        fig = px.line(df, x=x_col, y=y_col, title=f'Line Chart of {y_col} over {x_col}')
                        st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Pie Chart":
                st.info("A pie chart shows proportions of categories within a column.")
                selected_col = st.selectbox("Select a categorical column for Pie Chart", categorical_cols, key='pie_col')
                if st.button("Generate Pie Chart"):
                    if selected_col:
                        pie_data = df[selected_col].value_counts().reset_index()
                        pie_data.columns = [selected_col, 'Count']
                        fig = px.pie(pie_data, names=selected_col, values='Count', title=f'Pie Chart of {selected_col}')
                        st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "Violin Plot":
                st.info("A violin plot shows the distribution of a numerical column by categories.")
                num_col = st.selectbox("Select a numerical column", numerical_cols, key='violin_num')
                cat_col = st.selectbox("Select a categorical column for grouping", categorical_cols, key='violin_cat')
                if st.button("Generate Violin Plot"):
                    if num_col and cat_col:
                        fig = px.violin(df, x=cat_col, y=num_col, box=True, points="all", title=f'Violin Plot of {num_col} by {cat_col}')
                        st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "Pair Plot":
                st.info("A pair plot shows scatter plots for all combinations of numerical columns.")
                if st.button("Generate Pair Plot"):
                    fig = px.scatter_matrix(df[numerical_cols], dimensions=numerical_cols, title='Pair Plot of Numerical Features')
                    st.plotly_chart(fig, use_container_width=True)
            
            elif plot_type == "3D Scatter Plot":
                st.info("A 3D scatter plot shows the relationship between three numerical columns.")
                x_col = st.selectbox("Select X-axis column", numerical_cols, key='3d_scatter_x')
                y_col = st.selectbox("Select Y-axis column", numerical_cols, key='3d_scatter_y')
                z_col = st.selectbox("Select Z-axis column", numerical_cols, key='3d_scatter_z')
                color_col = st.selectbox("Optional: Select a column for color grouping (optional)", df.columns, key='3d_scatter_color')
                if st.button("Generate 3D Scatter Plot"):
                    if x_col and y_col and z_col:
                        fig = px.scatter_3d(df, x=x_col, y=y_col, z=z_col, color=color_col if color_col else None,
                                title=f'3D Scatter Plot: {x_col} vs {y_col} vs {z_col}')
                        st.plotly_chart(fig, use_container_width=True)

            elif plot_type == "3D Surface Plot":
                st.info("A 3D surface plot shows a continuous surface over two variables.")
                x_col = st.selectbox("Select X-axis column", numerical_cols, key='3d_surface_x')
                y_col = st.selectbox("Select Y-axis column", numerical_cols, key='3d_surface_y')
                z_col = st.selectbox("Select Z-axis column", numerical_cols, key='3d_surface_z')
                if st.button("Generate 3D Surface Plot"):
                    if x_col and y_col and z_col:
                        try:
                            pivot_table = df.pivot_table(index=y_col, columns=x_col, values=z_col, aggfunc='mean')
                            fig = go.Figure(data=[go.Surface(z=pivot_table.values, 
                                                            x=pivot_table.columns, 
                                                            y=pivot_table.index)])
                            fig.update_layout(title=f'3D Surface Plot of {z_col} over {x_col} and {y_col}',
                                            scene=dict(
                                                xaxis_title=x_col,
                                                yaxis_title=y_col,
                                                zaxis_title=z_col
                                            ))
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error generating surface plot: {e}")


        else:
            st.info("Please upload a dataset first to generate plots.")

with col13:
    with st.expander("📈 Pre Analysis", expanded=False):
        if st.session_state.updated_df is not None:
            # Create tabs for different analyses
            tab1, tab2 = st.tabs(["Statistical Summary", "Dataset Info"])
            
            with tab1:
                st.subheader("Statistical Summary (describe)")
                numeric_df = st.session_state.updated_df.select_dtypes(include=['float64', 'int64'])
                if not numeric_df.empty:
                # Display statistical summary
                    st.dataframe(numeric_df.describe())
                else:
                    st.warning("No numerical columns found in the dataset")

                if st.checkbox("Show additional statistics"):
                    st.write("Skewness:")
                    st.dataframe(numeric_df.skew())
                    st.write("Kurtosis:")
                    st.dataframe(numeric_df.kurtosis())

            with tab2:
                st.subheader("Dataset Information (info)")
                # Get DataFrame info
                buffer = io.StringIO()
                st.session_state.updated_df.info(buf=buffer)
                info_str = buffer.getvalue()
                
            # Display formatted info
                st.text(info_str)

                st.write("Quick Facts:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", st.session_state.updated_df.shape[0])
                with col2:
                    st.metric("Total Columns", st.session_state.updated_df.shape[1])
                with col3:
                    st.metric("Missing Values", st.session_state.updated_df.isna().sum().sum())
                
            # Display column types
                st.write("Column Data Types:")
                dtypes_df = pd.DataFrame(st.session_state.updated_df.dtypes, columns=['Data Type'])
                st.dataframe(dtypes_df)
        else:
            st.info("Please upload a dataset first.")


#----------------------------------------------------#

# Sidebar (Keep as is if you are simulating pages in a single file)
with st.sidebar:
    st.markdown('<b>🛠️ Tools</b>', unsafe_allow_html=True)

    # Store the active page in session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "main"

    if st.button("🏠 Home"):
        st.session_state.current_page = "main"
        st.rerun()

    if st.button("📝 Note -- Lite"):
        st.session_state.current_page = "note_lite"
        st.rerun()

    if st.button("😶‍🌫️ WordCloud"):
        st.session_state.current_page = "word_cloud"
        st.rerun()

    if st.button("🤖 Viz AI (img)"):
        st.session_state.current_page = "viz_ai_img"
        st.rerun()

    if st.button("🧮 Calculator"):
        st.session_state.current_page = "calculator"
        st.rerun()

    if st.button("⚙️ Viz Editor"):
        st.session_state.current_page = "note_edit"
        # No rerun here — handled differently maybe?

    if st.button("📄 Viz Report"):
        st.session_state.current_page = "generate_report"
        st.rerun()

    st.markdown("<hr>",unsafe_allow_html=True)
    st.markdown("### <center>Other Products</center>", unsafe_allow_html=True)
    

#---------------------------------------------------------------#

#---------------------------------------------------------------#

# Main content columns
col_main_left, col_main_right = st.columns([0.6, 0.4]) # Adjusted column widths for better layout

with col_main_left:
    st.markdown("<b style='font-size:20px;'>📂 Upload Your Dataset</b>", unsafe_allow_html=True)
    dataset = st.file_uploader("Choose a dataset file", type=["csv", "xlsx", "txt"], key="file_uploader_main") # Added key

    if dataset is not None:
        if 'last_uploaded_file_object' not in st.session_state or st.session_state.last_uploaded_file_object != dataset:
            st.session_state.last_uploaded_file_object = dataset
            st.session_state.original_df_uploaded = False
            st.session_state.updated_df = None
            st.session_state.X_train = st.session_state.X_test = st.session_state.y_train = st.session_state.y_test = None
            st.session_state.target_column = None
            st.session_state.feature_columns = None
            st.session_state.problem_type = None
            st.session_state.trained_model = None
            st.session_state.model_metrics = None
            st.session_state.scaler = None


            st.success("✅ File uploaded successfully!")
            st.write(f"File name: **{dataset.name}**")

            try:
                if dataset.name.endswith(".csv"):
                    df = pd.read_csv(dataset)
                elif dataset.name.endswith(".xlsx"):
                    df = pd.read_excel(dataset)
                elif dataset.name.endswith(".txt"):
                    df = pd.read_csv(dataset, delimiter="\t")
                else:
                    st.error("Unsupported file type. Please upload a CSV, XLSX, or TXT (tab-separated) file.")
                    df = None

                if df is not None:
                    st.session_state.updated_df = df.copy()
                    st.session_state.original_df_uploaded = True
                    st.rerun()

            except Exception as e:
                st.error(f"Error reading file: {e}. Please ensure it's a valid CSV, XLSX, or tab-separated TXT.")
                st.session_state.original_df_uploaded = False
                st.session_state.updated_df = None


    # Original Dataset Preview
    if st.session_state.original_df_uploaded and st.session_state.updated_df is not None:
        st.markdown('<div class="dataset-preview">', unsafe_allow_html=True)
        st.subheader("🔍 Original Dataset Preview")
        st.dataframe(st.session_state.updated_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Updated Dataset Preview (after imputation)
        st.markdown('<div class="dataset-preview">', unsafe_allow_html=True)
        st.subheader("🔄 Updated Dataset Preview (After Imputation)")
        st.dataframe(st.session_state.updated_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)


with col_main_right:
    if st.session_state.updated_df is not None:
        st.markdown('<div class="section-title">📊 Missing Values Report</div>', unsafe_allow_html=True)
        null_counts = st.session_state.updated_df.isnull().sum()
        total_nulls = null_counts.sum()

        if total_nulls == 0:
            st.success("✅ No null values found in the dataset!")
        else:
            st.warning(f"⚠️ Found {total_nulls} null values in the dataset.")
            st.write(null_counts[null_counts > 0])

            # Automatic Missing Value Handling
            st.markdown('<div class="section-title">🤖 Automatic Missing Value Handling</div>', unsafe_allow_html=True)

            with st.form("auto_impute_form"):
                st.write("Apply default handling for all missing values:")
                auto_impute_option = st.selectbox(
                    "Choose imputation method:",
                    ["None", "Mean (Numerical)", "Median (Numerical)", "Mode (All)", "Forward Fill", "Backward Fill"],
                    key="auto_impute_method"
                )
                auto_impute_button = st.form_submit_button("Apply Automatic Imputation")

                if auto_impute_button and auto_impute_option != "None":
                    df_to_impute = st.session_state.updated_df.copy()

                    if auto_impute_option == "Mean (Numerical)":
                        for col in df_to_impute.select_dtypes(include=['number']).columns:
                            if df_to_impute[col].isnull().sum() > 0:
                                df_to_impute[col].fillna(df_to_impute[col].mean(), inplace=True)
                    elif auto_impute_option == "Median (Numerical)":
                        for col in df_to_impute.select_dtypes(include=['number']).columns:
                            if df_to_impute[col].isnull().sum() > 0:
                                df_to_impute[col].fillna(df_to_impute[col].median(), inplace=True)
                    elif auto_impute_option == "Mode (All)":
                        for col in df_to_impute.columns:
                            if df_to_impute[col].isnull().sum() > 0:
                                if not df_to_impute[col].mode().empty:
                                    df_to_impute[col].fillna(df_to_impute[col].mode()[0], inplace=True)
                                else:
                                    st.warning(f"Could not compute mode for column '{col}'. Skipping.")
                    elif auto_impute_option == "Forward Fill":
                        df_to_impute.fillna(method='ffill', inplace=True)
                    elif auto_impute_option == "Backward Fill":
                        df_to_impute.fillna(method='bfill', inplace=True)

                    st.session_state.updated_df = df_to_impute
                    st.success(f"🎉 Missing values have been handled automatically using **{auto_impute_option}**!")
                    st.rerun()

            # Manual Missing Value Handling
            st.markdown('<div class="section-title">🛠️ Manual Missing Value Handling</div>', unsafe_allow_html=True)

            cols_with_missing = st.session_state.updated_df.columns[st.session_state.updated_df.isnull().any()].tolist()

            if cols_with_missing:
                selected_col_manual = st.selectbox(
                    "Select a column to manually handle missing values:",
                    ["--- Select a Column ---"] + cols_with_missing,
                    key="manual_col_select"
                )

                if selected_col_manual != "--- Select a Column ---":
                    col_dtype = st.session_state.updated_df[selected_col_manual].dtype
                    num_missing = st.session_state.updated_df[selected_col_manual].isnull().sum()
                    st.write(f"Column: **{selected_col_manual}** (Missing values: **{num_missing}**)")

                    with st.form(key=f"manual_impute_form_{selected_col_manual}"):
                        fill_value_to_apply = None
                        if col_dtype == "object":
                            manual_fill_option = st.selectbox(
                                f"Choose a method for '{selected_col_manual}'",
                                ["Mode", "Fill with custom value"],
                                key=f"cat_method_{selected_col_manual}"
                            )
                            if manual_fill_option == "Fill with custom value":
                                fill_value_to_apply = st.text_input(f"Enter the custom value to fill for '{selected_col_manual}'", key=f"cat_value_{selected_col_manual}")
                            elif manual_fill_option == "Mode":
                                if not st.session_state.updated_df[selected_col_manual].mode().empty:
                                    fill_value_to_apply = st.session_state.updated_df[selected_col_manual].mode()[0]
                                else:
                                    st.warning(f"Mode cannot be calculated for {selected_col_manual}. Please enter a custom value.")

                        else:
                            manual_fill_option = st.selectbox(
                                f"Choose a method for '{selected_col_manual}'",
                                ["Mean", "Median", "Mode", "Fill with custom value"],
                                key=f"num_method_{selected_col_manual}"
                            )
                            if manual_fill_option == "Fill with custom value":
                                fill_value_to_apply = st.number_input(f"Enter the custom value to fill for '{selected_col_manual}'", value=0.0, key=f"num_value_{selected_col_manual}")
                            elif manual_fill_option == "Mean":
                                fill_value_to_apply = st.session_state.updated_df[selected_col_manual].mean()
                            elif manual_fill_option == "Median":
                                fill_value_to_apply = st.session_state.updated_df[selected_col_manual].median()
                            elif manual_fill_option == "Mode":
                                if not st.session_state.updated_df[selected_col_manual].mode().empty:
                                    fill_value_to_apply = st.session_state.updated_df[selected_col_manual].mode()[0]
                                else:
                                    st.warning(f"Mode cannot be calculated for {selected_col_manual}. Please enter a custom value.")

                        submit_button = st.form_submit_button(f"Apply Manual Imputation to {selected_col_manual}")

                        if submit_button and fill_value_to_apply is not None:
                            st.session_state.updated_df[selected_col_manual].fillna(fill_value_to_apply, inplace=True)
                            st.success(f"Filled '{selected_col_manual}' missing values with **'{fill_value_to_apply}'** using {manual_fill_option}!")
                            st.rerun()
            else:
                st.info("No columns with missing values to display for manual handling.")

        # Pair Plot button is now below the missing values report
        st.markdown("---")
        if st.button("📈 Generate Pair Plot of Numerical Columns"):
            if st.session_state.updated_df is not None:
                numerical_data = st.session_state.updated_df.select_dtypes(include=['float64', 'int64'])
                if not numerical_data.empty:
                    st.markdown("##### 📘 Pair Plot - Seaborn (Static)", unsafe_allow_html=True)
                    fig1 = sns.pairplot(numerical_data)
                    st.pyplot(fig1)
                    plt.clf()
                    st.markdown("##### 🧠 Pair Plot - Plotly (Interactive)", unsafe_allow_html=True)
                    fig2 = px.scatter_matrix(numerical_data,
                                             dimensions=numerical_data.columns,
                                             height=800, width=800)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.warning("No numerical columns found to generate a pair plot.")
            else:
                st.warning("Please upload and process a dataset first.")


# --- Machine Learning Operations Section (Full Width, below the two columns) ---
st.markdown("---")
st.markdown("<h2 style='text-align: center; color: #4A90E2;'>🧠 Machine Learning Operations</h2>", unsafe_allow_html=True)

if st.session_state.updated_df is not None and st.session_state.trained_model is not None:
    st.markdown(f"### Model Training Results for **{st.session_state.selected_algo_name}**")

    if st.session_state.model_metrics:
        if st.session_state.problem_type == 'classification':
            st.markdown("#### Classification Metrics:")
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(label="Accuracy", value=f"{st.session_state.model_metrics['Accuracy']:.4f}")
            with col_m2:
                st.metric(label="Precision", value=f"{st.session_state.model_metrics['Precision']:.4f}")
            with col_m3:
                st.metric(label="Recall", value=f"{st.session_state.model_metrics['Recall']:.4f}")
            with col_m4:
                st.metric(label="F1 Score", value=f"{st.session_state.model_metrics['F1 Score']:.4f}")

            st.markdown("#### Confusion Matrix:")
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            sns.heatmap(st.session_state.model_metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            ax_cm.set_title('Confusion Matrix')
            st.pyplot(fig_cm)
            plt.clf()

        elif st.session_state.problem_type == 'regression':
            st.markdown("#### Regression Metrics:")
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.metric(label="Mean Squared Error", value=f"{st.session_state.model_metrics['Mean Squared Error']:.4f}")
            with col_r2:
                st.metric(label="R2 Score", value=f"{st.session_state.model_metrics['R2 Score']:.4f}")
                
    st.markdown("---")
    
    # --- Test Your Own Values and Download Model ---
    col_test, col_download = st.columns(2)

    with col_test:
        st.markdown("### 🧪 Test with Your Own Values")
        
        if st.session_state.feature_columns:
            input_data = {}
            for col in st.session_state.feature_columns:
                if st.session_state.updated_df[col].dtype == 'object':
                    unique_vals = st.session_state.updated_df[col].unique()
                    input_data[col] = st.selectbox(f"Select value for **{col}**", unique_vals)
                else:
                    input_data[col] = st.number_input(f"Enter value for **{col}**", value=float(st.session_state.updated_df[col].mean()))
            
            if st.button("Get Prediction"):
                input_df = pd.DataFrame([input_data])
                
                # Preprocess the input data similarly to the training data
                for col in input_df.select_dtypes(include=['object', 'bool']).columns:
                    le = LabelEncoder()
                    input_df[col] = le.fit_transform(input_df[col].astype(str))

                if st.session_state.scaler:
                    numerical_cols = input_df.select_dtypes(include=['number']).columns
                    if not numerical_cols.empty:
                        input_df[numerical_cols] = st.session_state.scaler.transform(input_df[numerical_cols])
                        
                prediction = st.session_state.trained_model.predict(input_df)
                st.success(f"**Prediction:** {prediction[0]}")

    with col_download:
        st.markdown("### 📥 Download Trained Model")
        
        # Serialize the model for download
        model_pkl = pickle.dumps(st.session_state.trained_model)
        b64 = base64.b64encode(model_pkl).decode()
        
        st.download_button(
            label="Download Model as .pkl",
            data=base64.b64decode(b64),
            file_name=f"{st.session_state.selected_algo_name}_model.pkl",
            mime="application/octet-stream"
        )


else:
    st.info("Upload a dataset and train a model to see results and test your own values.")

if st.session_state.current_page == "viz_ai_img":
    viz_ai_img.analyze_image_ui()

elif st.session_state.current_page == "word_cloud":
    # Make sure to import your word_cloud module if you have it
    word_cloud.render_word_cloud_page()

elif st.session_state.current_page == "note_lite":
    notepad_lite.render_notepad()

elif st.session_state.current_page == "calculator":
    calculator.render_calculator()

elif st.session_state.current_page == "generate_report":
    # Make sure to import your viz_report module if you have it
    # viz_report.generate_report()
    #viz_report.render_report_page()
    st.write("Viz Report Page (Implement logic here)")

# Add custom CSS for better styling
st.markdown("""
<style>
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        border: 1px solid #4A90E2;
        color: #4A90E2;
        background-color: white;
        padding: 10px;
        font-size: 16px;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #4A90E2;
        color: white;
    }
    .section-title {
        color: #4A90E2;
        font-size: 18px;
        margin-top: 15px;
        margin-bottom: 10px;
        font-weight: bold;
    }
    .dataset-preview {
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 10px;
        margin-top: 20px;
        background-color: #f9f9f9;
    }
    h1 {
        color: #4A90E2;
    }
    h2 {
        color: #4A90E2;
    }
    h3 {
        color: #333;
    }
    h4 {
        color: #555;
    }
    .st-emotion-cache-1jmvejs { # Targeting expander header for slightly different styling
        background-color: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)



st.markdown("""
    <div style="position: fixed; bottom: 0; left: 0; width: 100%; text-align: center; background-color: ; padding: 10px;">
        <p style="font-size: 12px;">Made with ❤️ by <a href = "https://avarshvir.github.io/arshvir">Arshvir</a> and <a href = "https://jaiho-labs.onrender.com">Jaiho Labs</a></p>
    </div>
""", unsafe_allow_html=True)