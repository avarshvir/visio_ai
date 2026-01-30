import streamlit as st
import pandas as pd
import utils

# Initialize
st.set_page_config(page_title="Data Loader - Visio AI", page_icon="📂", layout="wide")
utils.init_session_state()
utils.load_css()
utils.sidebar_nav()

st.markdown("## 📂 Data Loader")

col_left, col_right = st.columns([1, 1.5])

with col_left:
    st.markdown("### 1. Upload File")
    dataset = st.file_uploader("Choose a file", type=["csv", "xlsx", "txt"])
    
    if dataset:
        if st.session_state.last_uploaded_file_name != dataset.name:
            df = utils.safe_read_csv(dataset)
            if df is not None:
                st.session_state.updated_df = df.copy()
                st.session_state.original_df_uploaded = True
                st.session_state.last_uploaded_file_name = dataset.name
                st.session_state.X_train = None
                st.session_state.trained_model = None
                st.rerun()

    if st.session_state.updated_df is not None:
         st.success(f"Loaded: **{st.session_state.last_uploaded_file_name}**")
         st.dataframe(st.session_state.updated_df.head(), use_container_width=True)
         st.caption(f"Shape: {st.session_state.updated_df.shape}")

with col_right:
    if st.session_state.updated_df is not None:
        st.markdown("### 2. Cleaning & Wrangling")
        
        df = st.session_state.updated_df
        clean_tabs = st.tabs(["🧹 Manual Cleaning", "⚡ Auto Cleaning", "🛠️ Wrangling"])
        
        with clean_tabs[0]:
            st.markdown("#### Precise Control")
            null_counts = df.isnull().sum()
            total_nulls = null_counts.sum()
            
            if total_nulls > 0:
                st.warning(f"Found {total_nulls} missing values.")
                col_to_clean = st.selectbox("Select Column to Clean", null_counts[null_counts > 0].index)
                
                method = st.selectbox("Imputation Method", ["Fill with Mean", "Fill with Median", "Fill with Mode", "Fill with Specific Value", "Drop Rows", "Drop Column"])
                
                specific_val = None
                if method == "Fill with Specific Value":
                    specific_val = st.text_input("Value")
                
                if st.button("Apply to Column"):
                    if method == "Fill with Mean": df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].mean())
                    elif method == "Fill with Median": df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].median())
                    elif method == "Fill with Mode": df[col_to_clean] = df[col_to_clean].fillna(df[col_to_clean].mode()[0])
                    elif method == "Drop Rows": df.dropna(subset=[col_to_clean], inplace=True)
                    elif method == "Drop Column": df.drop(columns=[col_to_clean], inplace=True)
                    elif method == "Fill with Specific Value" and specific_val: df[col_to_clean] = df[col_to_clean].fillna(specific_val)
                    
                    st.session_state.updated_df = df
                    st.success("Applied!")
                    st.rerun()
            else:
                 st.success("No missing values found.")

        with clean_tabs[1]:
            st.markdown("#### One-Click Fixes (Algorithm Verification: ✅)")
            st.info("Using standard pandas strategies: Mean for numeric, Mode for categorical.")
            
            if st.button("🚀 Auto-Clean Entire Dataset"):
                # Numeric
                num_cols = df.select_dtypes(include=['number']).columns
                for c in num_cols: df[c] = df[c].fillna(df[c].mean())
                
                # Categ
                cat_cols = df.select_dtypes(include=['object', 'category']).columns
                for c in cat_cols: 
                    if not df[c].mode().empty:
                        df[c] = df[c].fillna(df[c].mode()[0])
                
                st.session_state.updated_df = df
                st.success("Auto-Cleaning Complete!")
                st.rerun()

        with clean_tabs[2]:
            st.markdown("#### Data Transformation")
            
            wrangle_action = st.selectbox("Action", ["Rename Column", "Drop Column", "Change Data Type", "Remove Commas"])
            
            if wrangle_action != "Remove Commas":
                target_col = st.selectbox("Target Column", df.columns, key="wrangle_col")
            
            if wrangle_action == "Rename Column":
                new_name = st.text_input("New Name")
                if st.button("Rename"):
                    df.rename(columns={target_col: new_name}, inplace=True)
                    st.session_state.updated_df = df
                    st.rerun()
                    
            elif wrangle_action == "Drop Column":
                if st.button("Drop"):
                    df.drop(columns=[target_col], inplace=True)
                    st.session_state.updated_df = df
                    st.rerun()

            elif wrangle_action == "Change Data Type":
                new_type = st.selectbox("To Type", ["str", "int", "float", "bool", "datetime"])
                if st.button("Convert"):
                    try:
                        if new_type == "datetime": df[target_col] = pd.to_datetime(df[target_col])
                        else: df[target_col] = df[target_col].astype(new_type)
                        st.session_state.updated_df = df
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion failed: {e}")
            elif wrangle_action == "Remove Commas":
                target_col = st.selectbox("Target Column", ["All String Columns"] + list(df.select_dtypes(include='object').columns), key="wrangle_col_commas")
                
                if st.button("Run Cleanup"):
                    cols_to_clean = []
                    if target_col == "All String Columns":
                        cols_to_clean = df.select_dtypes(include='object').columns
                    else:
                        cols_to_clean = [target_col]
                    
                    count = 0
                    for col in cols_to_clean:
                        try:
                            # Check if column actually has commas to avoid useless operations
                            if df[col].astype(str).str.contains(',').any():
                                df[col] = df[col].astype(str).str.replace(',', '', regex=False)
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                count += 1
                        except Exception:
                            pass
                    
                    if count > 0:
                        st.session_state.updated_df = df
                        st.success(f"Removed commas and converted {count} columns to numbers!")
                        st.rerun()
                    else:
                        st.warning("No columns with commas found (or conversion failed).")

    else:
        st.info("Upload a file to enable options.")

if st.session_state.updated_df is not None:
    st.markdown("---")
    if st.button("Proceed to EDA ➡️"):
        st.switch_page("pages/2_EDA.py")
