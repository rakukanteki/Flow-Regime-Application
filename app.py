import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# Page config
st.set_page_config(page_title="Flow Regime Visualizer", layout="wide", page_icon="🌊")

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<p class="main-header">Flow Regime Visualizer 🌊</p>', unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📖 How to Use", "🔬 Classify Flow Regime", "📊 Visualization"])

# -------------------------------
# Tab 1: How to Use
# -------------------------------
with tab1:
    st.header("How to Use This App")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Welcome to the Flow Regime Classification Tool!
        
        This application helps you classify and visualize multiphase flow regimes.
        
        #### Steps to Get Started:
        
        1. **Upload Your Data** 
           - Go to the **Classify Flow Regime** tab
           - Upload a CSV file containing your flow data
           - Required columns: velocity, density, viscosity, etc.
        
        2. **Classify Flow Regime**
           - Click the "Classify Flow" button
           - The system will analyze your data and predict the flow regime
        
        3. **Visualize Results**
           - Navigate to the **Visualization** tab
           - Explore interactive charts and flow maps
           - Analyze flow patterns and distributions
        
        #### Common Flow Regimes:
        - 🔵 **Bubble Flow**: Small bubbles dispersed in liquid
        - 🟢 **Slug Flow**: Alternating liquid slugs and gas pockets
        - 🟡 **Churn Flow**: Chaotic, oscillatory flow
        - 🔴 **Annular Flow**: Liquid film on pipe wall with gas core
        - 🟣 **Stratified Flow**: Separated liquid and gas layers
        """)
    
    with col2:
        st.info("💡 **Tip**: Make sure your CSV file has proper headers and numerical data for accurate classification.")
        
        st.success("✅ **Supported Formats**: CSV files with flow parameters")
        
        st.warning("⚠️ **Note**: This is a demonstration UI. Connect your ML model for real predictions.")

# -------------------------------
# Tab 2: Classify Flow Regime
# -------------------------------
with tab2:
    st.header("Classify Flow Regime")
    
    # File uploader
    uploaded_file = st.file_uploader("📁 Upload CSV data", type=["csv"], help="Upload your flow data in CSV format")
    
    if uploaded_file:
        # Read the data
        df = pd.read_csv(uploaded_file)
        
        # Display preview
        st.subheader("📋 Preview of Uploaded Data")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Data info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", len(df))
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
        
        st.divider()
        
        # Column selection for analysis
        st.subheader("🎯 Select Features for Classification")
        
        col1, col2 = st.columns(2)
        with col1:
            velocity_col = st.selectbox("Select Velocity Column", df.columns, help="Choose the column representing velocity")
        with col2:
            density_col = st.selectbox("Select Density Column", df.columns, help="Choose the column representing density")
        
        st.divider()
        
        # Classification button
        if st.button("🚀 Classify Flow Regime", type="primary", use_container_width=True):
            with st.spinner("Analyzing flow patterns..."):
                # Placeholder for ML model prediction
                # In real app, you would call your ML model here
                
                # Simulate processing
                import time
                time.sleep(1)
                
                # Example classification result
                flow_regimes = ["Bubble Flow", "Slug Flow", "Churn Flow", "Annular Flow", "Stratified Flow"]
                predicted_regime = np.random.choice(flow_regimes)
                confidence = np.random.uniform(0.75, 0.99)
                
                # Display results
                st.success("✅ Classification Complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("### Predicted Flow Regime")
                    st.markdown(f"## **{predicted_regime}**")
                with col2:
                    st.markdown("### Confidence Score")
                    st.markdown(f"## **{confidence:.2%}**")
                    st.progress(confidence)
                with col3:
                    st.markdown("### Status")
                    st.markdown("## ✅ **Success**")
                
                # Detailed results
                st.divider()
                st.subheader("📊 Classification Probabilities")
                
                # Generate sample probabilities
                probs = np.random.dirichlet(np.ones(5), size=1)[0]
                prob_df = pd.DataFrame({
                    'Flow Regime': flow_regimes,
                    'Probability': probs
                }).sort_values('Probability', ascending=False)
                
                # Create bar chart
                fig = px.bar(prob_df, x='Flow Regime', y='Probability', 
                            color='Probability',
                            color_continuous_scale='Viridis',
                            title='Probability Distribution Across Flow Regimes')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Add download button for results
                st.divider()
                result_df = df.copy()
                result_df['Predicted_Flow_Regime'] = predicted_regime
                result_df['Confidence'] = confidence
                
                csv = result_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name="flow_regime_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
    
    else:
        # Show sample template
        st.info("👆 Please upload a CSV file to begin classification")
        
        st.subheader("📄 Sample Data Format")
        sample_data = pd.DataFrame({
            'velocity': [1.2, 2.5, 3.8, 5.1],
            'density': [850, 920, 780, 860],
            'viscosity': [0.05, 0.08, 0.06, 0.07],
            'pressure': [101.3, 102.5, 100.8, 103.2]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        # Download sample template
        sample_csv = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Sample Template",
            data=sample_csv,
            file_name="sample_flow_data.csv",
            mime="text/csv"
        )

# -------------------------------
# Tab 3: Visualization
# -------------------------------
with tab3:
    st.header("Flow Regime Visualization")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        
        # Visualization options
        viz_option = st.selectbox(
            "Select Visualization Type",
            ["Flow Map", "Scatter Plot", "Distribution Plot", "Time Series", "Correlation Heatmap"]
        )
        
        if viz_option == "Flow Map":
            st.subheader("🗺️ Flow Regime Map")
            
            # Generate sample flow map data
            x = np.linspace(0, 10, 100)
            y = np.linspace(0, 10, 100)
            X, Y = np.meshgrid(x, y)
            Z = np.sin(X) * np.cos(Y)
            
            fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='Jet'))
            fig.update_layout(
                title="Flow Regime Map",
                xaxis_title="Superficial Gas Velocity",
                yaxis_title="Superficial Liquid Velocity",
                height=600
            )
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Scatter Plot":
            st.subheader("📍 Scatter Plot Analysis")
            
            col1, col2 = st.columns(2)
            with col1:
                x_axis = st.selectbox("X-axis", df.columns, key="scatter_x")
            with col2:
                y_axis = st.selectbox("Y-axis", df.columns, key="scatter_y")
            
            fig = px.scatter(df, x=x_axis, y=y_axis, 
                           title=f"{x_axis} vs {y_axis}",
                           color=df.columns[0] if len(df.columns) > 2 else None)
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Distribution Plot":
            st.subheader("📊 Distribution Analysis")
            
            selected_col = st.selectbox("Select Column", df.columns)
            
            col1, col2 = st.columns(2)
            with col1:
                fig_hist = px.histogram(df, x=selected_col, 
                                       title=f"Histogram of {selected_col}",
                                       nbins=30)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                fig_box = px.box(df, y=selected_col, 
                                title=f"Box Plot of {selected_col}")
                st.plotly_chart(fig_box, use_container_width=True)
        
        elif viz_option == "Time Series":
            st.subheader("📈 Time Series Analysis")
            
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_cols = st.multiselect("Select Variables", numeric_cols, default=numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols)
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Scatter(y=df[col], mode='lines', name=col))
                
                fig.update_layout(
                    title="Time Series Plot",
                    xaxis_title="Index",
                    yaxis_title="Value",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        
        elif viz_option == "Correlation Heatmap":
            st.subheader("🔥 Correlation Heatmap")
            
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig = px.imshow(corr, 
                          text_auto=True,
                          aspect="auto",
                          color_continuous_scale='RdBu',
                          title="Feature Correlation Matrix")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.info("👆 Please upload data in the 'Classify Flow Regime' tab first")
        
        # Show placeholder visualization
        st.subheader("Sample Flow Regime Map")
        
        x = np.linspace(0, 10, 100)
        y = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(x, y)
        Z = np.sin(X) * np.cos(Y)
        
        fig = go.Figure(data=go.Contour(z=Z, x=x, y=y, colorscale='Jet'))
        fig.update_layout(
            title="Example: Flow Regime Map",
            xaxis_title="Superficial Gas Velocity (m/s)",
            yaxis_title="Superficial Liquid Velocity (m/s)",
            height=500
        )
        st.plotly_chart(fig, use_container_width=True)

# Sidebar with additional info
with st.sidebar:
    st.header("About")
    st.markdown("""
    **Flow Regime Classifier**
    
    This tool helps classify multiphase flow regimes based on input parameters.
    
    **Version**: 1.0  
    **Framework**: Streamlit
    """)
    
    st.divider()
    
    st.header("Settings")
    theme = st.selectbox("Theme", ["Light", "Dark"])
    show_raw_data = st.checkbox("Show Raw Data", value=True)
    
    st.divider()
    
    st.markdown("Made with ❤️ for Flow Analysis")