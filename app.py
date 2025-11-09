import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Set page config
st.set_page_config(page_title="Flow Regime Visualizer", layout="wide")

# App title
st.title("Flow Regime Visualizer 🌊")

# Create tabs
tabs = st.tabs(["How to Use", "Classify Flow Regime", "Visualization"])

# -------------------------------
# Tab 1: How to Use
# -------------------------------
with tabs[0]:
    st.header("How to Use This Application")
    st.markdown("""
    1. Go to the **Classify Flow Regime** tab to upload your data and classify the flow regime.
    2. Go to the **Visualization** tab to see visual representations of the flow.
    3. Follow on-screen instructions and explore different datasets.
    """)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f3/Water_flowing_in_a_river.jpg/320px-Water_flowing_in_a_river.jpg", caption="Flow Visualization Example")

# -------------------------------
# Tab 2: Classify Flow Regime
# -------------------------------
with tabs[1]:
    st.header("Classify Flow Regime")
    
    uploaded_file = st.file_uploader("Upload your data file (CSV)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(df.head())
        
        # Placeholder classification
        if st.button("Classify Flow Regime"):
            # Here you would call your ML model
            st.success("Flow regime classified as: **Slug Flow**")  # Example output

# -------------------------------
# Tab 3: Visualization
# -------------------------------
with tabs[2]:
    st.header("Visualization of Flow")
    
    st.markdown("Example: Random pressure signal plot")
    
    # Example random data
    x = np.linspace(0, 10, 500)
    y = np.sin(x) + 0.1 * np.random.randn(500)
    
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pressure (Pa)")
    ax.set_title("Pressure Signal")
    
    st.pyplot(fig)
