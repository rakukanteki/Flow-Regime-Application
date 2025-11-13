import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torch.nn as nn
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, RobustScaler
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="Flow Regime Visual Twin",
    layout="wide"
)

# Model configuration
WINDOW_SIZE = 40
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CLASS_NAMES = ["Dispersed Flow", "Plug Flow", "Slug Flow"]

# --- Model Definition ---
class MultiTaskPINN(nn.Module):
    def __init__(self, input_size=40, hidden_size=128, num_classes=3):
        super(MultiTaskPINN, self).__init__()
        
        self.conv1 = nn.Conv1d(1, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.4)
        
        self.fc_features = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.fc_velocities = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        conv_output_size = 64 * (WINDOW_SIZE // 2)
        combined_size = conv_output_size + 64 + 32
        
        self.shared_layer = nn.Sequential(
            nn.Linear(combined_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.4)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size // 2, num_classes)
        )
        
        self.velocity_regressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        
        self.physics_net = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, pressure_window, features, velocities):
        x = pressure_window.unsqueeze(1)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        feat = self.fc_features(features)
        vel = self.fc_velocities(velocities)
        
        combined = torch.cat([x, feat, vel], dim=1)
        shared_repr = self.shared_layer(combined)
        
        class_output = self.classifier(shared_repr)
        velocity_output = self.velocity_regressor(shared_repr)
        physics_output = self.physics_net(shared_repr)
        
        return class_output, velocity_output, physics_output

# --- Helper Functions ---
@st.cache_resource
def load_model_and_scalers():
    """Load the trained model and scalers"""
    try:
        # Load model
        model_path = "models/best_multitask_pinn_fold_5.pth"
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        model = MultiTaskPINN(input_size=WINDOW_SIZE, hidden_size=128, num_classes=3)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        
        # Try to load scalers from checkpoint first
        scalers = checkpoint.get('scalers', None)
        
        # If not in checkpoint, try loading from pickle file
        if scalers is None:
            try:
                scalers_path = "training_scalers.pkl"
                with open(scalers_path, 'rb') as f:
                    scalers = pickle.load(f)
                st.success("✅ Scalers loaded from training_scalers.pkl")
            except FileNotFoundError:
                st.error("❌ Scalers file not found at: training_scalers.pkl")
                return model, None
            except Exception as e:
                st.error(f"❌ Error loading scalers: {str(e)}")
                return model, None
        
        # Validate that all required scalers are present
        required_scalers = ['scaler_pressure', 'scaler_features', 'scaler_vsg', 'scaler_vsl']
        missing_scalers = [s for s in required_scalers if s not in scalers]
        
        if missing_scalers:
            st.warning(f"⚠️ Missing scalers: {', '.join(missing_scalers)}")
            st.info(f"Available scalers: {list(scalers.keys())}")
            return model, None
        
        return model, scalers
    except FileNotFoundError:
        st.error(f"❌ Model file not found at: models/best_multitask_pinn_fold_5.pth")
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.exception(e)
        return None, None

def extract_features(pressure_window):
    """Extract features from pressure window"""
    features = []
    features.append(np.mean(pressure_window))
    features.append(np.std(pressure_window))
    features.append(np.max(pressure_window) - np.min(pressure_window))
    
    gradient = np.gradient(pressure_window)
    features.append(np.mean(gradient))
    features.append(np.std(gradient))
    features.append(np.max(np.abs(gradient)))
    
    if len(pressure_window) > 4:
        freqs = fftfreq(len(pressure_window), d=0.05)
        fft_vals = np.abs(fft(pressure_window))
        positive_freqs = freqs[:len(freqs)//2]
        positive_fft = fft_vals[:len(fft_vals)//2]
        if len(positive_fft) > 0:
            dominant_freq_idx = np.argmax(positive_fft)
            features.append(positive_freqs[dominant_freq_idx])
            features.append(positive_fft[dominant_freq_idx])
        else:
            features.append(0.0)
            features.append(0.0)
    else:
        features.append(0.0)
        features.append(0.0)
    
    return np.array(features)

def predict_flow_regime(model, scalers, pressure_data, vsg=0.5, vsl=0.5):
    """Make prediction using the trained model"""
    try:
        # Check if scalers are available
        if scalers is None:
            st.error("❌ Scalers not available. The model checkpoint may not contain scalers.")
            st.info("💡 Please retrain the model and save scalers in the checkpoint, or provide them separately.")
            return None
        
        stride = 20
        windows = []
        for i in range(0, len(pressure_data) - WINDOW_SIZE + 1, stride):
            window = pressure_data[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)
        
        if len(windows) == 0:
            st.warning(f"⚠️ Not enough data points. Need at least {WINDOW_SIZE} points, but got {len(pressure_data)}.")
            return None
        
        features_list = [extract_features(w) for w in windows]
        
        pressure_windows = np.array(windows)
        features_array = np.array(features_list)
        
        # Apply scaling with error handling
        try:
            pressure_scaled = scalers['scaler_pressure'].transform(pressure_windows)
            features_scaled = scalers['scaler_features'].transform(features_array)
            vsg_scaled = scalers['scaler_vsg'].transform([[vsg]] * len(windows))
            vsl_scaled = scalers['scaler_vsl'].transform([[vsl]] * len(windows))
            velocities_scaled = np.hstack([vsg_scaled, vsl_scaled])
        except KeyError as e:
            st.error(f"❌ Missing scaler: {e}")
            st.info(f"Available scalers: {list(scalers.keys())}")
            return None
        except Exception as e:
            st.error(f"❌ Error during scaling: {str(e)}")
            return None
        
        pressure_tensor = torch.FloatTensor(pressure_scaled).to(DEVICE)
        features_tensor = torch.FloatTensor(features_scaled).to(DEVICE)
        velocities_tensor = torch.FloatTensor(velocities_scaled).to(DEVICE)
        
        with torch.no_grad():
            class_output, velocity_output, _ = model(pressure_tensor, features_tensor, velocities_tensor)
            
            probabilities = torch.softmax(class_output, dim=1)
            avg_probabilities = probabilities.mean(dim=0).cpu().numpy()
            
            predicted_class = torch.argmax(probabilities.mean(dim=0)).item()
            confidence = avg_probabilities[predicted_class] * 100
            
            velocity_pred = velocity_output.mean(dim=0).cpu().numpy()
            vsg_pred = scalers['scaler_vsg'].inverse_transform([[velocity_pred[0]]])[0][0]
            vsl_pred = scalers['scaler_vsl'].inverse_transform([[velocity_pred[1]]])[0][0]
        
        return {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'probabilities': avg_probabilities,
            'vsg_predicted': vsg_pred,
            'vsl_predicted': vsl_pred,
            'num_windows': len(windows)
        }
    
    except Exception as e:
        st.error(f"❌ Prediction error: {str(e)}")
        st.exception(e)
        return None

# --- Initialize session state ---
if "page" not in st.session_state:
    st.session_state["page"] = "Home"

# --- Custom CSS ---
st.markdown("""
<style>
    [data-testid="stSidebar"] {
        background-color: #1e2130;
    }
    .sidebar-button {
        background-color: #262c3d;
        color: #ffffff;
        padding: 14px 20px;
        border-radius: 8px;
        border: 1px solid #3d4463;
        margin-bottom: 10px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        width: 100%;
        text-align: left;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
    }
    .sidebar-button:hover {
        background-color: #2d3348;
        border-color: #4d5578;
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.4);
    }
    [data-testid="stSidebar"] h1 {
        color: #ffffff;
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6366f1;
    }
    .stMarkdown, p, li { color: #e0e0e0; }
    h1, h2, h3 { color: #ffffff; }
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 30px;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
        margin: 20px 0;
        color: white;
    }
    .card-title {
        font-size: 32px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    .card-subtitle {
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
        opacity: 0.95;
    }
    .card-footer {
        font-size: 14px;
        text-align: center;
        margin-top: 15px;
        padding-top: 15px;
        border-top: 1px solid rgba(255, 255, 255, 0.3);
        font-style: italic;
    }
    .justified-text {
        text-align: justify;
        line-height: 1.8;
        margin-bottom: 20px;
    }
    .justified-text ul {
        text-align: justify;
        line-height: 1.8;
    }
    .justified-text li {
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar Navigation ---
with st.sidebar:
    st.title("Navigation")

    if st.button("Home", use_container_width=True):
        st.session_state["page"] = "Home"
    if st.button("Evaluate Model", use_container_width=True):
        st.session_state["page"] = "Evaluate Model"
    if st.button("User Guideline", use_container_width=True):
        st.session_state["page"] = "User Guideline"
    if st.button("Privacy and Policy", use_container_width=True):
        st.session_state["page"] = "Privacy and Policy"

# --- Page Selection ---
page = st.session_state["page"]

# ------------------------------------------------------------------------
# 🏠 HOME PAGE
# ------------------------------------------------------------------------
if page == "Home":
    st.title("MTPINN for Multiphase Flow Regime")
    
    st.markdown("""
    <div class="info-card">
        <div class="card-title">MTPINN</div>
        <div class="card-subtitle">
            Multi-Task Physics-Informed Neural Network for<br>
            Advanced Flow Regime Classification
        </div>
        <div class="card-footer">
            This research is led by Dr. Amith Khandakar and Dr. Mohammad Azizur Rahman
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("About Multiphase Flow Regime")
    st.markdown("""
    <div class="justified-text">
    Multiphase flow refers to the simultaneous flow of materials with different phases (gas, liquid, and/or solid)
    within pipelines or process systems. Understanding the flow regime (such as bubbly, slug, annular, or dispersed flows)
    is critical because it affects pressure drop, heat transfer, and mass transport efficiency.
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Impact of Flow Regimes in Multiphase Systems")
    st.markdown("""
    <div class="justified-text">
    Different flow regimes have a significant impact on system performance and safety:
    <ul>
    <li><strong>Pressure Drop:</strong> Certain regimes (like slug flow) can cause large fluctuations in pressure.</li>
    <li><strong>Separation Efficiency:</strong> Flow regime affects the performance of separators.</li>
    <li><strong>Equipment Design:</strong> Correct prediction of flow regime is essential for pumps, pipelines, and reactors.</li>
    <li><strong>Operational Safety:</strong> Unstable flow regimes can lead to erosion, vibration, and operational hazards.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("assets/flow-regime.png", caption="Common Multiphase Flow Regimes", use_container_width=True)

    st.subheader("How the MTPINN Works")
    st.markdown("""
    <div class="justified-text">
    The Multi-Task Physics-Informed Neural Network (MTPINN) combines deep learning with physics-based constraints 
    to accurately classify multiphase flow regimes. The system processes experimental video data through a sophisticated 
    pipeline that extracts temporal and spatial features, integrates physical laws of fluid dynamics, and performs 
    multi-task learning to simultaneously predict flow patterns and estimate key flow parameters.
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([0.5, 3, 0.5])
    with col2:
        st.image("assets/Methodoloy-Fig.svg", caption="MTPINN Methodology and Architecture", use_container_width=True)

# ------------------------------------------------------------------------
# 📊 EVALUATE MODEL PAGE
# ------------------------------------------------------------------------
elif page == "Evaluate Model":
    st.title("Evaluate the model")
    st.markdown("""
    Upload a **CSV or Excel file** containing flow measurement data with a **'Pressure (barA)'** column.  
    The system will visualize the data and predict the **flow regime** using the trained MTPINN model.
    """)

    # Load model
    model, scalers = load_model_and_scalers()
    
    if model is None:
        st.error("❌ Failed to load model. Please check if the model file exists at 'models/best_multitask_pinn_fold_5.pth'")
    else:
        uploaded_file = st.file_uploader("📂 Upload your dataset", type=["csv", "xlsx"])

        if uploaded_file is not None:
            # --- Load Data ---
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.success("✅ File uploaded successfully!")
            
            # Show data preview
            with st.expander("👀 View Data Preview"):
                st.dataframe(df.head(10), use_container_width=False)
                st.caption(f"**Dataset shape:** {df.shape[0]} rows × {df.shape[1]} columns")

            # Check for pressure column
            pressure_col = None
            for col in df.columns:
                if 'pressure' in col.lower():
                    pressure_col = col
                    break
            
            if pressure_col is None:
                st.error("❌ No 'Pressure' column found in the dataset. Please ensure your file contains a pressure column.")
            else:
                st.info(f"✓ Using column: **{pressure_col}**")

                # --- Plot pressure signal ---
                numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

                if len(numeric_cols) >= 1:
                    st.subheader("📈 Data Visualization")

                    # Identify time column
                    time_col = None
                    time_keywords = ['time', 'timestamp', 'date', 't', 'sec', 'second']
                    for col in df.columns:
                        if any(keyword in col.lower() for keyword in time_keywords):
                            time_col = col
                            break
                    
                    # Plot pressure signal
                    st.markdown("**Pressure Signal:**")
                    fig, ax = plt.subplots(figsize=(10, 4))
                    
                    if time_col and time_col in df.columns:
                        ax.plot(df[time_col], df[pressure_col], linewidth=1.5, color='#667eea')
                        ax.set_xlabel(time_col, fontsize=10)
                    else:
                        ax.plot(df.index, df[pressure_col], linewidth=1.5, color='#667eea')
                        ax.set_xlabel("Sample Index", fontsize=10)
                    
                    ax.set_title("Pressure Signal Over Time", fontsize=12)
                    ax.set_ylabel("Pressure (barA)", fontsize=10)
                    ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig)

                    # Correlation heatmap if multiple numeric columns
                    plot_cols = [col for col in numeric_cols if col != time_col]
                    if len(plot_cols) >= 2:
                        st.markdown("**Correlation Between Features:**")
                        corr = df[plot_cols[:5]].corr()
                        fig2, ax2 = plt.subplots(figsize=(6, 5))
                        cax = ax2.matshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
                        fig2.colorbar(cax)
                        ax2.set_xticks(range(len(corr.columns)))
                        ax2.set_yticks(range(len(corr.columns)))
                        ax2.set_xticklabels(corr.columns, rotation=45, ha="left", fontsize=9)
                        ax2.set_yticklabels(corr.columns, fontsize=9)
                        ax2.set_title("Feature Correlation Heatmap", pad=20, fontsize=12)
                        
                        for i in range(len(corr.columns)):
                            for j in range(len(corr.columns)):
                                ax2.text(j, i, f'{corr.iloc[i, j]:.2f}', 
                                        ha='center', va='center', color='black', fontsize=8)
                        
                        plt.tight_layout()
                        st.pyplot(fig2)

                # --- Velocity Input ---
                st.subheader("⚙️ Flow Parameters")
                col1, col2 = st.columns(2)
                with col1:
                    vsg_input = st.number_input("Superficial Gas Velocity (Vsg) [m/s]", 
                                                min_value=0.0, max_value=10.0, value=0.5, step=0.1)
                with col2:
                    vsl_input = st.number_input("Superficial Liquid Velocity (Vsl) [m/s]", 
                                                min_value=0.0, max_value=10.0, value=0.5, step=0.1)

                # --- Model Prediction ---
                if st.button("🔍 Predict Flow Regime", use_container_width=False):
                    with st.spinner("Running prediction..."):
                        pressure_data = df[pressure_col].values
                        
                        result = predict_flow_regime(model, scalers, pressure_data, vsg_input, vsl_input)
                        
                        if result:
                            st.subheader("🧠 Prediction Results")
                            
                            predicted_class_name = CLASS_NAMES[result['predicted_class']]
                            
                            # Display prediction metrics
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Predicted Flow Regime", predicted_class_name)
                            with col2:
                                st.metric("Windows Analyzed", result['num_windows'])
                            
                            # Display velocity predictions
                            st.markdown("**Predicted Velocities:**")
                            vel_col1, vel_col2 = st.columns(2)
                            with vel_col1:
                                st.metric("Vsg (Predicted)", f"{result['vsg_predicted']:.3f} m/s", 
                                         delta=f"{result['vsg_predicted']-vsg_input:.3f}")
                            with vel_col2:
                                st.metric("Vsl (Predicted)", f"{result['vsl_predicted']:.3f} m/s",
                                         delta=f"{result['vsl_predicted']-vsl_input:.3f}")
                            
                            # Display flow regime GIF/image
                            gif_mapping = {
                                "Dispersed Flow": "video_library/Dispersed-Flow/Dispersed-Flow.gif",
                                "Slug Flow": "video_library/Slug-Flow/Slug-Flow.gif",
                                "Plug Flow": "video_library/Plug-Flow/Plug-Flow.gif"
                            }
                            
                            image_mapping = {
                                "Dispersed Flow": "video_library/Dispersed-Flow/Dispersed-Flow.png",
                                "Slug Flow": "video_library/Slug-Flow/Slug-Flow.png",
                                "Plug Flow": "video_library/Plug-Flow/Plug-Flow.png"
                            }
                            
                            # Try GIF first, then fallback to PNG
                            gif_path = gif_mapping.get(predicted_class_name)
                            image_path = image_mapping.get(predicted_class_name)
                            
                            media_path = None
                            if gif_path and os.path.exists(gif_path):
                                media_path = gif_path
                            elif image_path and os.path.exists(image_path):
                                media_path = image_path
                            
                            if media_path:
                                st.markdown(f"**{predicted_class_name} Visualization:**")
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(media_path, caption=f"{predicted_class_name} Animation", 
                                           use_container_width=True)
                            else:
                                st.warning(f"⚠️ Visualization not found for {predicted_class_name}")

        else:
            st.info("👆 Please upload a file to begin visualization and prediction.")

# ------------------------------------------------------------------------
# 📋 USER GUIDELINE PAGE
# ------------------------------------------------------------------------
elif page == "User Guideline":
    st.title("User Guideline")
    st.markdown("""
    <div class="justified-text">
    <h3>How to Use the Flow Regime Classification System</h3>
    
    <h4>Step 1: Prepare Your Data</h4>
    Ensure your dataset meets the following requirements:
    <ul>
    <li>File format: CSV or Excel (.xlsx)</li>
    <li>Must contain a column with "Pressure" in its name (e.g., "Pressure (barA)", "Pressure_Data")</li>
    <li>Minimum of 40 data points for analysis</li>
    <li>Pressure values should be in barA (absolute pressure)</li>
    </ul>
    
    <h4>Step 2: Navigate to Evaluate Model</h4>
    Click on the <strong>"Evaluate Model"</strong> button in the sidebar to access the prediction interface.
    
    <h4>Step 3: Upload Your Dataset</h4>
    <ul>
    <li>Click the "Browse files" button</li>
    <li>Select your CSV or Excel file containing pressure measurements</li>
    <li>The system will automatically detect and validate the pressure column</li>
    <li>Preview your data to ensure it loaded correctly</li>
    </ul>
    
    <h4>Step 4: Enter Flow Parameters</h4>
    Input the operating conditions for your multiphase flow system:
    <ul>
    <li><strong>Superficial Gas Velocity (Vsg):</strong> Enter the gas phase velocity in m/s (range: 0-10 m/s)</li>
    <li><strong>Superficial Liquid Velocity (Vsl):</strong> Enter the liquid phase velocity in m/s (range: 0-10 m/s)</li>
    </ul>
    
    <h4>Step 5: Run Prediction</h4>
    <ul>
    <li>Click the <strong>"Predict Flow Regime"</strong> button</li>
    <li>The system will analyze your pressure data using sliding windows</li>
    <li>Processing typically takes a few seconds depending on data size</li>
    </ul>
    
    <h4>Step 6: Interpret Results</h4>
    The prediction results include:
    <ul>
    <li><strong>Predicted Flow Regime:</strong> The most likely flow pattern (Dispersed, Plug, or Slug Flow)</li>
    <li><strong>Confidence Score:</strong> Model confidence in the prediction (0-100%)</li>
    <li><strong>Class Probabilities:</strong> Probability distribution across all flow regime classes</li>
    <li><strong>Predicted Velocities:</strong> Model's estimate of Vsg and Vsl based on pressure patterns</li>
    <li><strong>Flow Visualization:</strong> Representative image of the predicted flow regime</li>
    </ul>
    
    <h4>Understanding Flow Regimes</h4>
    <ul>
    <li><strong>Dispersed Flow:</strong> Small gas bubbles uniformly distributed in continuous liquid phase</li>
    <li><strong>Plug Flow:</strong> Large elongated gas bubbles (plugs) separated by liquid slugs</li>
    <li><strong>Slug Flow:</strong> Intermittent flow with alternating gas pockets and liquid slugs</li>
    </ul>
    
    <h4>Tips for Best Results</h4>
    <ul>
    <li>Use data collected at steady-state conditions for more accurate predictions</li>
    <li>Provide accurate velocity values that match your experimental conditions</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)


# ------------------------------------------------------------------------
# 🔒 PRIVACY AND POLICY PAGE
# ------------------------------------------------------------------------
elif page == "Privacy and Policy":
    st.title("Privacy and Policy")
    st.markdown("""
    <div class="justified-text">
    
    <h3>Data Privacy and Security</h3>
    
    <h4>Data Collection and Usage</h4>
    The MTPINN Flow Regime Classification System is designed with privacy and security in mind. We are committed to protecting your data and ensuring transparency in how it is processed.
    
    <h4>What Data We Process</h4>
    <ul>
    <li><strong>Uploaded Files:</strong> CSV and Excel files containing pressure measurement data and flow parameters</li>
    <li><strong>Input Parameters:</strong> Superficial gas velocity (Vsg) and superficial liquid velocity (Vsl) values</li>
    <li><strong>Session Data:</strong> Temporary navigation state stored locally in your browser</li>
    </ul>
    
    <h4>How We Handle Your Data</h4>
    <ul>
    <li><strong>Local Processing:</strong> All uploaded data is processed locally in real-time and is not stored on any server</li>
    <li><strong>No Data Storage:</strong> We do not permanently store, save, or retain any uploaded files or user data</li>
    <li><strong>Session-Based:</strong> Data exists only during your active session and is automatically discarded when you close the application</li>
    <li><strong>No Third-Party Sharing:</strong> Your data is never shared with, sold to, or transferred to any third parties</li>
    <li><strong>No User Tracking:</strong> We do not track user behavior, collect personal information, or use cookies for analytics</li>
    </ul>
    
    <h4>Model and Algorithm</h4>
    <ul>
    <li>The classification model is a Multi-Task Physics-Informed Neural Network (MTPINN)</li>
    <li>The model processes pressure signals, extracts features, and predicts flow regimes based on learned patterns</li>
    <li>Predictions are made using pre-trained model weights and do not require external API calls</li>
    <li>All computations are performed locally on the server hosting this application</li>
    </ul>
    
    <h4>Research and Academic Use</h4>
    This application is developed for research and educational purposes at Qatar University under the leadership of:
    <ul>
    <li>Dr. Amith Khandakar</li>
    <li>Dr. Mohammad Azizur Rahman</li>
    </ul>
    
    The MTPINN model integrates:
    <ul>
    <li>Deep learning for pattern recognition in pressure signals</li>
    <li>Physics-informed constraints based on fluid dynamics principles</li>
    <li>Multi-task learning for simultaneous flow regime classification and velocity prediction</li>
    </ul>
    
    <h4>Limitations and Disclaimers</h4>
    <ul>
    <li><strong>Experimental Tool:</strong> This application is a research prototype and should not be used as the sole basis for critical operational decisions</li>
    <li><strong>Accuracy:</strong> While the model achieves high accuracy on test data, predictions may vary based on data quality and operating conditions</li>
    <li><strong>Scope:</strong> The model is trained on specific flow conditions and may not generalize to all multiphase flow scenarios</li>
    <li><strong>No Warranty:</strong> The application is provided "as is" without warranties of any kind, express or implied</li>
    </ul>
    
    <h4>User Responsibilities</h4>
    <ul>
    <li>Ensure uploaded data does not contain sensitive, proprietary, or confidential information beyond flow measurements</li>
    <li>Verify that your use of this application complies with your organization's data policies</li>
    <li>Validate predictions against experimental observations or other measurement methods</li>
    <li>Do not rely solely on model predictions for safety-critical applications</li>
    </ul>
    
    <h4>Future Development</h4>
    Future versions of this system may integrate:
    <ul>
    <li>CNN-based video feature extraction for visual flow pattern analysis</li>
    <li>Enhanced physics-informed constraints for improved accuracy</li>
    <li>Expanded multi-task learning capabilities for additional flow parameters</li>
    <li>Support for additional flow regimes and operating conditions</li>
    </ul>
    
    <h4>Contact and Support</h4>
    For questions, feedback, or research collaboration inquiries, please contact:
    <ul>
    <li>Qatar University Research Team</li>
    </ul>
    
    <h4>Updates to This Policy</h4>
    This privacy policy may be updated periodically to reflect changes in functionality or data handling practices. Users will be notified of significant changes through the application interface.
    
    <h4>Acceptance of Terms</h4>
    By using this application, you acknowledge that you have read and understood this privacy policy and agree to its terms regarding data processing and usage limitations.
    
    <hr>
    
    <p style="text-align: center; font-size: 0.9em; color: #999;">
    <strong>Last Updated:</strong> November 2024<br>
    <strong>Version:</strong> 1.0<br>
    © Qatar University Research Team
    </p>
    
    </div>
    """, unsafe_allow_html=True)