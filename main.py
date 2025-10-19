"""
HVAC Cost Estimation Dashboard
Streamlit application for classifying rooms and estimating HVAC costs
"""

import streamlit as st
import pandas as pd
import os
from src.pipeline import run_pipeline

# Page configuration
st.set_page_config(
    page_title="FachPlaner Dashboard",
    page_icon="🏢",
    layout="wide"
)

# Title
st.title("🏢 FachPlanner DashBoard")
st.markdown("Upload your architect data get cost estimations")

st.divider()

# File upload
st.header("📤 Upload Input File")
uploaded_file = st.file_uploader(
    "Choose a CSV file (Architect room names, Area, Volume)",
    type=['csv']
)

if uploaded_file is not None:
    # Read file
    input_df = pd.read_csv(uploaded_file)
    
    # Validate columns
    required_cols = ['Architect room names', 'Area', 'Volume']
    if not all(col in input_df.columns for col in required_cols):
        st.error("❌ Missing required columns: 'Architect room names', 'Area', 'Volume'")
        st.stop()
    
    # Show preview
    st.success(f"✅ Loaded {len(input_df)} rooms")
    st.dataframe(input_df, use_container_width=True)
    
    st.divider()
    
    # Run button
    if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
        # Save input file
        os.makedirs("temp", exist_ok=True)
        input_path = "temp/input.csv"
        input_df.to_csv(input_path, index=False)
        
        # Check reference file
        reference_path = "data/reference.csv"
        if not os.path.exists(reference_path):
            st.error("❌ Reference file not found at data/reference.csv")
            st.stop()
        
        # Run pipeline
        with st.spinner("Running analysis..."):
            try:
                classified_df, power_df, cost_results = run_pipeline(
                    input_csv=input_path,
                    reference_csv=reference_path,
                    output_dir="output"
                )
                
                st.success("✅ Analysis complete!")
                
                # Results
                st.header("📊 Results")
                
                # Step 1: Classification
                st.subheader("🤖 Step 1: Room Classification")
                st.dataframe(classified_df, use_container_width=True)
                
                # Step 2: Power Requirements
                st.subheader("⚡ Step 2: Power Requirements")
                
                total_klt = power_df['KLT_required_KW'].sum()
                total_hlt = power_df['HLT_required_KW'].sum()
                total_rlt = power_df['RLT_required_m3h'].sum()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("🧊 Cooling (KLT)", f"{total_klt:.2f} KW")
                col2.metric("🔥 Heating (HLT)", f"{total_hlt:.2f} KW")
                col3.metric("💨 Ventilation (RLT)", f"{total_rlt:.2f} m³/h")
                
                st.dataframe(power_df[['Architect room names', 'BKW_name', 'Area', 
                                       'KLT_required_KW', 'HLT_required_KW', 'RLT_required_m3h']].round(2),
                            use_container_width=True)
                
                # Step 3: Cost Estimates
                st.subheader("💰 Step 3: Cost Estimates")
                
                tab1, tab2, tab3 = st.tabs(["🧊 Cooling", "🔥 Heating", "💨 Ventilation"])
                
                with tab1:
                    st.markdown(f"**Capacity:** {total_klt:.2f} KW")
                    st.markdown(cost_results['KLT'])
                
                with tab2:
                    st.markdown(f"**Capacity:** {total_hlt:.2f} KW")
                    st.markdown(cost_results['HLT'])
                
                with tab3:
                    st.markdown(f"**Capacity:** {total_rlt:.2f} m³/h")
                    st.markdown(cost_results['RLT'])
                
                # Downloads
                st.divider()
                st.subheader("📥 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    with open("output/classified_rooms.csv", "rb") as f:
                        st.download_button("📄 Classifications", f, "classified_rooms.csv", "text/csv")
                
                with col2:
                    with open("output/power_requirements.csv", "rb") as f:
                        st.download_button("⚡ Power Requirements", f, "power_requirements.csv", "text/csv")
                
                with col3:
                    with open("output/cost_estimates.txt", "rb") as f:
                        st.download_button("💰 Cost Estimates", f, "cost_estimates.txt", "text/plain")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.exception(e)

else:
    st.info("👆 Upload a CSV file to begin")