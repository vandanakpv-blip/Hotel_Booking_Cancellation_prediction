import os
import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import joblib

# Set Page Config
st.set_page_config(
    page_title="Hotel Booking Cancellation Prediction",
    page_icon="🏨",
    layout="wide"
)

# Robust Base Directory Resolution
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "hotel_booking_cancellation_model.pkl").exists() and (BASE_DIR / "Hotel_Booking_Cancellation_Prediction_Sklearn" / "hotel_booking_cancellation_model.pkl").exists():
    BASE_DIR = BASE_DIR / "Hotel_Booking_Cancellation_Prediction_Sklearn"

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-cancelled {
        background: linear-gradient(135deg, #EF4444 0%, #B91C1C 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
    .badge-confirmed {
        background: linear-gradient(135deg, #10B981 0%, #047857 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .metric-banner {
        font-size: 2.3rem;
        font-weight: 800;
        margin: 0.3rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🏨 Hotel Booking Cancellation Risk Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Predict hotel reservation cancellations and protect room occupancy using a trained <b>RandomForest Pipeline</b> with categorical encoding.</div>', unsafe_allow_html=True)

model_path = BASE_DIR / "hotel_booking_cancellation_model.pkl"
csv_path = BASE_DIR / "data" / "hotel_bookings.csv"
chart_path = BASE_DIR / "feature_importance.png"

if not model_path.exists():
    st.error(f"Model file not found at `{model_path}`! Please run 'python train_model.py' first.")
    st.stop()

@st.cache_resource
def get_model():
    return joblib.load(str(model_path))

pipeline = get_model()

tab1, tab2, tab3 = st.tabs(["🔮 Reservation Cancellation Predictor", "📈 Feature Importance & Drivers", "📋 Historical Bookings Dataset"])

with tab1:
    col_input, col_result = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("Reservation Details")
        
        scenario = st.selectbox(
            "⚡ Quick Booking Scenario Preset",
            ["Custom Reservation", "🚨 High-Risk Long Lead Online TA (No Deposit)", "💼 Secure Corporate Business Traveler", "🏖️ Direct Family Vacationer"]
        )
        
        if scenario == "🚨 High-Risk Long Lead Online TA (No Deposit)":
            def_lead, def_ad, def_ch, def_we, def_wd, def_pc, def_bc, def_dep, def_cust, def_seg, def_spec = 150, 2, 0, 2, 4, 1, 0, "No Deposit", "Transient", "Online TA", 0
        elif scenario == "💼 Secure Corporate Business Traveler":
            def_lead, def_ad, def_ch, def_we, def_wd, def_pc, def_bc, def_dep, def_cust, def_seg, def_spec = 12, 1, 0, 0, 2, 0, 1, "Non Refund", "Contract", "Corporate", 1
        elif scenario == "🏖️ Direct Family Vacationer":
            def_lead, def_ad, def_ch, def_we, def_wd, def_pc, def_bc, def_dep, def_cust, def_seg, def_spec = 35, 2, 2, 2, 3, 0, 2, "Refundable", "Transient", "Direct", 2
        else:
            def_lead, def_ad, def_ch, def_we, def_wd, def_pc, def_bc, def_dep, def_cust, def_seg, def_spec = 45, 2, 0, 1, 2, 0, 0, "No Deposit", "Transient", "Online TA", 1
            
        with st.form("hotel_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                lead_time = st.slider("Lead Time (Days before arrival)", 0, 365, def_lead)
                adults = st.slider("Adults", 1, 5, def_ad)
                children = st.slider("Children", 0, 5, def_ch)
                weekend_nights = st.slider("Weekend Nights", 0, 8, def_we)
                weekday_nights = st.slider("Weekday Nights", 0, 15, def_wd)
                special_requests = st.slider("Special Requests Count", 0, 5, def_spec)
            with col_b:
                deposit_type = st.selectbox("Deposit Type", ["No Deposit", "Non Refund", "Refundable"], index=["No Deposit", "Non Refund", "Refundable"].index(def_dep))
                customer_type = st.selectbox("Customer Type", ["Transient", "Contract", "Group", "Transient-Party"], index=["Transient", "Contract", "Group", "Transient-Party"].index(def_cust))
                market_segment = st.selectbox("Market Segment", ["Online TA", "Offline TA", "Direct", "Corporate", "Groups"], index=["Online TA", "Offline TA", "Direct", "Corporate", "Groups"].index(def_seg))
                previous_cancellations = st.slider("Previous Cancellations History", 0, 10, def_pc)
                booking_changes = st.slider("Changes Made to Booking", 0, 10, def_bc)
                
            submit_btn = st.form_submit_button("🚀 Check Cancellation Risk", use_container_width=True)
            
    with col_result:
        st.subheader("Reservation Risk Outcome")
        if submit_btn:
            booking = pd.DataFrame([{
                "lead_time": lead_time,
                "adults": adults,
                "children": children,
                "weekend_nights": weekend_nights,
                "weekday_nights": weekday_nights,
                "previous_cancellations": previous_cancellations,
                "booking_changes": booking_changes,
                "deposit_type": deposit_type,
                "customer_type": customer_type,
                "market_segment": market_segment,
                "special_requests": special_requests
            }])
            
            pred = pipeline.predict(booking)[0]
            prob_arr = pipeline.predict_proba(booking)[0]
            # Prob of cancellation (class 1)
            classes = list(pipeline.classes_)
            cancel_idx = classes.index(1) if 1 in classes else 1
            prob_cancel = prob_arr[cancel_idx]
            
            if pred == 1:
                st.markdown(f"""
                <div class="badge-cancelled">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Prediction Outcome</div>
                    <div class="metric-banner">🚨 LIKELY TO CANCEL</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Cancellation Probability: {prob_cancel:.1%}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="badge-confirmed">
                    <div style="font-size: 0.95rem; opacity: 0.9;">Prediction Outcome</div>
                    <div class="metric-banner">✅ LIKELY TO HONOUR BOOKING</div>
                    <div style="font-size: 1.15rem; font-weight: 600;">Fulfillment Likelihood: {(1 - prob_cancel):.1%}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.write("")
            st.markdown("### 📊 Probability Breakdown")
            st.progress(float(prob_cancel))
            col_p1, col_p2 = st.columns(2)
            col_p1.metric("Cancellation Risk", f"{prob_cancel:.1%}")
            col_p2.metric("Expected Check-in", f"{(1 - prob_cancel):.1%}")
            
            st.markdown("### 🛎️ Revenue Management Strategy")
            if lead_time > 60 and deposit_type == "No Deposit":
                st.warning("⚠️ **Extended Window Alert**: High lead time (>60 days) without deposit has historically high fallout rates. Send an email confirmation re-engagement sequence.")
            if deposit_type == "Non Refund":
                st.success("🔒 **Revenue Secured**: Non-refundable policy protects room revenue even in case of no-show.")
            if special_requests >= 2:
                st.info("⭐ **High Engagement**: Guest submitted multiple special requests, correlating with higher intent to stay.")
            if previous_cancellations > 0:
                st.error("📉 **Guest History**: Account has prior cancellations on record.")
                
            with st.expander("🔍 Booking Input Vector"):
                st.json(booking.to_dict(orient="records")[0])
        else:
            st.info("👈 Enter reservation details and click **'Check Cancellation Risk'**.")

with tab2:
    st.subheader("Top Predictive Factors")
    if chart_path.exists():
        st.image(str(chart_path), caption="Top Features for Hotel Booking Cancellation", use_container_width=True)
    else:
        st.info("Chart will appear after running train_model.py")

with tab3:
    st.subheader("Historical Reservations (hotel_bookings.csv)")
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Bookings", f"{len(df):,}")
        col2.metric("Overall Cancellation Rate", f"{df['cancelled'].mean():.1%}")
        col3.metric("Avg Lead Time", f"{df['lead_time'].mean():.0f} days")
        st.dataframe(df.head(100), use_container_width=True)
    else:
        st.warning("Dataset not found.")
