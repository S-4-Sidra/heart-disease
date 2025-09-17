import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import time
from PIL import Image
import requests
from io import BytesIO

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="❤️ HeartGuard - Heart Disease Predictor", layout="wide")

# -----------------------------
# Session State Defaults
# -----------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "page" not in st.session_state:
    st.session_state.page = "Welcome"
if "history" not in st.session_state:
    st.session_state.history = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# -----------------------------
# UI Helpers
# -----------------------------
def inject_css(dark=False):
    body_bg = "#0b1020" if dark else "#f9f9fb"
    text_color = "#e5e7eb" if dark else "#111827"
    card_bg = "#1f2937" if dark else "white"
    card_shadow = "0 4px 30px rgba(0,0,0,0.45)" if dark else "0 4px 30px rgba(0,0,0,0.12)"
    st.markdown(f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
        
        html, body, [class*="css"] {{
            background: {body_bg} !important;
            color: {text_color} !important;
            font-family: 'Roboto', sans-serif;
        }}
        .card {{
            background: {card_bg};
            border-radius: 16px;
            padding: 16px 18px;
            box-shadow: {card_shadow};
        }}
        @keyframes heartbeat {{
            0%, 100% {{ transform: scale(1); }}
            25%, 75% {{ transform: scale(1.1); }}
            50% {{ transform: scale(1.05); }}
        }}
        .heartbeat {{
            animation: heartbeat 2s infinite;
        }}
        .main-header {{
            font-size: 4rem;
            color: #ff4b4b;
            text-align: center;
            margin-bottom: 1.5rem;
            font-weight: 900;
            text-shadow: 3px 3px 6px rgba(0,0,0,0.2);
        }}
        .feature-box {{
            background: linear-gradient(145deg, #f0faff 0%, #d0e7ff 100%);
            padding: 2rem;
            border-radius: 20px;
            box-shadow: 0 12px 24px rgba(0,0,0,0.15);
            margin: 15px;
            text-align: center;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }}
        .feature-box:hover {{
            transform: translateY(-8px);
            box-shadow: 0 20px 30px rgba(0,0,0,0.2);
        }}
        .start-button {{
            display: block;
            width: 260px;
            margin: 2rem auto;
            background: linear-gradient(135deg, #ff4b4b 0%, #ff7373 100%);
            color: #fff;
            font-weight: 700;
            padding: 1rem 0;
            border-radius: 40px;
            font-size: 1.3rem;
            cursor: pointer;
            border: none;
            transition: all 0.3s ease;
        }}
        .start-button:hover {{
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(255,75,75,0.5);
        }}
        .emergency-card {{
            background: linear-gradient(135deg, #ffe6e6 0%, #ffb3b3 100%);
            padding: 1.5rem;
            border-radius: 20px;
            margin: 1rem 0;
            box-shadow: 0 6px 12px rgba(0,0,0,0.1);
        }}
        .success-icon {{
            font-size: 5rem;
            text-align: center;
            display: block;
            margin: 1rem auto;
            color: #4CAF50;
        }}
        </style>
    """, unsafe_allow_html=True)

def set_matplotlib_theme(dark=False):
    plt.style.use("dark_background" if dark else "default")

# -----------------------------
# Dummy model for demonstration
# -----------------------------
def create_dummy_model():
    from sklearn.ensemble import RandomForestClassifier
    np.random.seed(42)
    X_dummy = np.random.rand(100, 13)
    y_dummy = np.random.randint(0, 2, 100)
    model = RandomForestClassifier()
    model.fit(X_dummy, y_dummy)
    return model

# -----------------------------
# Load medical image
# -----------------------------
def load_medical_image():
    try:
        url = "https://img.freepik.com/free-vector/heart-care-medical-background_53876-95157.jpg"
        img = Image.open(BytesIO(requests.get(url).content))
        return img
    except:
        return None

# -----------------------------
# Diet plan and doctor functions
# -----------------------------
def get_diet_plan(risk_level):
    diet_plans = {
        "low": {
            "Breakfast": "Oatmeal with berries and nuts",
            "Lunch": "Grilled chicken salad with olive oil dressing",
            "Dinner": "Baked salmon with steamed vegetables",
            "Snacks": "Fresh fruits, yogurt, handful of almonds"
        },
        "medium": {
            "Breakfast": "Whole grain toast with avocado and eggs",
            "Lunch": "Quinoa bowl with vegetables and lean protein",
            "Dinner": "Grilled fish with brown rice and greens",
            "Snacks": "Apple slices with peanut butter, carrot sticks"
        },
        "high": {
            "Breakfast": "Smoothie with spinach, banana, and protein powder",
            "Lunch": "Lentil soup with whole grain bread",
            "Dinner": "Baked chicken with sweet potato and broccoli",
            "Snacks": "Walnuts, Greek yogurt, cucumber slices"
        }
    }
    return diet_plans.get(risk_level, diet_plans["medium"])

def get_doctor_recommendations(risk_level):
    recommendations = {
        "low": "Continue maintaining a healthy lifestyle with regular check-ups.",
        "medium": "Consider consulting a cardiologist for preventive advice.",
        "high": "Schedule an appointment with a cardiologist as soon as possible."
    }
    return recommendations.get(risk_level, "Please consult with a healthcare professional.")

def get_emergency_info():
    return {
        "Symptoms": "Chest pain, shortness of breath, palpitations, dizziness",
        "Immediate Action": "Call emergency services immediately",
        "While Waiting": "Sit down, try to stay calm, and take prescribed medication if available"
    }

# -----------------------------
# Dark Mode Toggle (Top Right)
# -----------------------------
t1, t2 = st.columns([0.75,0.25])
with t2:
    st.session_state.dark_mode = st.toggle("🌙 Dark mode", value=st.session_state.dark_mode)
inject_css(st.session_state.dark_mode)

# -----------------------------
# Welcome Page
# -----------------------------
if st.session_state.page == "Welcome":
    st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ff4b4b 0%, #ff7373 100%);
            padding: 120px 20px;
            border-radius: 20px;
            text-align: center;
            color: white;
            box-shadow: 0 4px 30px rgba(0,0,0,0.45);
        ">
            <h1 class="main-header">❤️ HeartGuard</h1>
            <h2 style="font-size:22px;">AI-powered heart disease risk assessment</h2>
            <p>Predict your heart disease risk instantly with advanced machine learning.</p>
        </div>
    """, unsafe_allow_html=True)

    st.write("### 🔎 Why use this app?")
    st.write("- ✅ Assess your heart disease risk instantly")
    st.write("- ✅ Get personalized diet and lifestyle recommendations")
    st.write("- ✅ Save and analyze your past assessments")
    st.write("- ✅ Professional tool for heart health monitoring")

    if st.button("🚀 Get Started", key="start_button"):
        st.session_state.logged_in = True
        st.session_state.page = "App"
        st.rerun()

    st.markdown("<div style='text-align:center; color: gray;'>© 2025 HeartGuard | Developed with ❤️ by Medical AI Team</div>", unsafe_allow_html=True)

# -----------------------------
# Main App with Tabs
# -----------------------------
elif st.session_state.logged_in:
    tabs = st.tabs([
        "❤️ Assessment",
        "📊 Results",
        "🍽️ Diet Plan",
        "🏥 Doctor Info",
        "🆘 Emergency",
        "🗃️ History",
        "🔑 Logout"
    ])

    # Load model
    try:
        model = create_dummy_model()
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        st.stop()

    # -----------------------------
    # ASSESSMENT TAB
    # -----------------------------
    with tabs[0]:
        st.title("❤️ Heart Disease Risk Assessment")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", 18, 100, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
            trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
            chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 600, 200)
            fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
            restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Abnormality", "Left Ventricular Hypertrophy"])
            
        with col2:
            thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
            exang = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
            oldpeak = st.number_input("ST Depression Induced by Exercise", 0.0, 6.2, 1.0, step=0.1)
            slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
            ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", 0, 3, 1)
            thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])
            
        # Convert inputs to model features
        sex_val = 1 if sex == "Male" else 0
        cp_map = {"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}
        cp_val = cp_map[cp]
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_map = {"Normal": 0, "ST-T Abnormality": 1, "Left Ventricular Hypertrophy": 2}
        restecg_val = restecg_map[restecg]
        exang_val = 1 if exang == "Yes" else 0
        slope_map = {"Upsloping": 0, "Flat": 1, "Downsloping": 2}
        slope_val = slope_map[slope]
        thal_map = {"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}
        thal_val = thal_map[thal]
        
        # Create feature array
        features = np.array([[age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val, 
                             thalach, exang_val, oldpeak, slope_val, ca, thal_val]])
        
        if st.button("Assess Risk 💓", key="assess_button"):
            with st.spinner("Analyzing your heart health..."):
                time.sleep(2)  # Simulate processing time
                
                # Generate prediction (using dummy model for demonstration)
                prediction = model.predict(features)[0]
                probability = model.predict_proba(features)[0][1]
                
                # Determine risk level
                if probability < 0.3:
                    risk_level = "low"
                    risk_text = "Low Risk"
                    risk_color = "green"
                elif probability < 0.6:
                    risk_level = "medium"
                    risk_text = "Medium Risk"
                    risk_color = "orange"
                else:
                    risk_level = "high"
                    risk_text = "High Risk"
                    risk_color = "red"
                
                # Store results in session state
                st.session_state.prediction = {
                    "risk_level": risk_level,
                    "risk_text": risk_text,
                    "risk_color": risk_color,
                    "probability": probability,
                    "features": {
                        "Age": age, "Sex": sex, "Chest Pain": cp, 
                        "Blood Pressure": trestbps, "Cholesterol": chol,
                        "Max Heart Rate": thalach
                    }
                }
                
                # Add to history
                st.session_state.history.append({
                    "Date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                    "Risk Level": risk_text,
                    "Probability": f"{probability:.2%}",
                    "Age": age,
                    "Sex": sex
                })
                
                st.success("Assessment complete! Navigate to the Results tab to see your analysis.")

    # -----------------------------
    # RESULTS TAB
    # -----------------------------
    with tabs[1]:
        st.title("📊 Assessment Results")
        
        if "prediction" in st.session_state:
            prediction = st.session_state.prediction
            
            st.markdown(f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f0f8ff;">
                    <h2 style="color: {prediction['risk_color']};">Your Heart Disease Risk: {prediction['risk_text']}</h2>
                    <h3>Probability: {prediction['probability']:.2%}</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Display feature importance (simulated)
            st.subheader("Key Factors in Your Assessment")
            feature_importance_data = {
                "Factor": ["Age", "Cholesterol", "Blood Pressure", "Max Heart Rate", "Chest Pain Type"],
                "Importance": [0.25, 0.20, 0.18, 0.15, 0.12]
            }
            feature_importance_df = pd.DataFrame(feature_importance_data)
            st.bar_chart(feature_importance_df.set_index("Factor"))
            
            # Recommendations based on risk level
            st.subheader("Recommendations")
            if prediction["risk_level"] == "low":
                st.success("""
                - Maintain your healthy lifestyle
                - Continue regular exercise
                - Annual check-ups are sufficient
                """)
            elif prediction["risk_level"] == "medium":
                st.warning("""
                - Consider lifestyle modifications
                - Increase physical activity
                - Monitor blood pressure regularly
                - Consider consulting a cardiologist
                """)
            else:
                st.error("""
                - Consult a cardiologist as soon as possible
                - Implement significant lifestyle changes
                - Monitor your health indicators regularly
                - Follow a heart-healthy diet strictly
                """)
        else:
            st.info("Complete an assessment on the Assessment tab to see your results here.")

    # -----------------------------
    # DIET PLAN TAB
    # -----------------------------
    with tabs[2]:
        st.title("🍽️ Personalized Diet Plan")
        
        if "prediction" in st.session_state:
            risk_level = st.session_state.prediction["risk_level"]
            diet_plan = get_diet_plan(risk_level)
            
            st.subheader(f"Diet Plan for {st.session_state.prediction['risk_text']} Risk")
            
            for meal, description in diet_plan.items():
                with st.expander(meal):
                    st.write(description)
            
            st.subheader("General Heart-Healthy Eating Tips")
            st.write("""
            - Choose foods low in saturated and trans fats
            - Increase intake of fruits and vegetables
            - Select whole grains over refined grains
            - Limit sodium intake
            - Choose lean protein sources
            """)
        else:
            st.info("Complete an assessment first to get a personalized diet plan.")

    # -----------------------------
    # DOCTOR INFO TAB
    # -----------------------------
    with tabs[3]:
        st.title("🏥 Doctor Information")
        
        if "prediction" in st.session_state:
            risk_level = st.session_state.prediction["risk_level"]
            recommendation = get_doctor_recommendations(risk_level)
            
            st.subheader("Professional Recommendation")
            st.info(recommendation)
            
            # Simulated doctor listings
            st.subheader("Cardiologists Near You")
            doctors_data = {
                "Name": ["Dr. Sarah Johnson", "Dr. Michael Chen", "Dr. Emily Williams"],
                "Specialty": ["Preventive Cardiology", "Interventional Cardiology", "Heart Failure Specialist"],
                "Hospital": ["City General Hospital", "University Medical Center", "Heart Institute"],
                "Rating": ["4.8/5", "4.7/5", "4.9/5"]
            }
            doctors_df = pd.DataFrame(doctors_data)
            st.dataframe(doctors_df, use_container_width=True)
        else:
            st.info("Complete an assessment first to get doctor recommendations.")

    # -----------------------------
    # EMERGENCY TAB
    # -----------------------------
    with tabs[4]:
        st.title("🆘 Emergency Information")
        
        emergency_info = get_emergency_info()
        
        st.markdown("""
        <div class="emergency-card">
            <h3 style="color: red;">Heart Attack Warning Signs</h3>
        </div>
        """, unsafe_allow_html=True)
        
        for key, value in emergency_info.items():
            with st.expander(key):
                st.write(value)
        
        st.subheader("Emergency Contacts")
        contacts_data = {
            "Service": ["Local Emergency", "National Heart Helpline", "Poison Control", "Local Hospital"],
            "Phone Number": ["911", "1-800-HEART", "1-800-222-1222", "Check local listing"]
        }
        contacts_df = pd.DataFrame(contacts_data)
        st.dataframe(contacts_df, use_container_width=True, hide_index=True)

    # -----------------------------
    # HISTORY TAB
    # -----------------------------
    with tabs[5]:
        st.title("🗃️ Assessment History")
        
        if st.session_state.history:
            df = pd.DataFrame(st.session_state.history)
            st.dataframe(df, use_container_width=True)
            
            # Add some simple analytics
            st.subheader("Risk Trend Over Time")
            if len(st.session_state.history) > 1:
                risk_map = {"Low Risk": 1, "Medium Risk": 2, "High Risk": 3}
                df["Risk Numeric"] = df["Risk Level"].map(risk_map)
                st.line_chart(df.set_index("Date")["Risk Numeric"])
            
            # Export option
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download History CSV", data=csv, 
                              file_name="heart_health_history.csv", mime="text/csv")
        else:
            st.info("No assessments yet. Complete an assessment to see your history here.")

    # -----------------------------
    # LOGOUT TAB
    # -----------------------------
    with tabs[6]:
        st.title("🔑 Logout")
        
        st.markdown("""
        <div style="text-align: center; padding: 40px;">
            <div class="heartbeat" style="font-size: 80px; color: #ff4b4b;">❤️</div>
            <h2>Thank you for using HeartGuard!</h2>
            <p>We hope our assessment helps you maintain a healthy heart.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Confirm Logout"):
            st.session_state.logged_in = False
            st.session_state.page = "Welcome"
            st.rerun()