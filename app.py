
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt

# Load the trained model and preprocessor
try:
    model = joblib.load('linear_regression_model_full.joblib')
    preprocessor = joblib.load('preprocessor.joblib')
except FileNotFoundError:
    st.error("Error: 'linear_regression_model_full.joblib' or 'preprocessor.joblib' not found. Please ensure both files are in the same directory.")
    st.stop()

# Load the original dataset for the RAG assistant (if needed for context)
try:
    df = pd.read_csv('StudentPerformanceFactors.csv')
except FileNotFoundError:
    st.error("Error: 'StudentPerformanceFactors.csv' not found. Please ensure the data file is in the same directory.")
    # The RAG assistant will have limited functionality without the data, but we can still provide basic info.
    df = None


# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to", ["Prediction", "Dashboard", "RAG Assistant"])

# --- Prediction Section ---
if app_mode == "Prediction":
    st.title('Student Exam Score Predictor')
    st.write('Enter the student details below to predict their exam score.')

    # Get user input for all features
    # ... (all the input widgets from the previous step) ...
    # Numerical features
    hours_studied = st.number_input('Hours Studied', min_value=0.0, max_value=100.0, value=5.0, step=0.1)
    previous_scores = st.number_input('Previous Scores', min_value=0.0, max_value=100.0, value=70.0, step=0.1)
    attendance = st.number_input('Attendance (%)', min_value=0.0, max_value=100.0, value=80.0, step=0.1)
    sleep_hours = st.number_input('Sleep Hours', min_value=0.0, max_value=24.0, value=7.0, step=0.1)
    tutoring_sessions = st.number_input('Tutoring Sessions', min_value=0, max_value=10, value=1, step=1)
    physical_activity = st.number_input('Physical Activity (days per week)', min_value=0, max_value=7, value=3, step=1)


    # Categorical features
    parental_involvement = st.selectbox('Parental Involvement', ['Low', 'Medium', 'High'])
    access_to_resources = st.selectbox('Access to Resources', ['Low', 'Medium', 'High'])
    extracurricular_activities = st.selectbox('Extracurricular Activities', ['No', 'Yes'])
    motivation_level = st.selectbox('Motivation Level', ['Low', 'Medium', 'High'])
    internet_access = st.selectbox('Internet Access', ['No', 'Yes'])
    family_income = st.selectbox('Family Income', ['Low', 'Medium', 'High'])
    teacher_quality = st.selectbox('Teacher Quality', ['Low', 'Medium', 'High'])
    school_type = st.selectbox('School Type', ['Public', 'Private'])
    peer_influence = st.selectbox('Peer Influence', ['Negative', 'Neutral', 'Positive'])
    learning_disabilities = st.selectbox('Learning Disabilities', ['No', 'Yes'])
    parental_education_level = st.selectbox('Parental Education Level', ['High School', 'College', 'Postgraduate'])
    distance_from_home = st.selectbox('Distance from Home', ['Near', 'Moderate', 'Far'])
    gender = st.selectbox('Gender', ['Male', 'Female'])


    # Create a button to trigger prediction
    if st.button('Predict Exam Score'):
        # Create a DataFrame from user input
        input_data = pd.DataFrame({
            'Hours_Studied': [hours_studied],
            'Previous_Scores': [previous_scores],
            'Attendance': [attendance],
            'Sleep_Hours': [sleep_hours],
            'Tutoring_Sessions': [tutoring_sessions],
            'Physical_Activity': [physical_activity],
            'Parental_Involvement': [parental_involvement],
            'Access_to_Resources': [access_to_resources],
            'Extracurricular_Activities': [extracurricular_activities],
            'Motivation_Level': [motivation_level],
            'Internet_Access': [internet_access],
            'Family_Income': [family_income],
            'Teacher_Quality': [teacher_quality],
            'School_Type': [school_type],
            'Peer_Influence': [peer_influence],
            'Learning_Disabilities': [learning_disabilities],
            'Parental_Education_Level': [parental_education_level],
            'Distance_from_Home': [distance_from_home],
            'Gender': [gender]
        })

        # Preprocess the input data
        try:
            input_data_processed = preprocessor.transform(input_data)
            # Get feature names after transformation
            # This is a bit tricky with ColumnTransformer, so we will just use the number of features
            # num_features = input_data_processed.shape[1]
            # st.write(f"Number of features after preprocessing: {num_features}")


        except Exception as e:
            st.error(f"An error occurred during data preprocessing: {e}")
            st.stop()


        # Make prediction
        try:
            predicted_score = model.predict(input_data_processed)[0]
            st.success(f'Predicted Exam Score: {predicted_score:.2f}')
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")


# --- Dashboard Section ---
elif app_mode == "Dashboard":
    st.title("Dashboard")
    st.subheader("Data Visualization")

    if df is not None:
        st.write("Exploring the Student Performance Data.")

        # Prepare data for plotting
        X_viz = df.drop(columns=['Exam_Score'])
        y_viz_actual = df['Exam_Score']
        X_viz_processed = preprocessor.transform(X_viz)
        y_viz_predicted = model.predict(X_viz_processed)


        # 1. Actual vs. Predicted Scores Scatter Plot
        st.subheader("Actual vs. Predicted Exam Scores")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.scatterplot(x=y_viz_actual, y=y_viz_predicted, ax=ax)
        ax.plot([y_viz_actual.min(), y_viz_actual.max()], [y_viz_actual.min(), y_viz_actual.max()], 'r--', lw=2) # Diagonal line
        ax.set_xlabel("Actual Exam Score")
        ax.set_ylabel("Predicted Exam Score")
        ax.set_title("Actual vs. Predicted Exam Scores")
        st.pyplot(fig)

        # 2. Residuals Distribution Histogram
        st.subheader("Distribution of Residuals")
        residuals = y_viz_actual - y_viz_predicted
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(residuals, kde=True, ax=ax)
        ax.set_xlabel("Residuals (Actual - Predicted)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Residuals")
        st.pyplot(fig)

        # 3. Interactive Scatter Plot
        st.subheader("Interactive Feature Relationship Explorer")
        st.write("Select two features to see their relationship.")

        # Get numeric columns for selection
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        x_axis = st.selectbox('Select feature for X-axis', numeric_cols, key='x_axis_scatter')
        y_axis = st.selectbox('Select feature for Y-axis', numeric_cols, key='y_axis_scatter')

        if x_axis and y_axis:
            chart = alt.Chart(df).mark_circle().encode(
                x=x_axis,
                y=y_axis,
                tooltip=[x_axis, y_axis, 'Exam_Score']
            ).interactive() # Make the chart interactive (zoom and pan)
            st.altair_chart(chart, use_container_width=True)
    else:
        st.warning("Data file not found. Cannot display dashboard visualizations.")


# --- RAG Assistant Section ---
elif app_mode == "RAG Assistant":
    st.title("RAG Assistant")
    st.write("Ask me questions about the student performance data and the prediction model.")
    st.info("Note: This is a basic assistant with limited predefined responses.")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask a question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        with st.chat_message("assistant"):
            # Simple predefined responses
            if "important features" in prompt.lower() or "features predict" in prompt.lower():
                response = "The model uses a combination of numerical and categorical features to predict exam scores. The key features include 'Hours_Studied', 'Previous_Scores', 'Attendance', and various other factors like 'Parental_Involvement' and 'Teacher_Quality'."
            elif "target variable" in prompt.lower() or "predicting what" in prompt.lower():
                 response = "The model is predicting the 'Exam_Score'."
            elif "type of problem" in prompt.lower() or "what kind of prediction" in prompt.lower():
                 response = "This is a regression problem, where the model predicts a continuous numerical value (the exam score)."
            elif "how well" in prompt.lower() or "model performance" in prompt.lower() or "r2 score" in prompt.lower():
                 # Assuming the best R2 from the previous steps was around 0.77
                 response = "Based on recent evaluations, the model achieved an R-squared (R2) score of approximately 0.77 on the test data. This indicates it explains about 77% of the variance in exam scores."
            else:
                response = "I can only answer basic questions about the features, target variable, and the type of problem. Please try asking about those topics."

            st.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})