Hospitals and clinics lose a lot of time and money because many patients book appointments but do not show up. This causes empty slots, long waiting times for other patients, and poor staff planning. This project predicts which patients are likely to miss their appointment and forecasts daily patient demand so clinics can plan better.
For this we create three tabs as- Risk Prediction (Classification),Demand Forecast (Regression),Model Insights & Evaluation.
## Results
- No-Show: Accuracy 0.73, F1 0.722, No-Show Recall 0.83
- Demand: TEST MAPE 13.4%, R2 0.903 | TRAIN MAPE 4.8%, R2 0.992

## How to Run
pip install -r requirements.txt
streamlit run app.py

## Models
Models are saved as .pkl files in the models/ folder:
- noshow_model.pkl - Random Forest Classifier
- demand_model.pkl - Random Forest Regressor
- noshow_features.pkl, demand_features.pkl, daily.pkl

