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
## app screenshots
tab1:<img width="426" height="239" alt="me1" src="https://github.com/user-attachments/assets/1df51919-8a15-4aa0-b7f0-300662dbbf67" />
<img width="541" height="323" alt="me2" src="https://github.com/user-attachments/assets/2af14f78-079a-421c-bed8-341086f65cbc" />
tab2:<img width="470" height="211" alt="me3" src="https://github.com/user-attachments/assets/372e8d4c-7472-498a-9aaf-ed0896bb6a42" />
tab3:<img width="545" height="298" alt="me7" src="https://github.com/user-attachments/assets/40f17230-a1d3-419f-a498-0590916e3fe4" />



