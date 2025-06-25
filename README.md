# 🛢️ US Pipeline Incidents: A Full Data Project

This project explores and analyzes pipeline incidents across the United States using a complete data science workflow. It includes:

- 📊 **Exploratory Data Analysis** in Python  
- 📈 **Interactive Dashboards** with Power BI  
- 🤖 **Predictive Modeling and Deployment** using Streamlit

---

## 🔍 Part 1: Python Analysis

Using a dataset of reported pipeline accidents from 2010–2021, I explored patterns in causes, locations, financial impacts, and more.

### 📌 Key Insights:
- **Incident Trend**: Peaked in 2016, with monthly fluctuations.
- **Delay in Identification**: Some incidents were only discovered long after they occurred.
- **Concentration**: A handful of operators and states (like Texas) accounted for most incidents.
- **Causes**: Equipment failure and corrosion were dominant.
- **Material**: Carbon steel appeared in 77% of incidents.
- **Environmental**: Soil contamination was most common.
- **Cost**: Emergency response made up 55% of total reported costs, largely due to a $480M event in 2022.

### ✅ Suggestions:
- Enhanced oversight for high-incident operators.
- Improve corrosion detection and monitoring.
- Optimize emergency response and cleanup strategies.
- Develop data-driven preventive maintenance.

---

## 📊 Part 2: Power BI Dashboard

I built an interactive dashboard in Power BI to visualize:

- Incident trends by year and month  
- Geographic breakdowns  
- Costs by location, cause, and facility  
- Spill details, injuries, fatalities, and more  

📷 _[Include dashboard screenshots in your GitHub repo or README]_

---

## 🤖 Part 3: Predictive Model & Deployment

After trying different models (Random Forest, XGBoost), I chose a **Support Vector Regressor** as the final model based on performance:

- **MAE**: `15.78%`
- **Median Absolute Error**: `1.33%`
- **Target**: Percent of unintentional spill volume recovered

💡 I deployed the model using **Streamlit**, allowing users to test the recovery rate prediction based on incident details.

▶️ [View the Live App](https://oil-facility-accidents-sv-model.streamlit.app/)  

---

## 📂 Project Structure

```bash
.
├── notebooks/               # Python analysis and EDA
├── dashboard/               # Power BI dashboard files
├── model/                   # Model training, evaluation, and pipeline
├── streamlit_app/           # Deployed app code
├── data/                    # Processed datasets
└── README.md

