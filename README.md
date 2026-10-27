# ⚡ EV Charging Congestion & Reservation Intelligence

**Live Dashboard:** - https://Ashish-ev-intelligence-muj.streamlit.app/

## 📌 Overview
An interactive, data-driven platform designed to help EV drivers and charging-network operators understand charging demand, identify congestion, and simulate capacity optimization. Built with Python and Streamlit, this application leverages machine learning and queueing theory to predict network demand and evaluate reservation priority rules.

## 🚀 Key Features
* **Congestion Analytics:** Implements an M/M/c queueing theory approximation to classify station congestion and estimate dynamic wait times based on historical arrival and service rates.
* **Demand Forecasting:** Utilizes an XGBoost regression model to predict hourly station demand, utilizing temporal features and autoregressive 24-hour lag data.
* **Discrete-Event Simulation:** Features a custom Python-based reservation simulator using priority queues (`heapq`) to evaluate "what-if" scenarios, tracking minute-by-minute port occupancy and reservation priority rules.
* **Geospatial Intelligence:** Interactive city-wide mapping of station network congestion using Folium.

## 🛠️ Tech Stack
* **Language:** Python 3
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Web Framework:** Streamlit
* **Visualization:** Plotly, Folium

## 💻 Run Locally
1. Clone the repository: `git clone [https://github.com/AshishM2196/ev-charging-intelligence.git](https://github.com/AshishM2196/ev-charging-intelligence.git)`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 👨‍💻 Author
**Ashish Meena**  
Master of Technology in Data Science, Manipal University Jaipur
