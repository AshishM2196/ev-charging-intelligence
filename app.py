import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import folium
from streamlit_folium import st_folium
from xgboost import XGBRegressor
import heapq

st.set_page_config(page_title="EV Intelligence", layout="wide", page_icon="⚡")

# ==========================================
# 1. DATA GENERATION & LOADING (Cached)
# ==========================================
@st.cache_data
def load_and_generate_data():
    np.random.seed(42)
    
    # Stations
    stations = []
    base_lat, base_lon = 26.9124, 75.7873 # Jaipur coordinates
    for i in range(1, 21):
        stations.append({
            "station_id": f"ST-{i:03d}",
            "name": f"Station {i}",
            "latitude": base_lat + np.random.uniform(-0.05, 0.05),
            "longitude": base_lon + np.random.uniform(-0.05, 0.05),
            "total_ports": np.random.choice([2, 4, 8, 12]),
            "power_kw": np.random.choice([11, 22, 50, 150])
        })
    stations_df = pd.DataFrame(stations)
    
    # Sessions
    sessions = []
    start_date = datetime.now() - timedelta(days=30)
    hourly_probs = [0.01]*6 + [0.05, 0.10, 0.15, 0.10, 0.05, 0.05, 0.04, 0.04, 0.04, 0.05, 0.06, 0.08, 0.05, 0.03, 0.02, 0.01, 0.01, 0.01]
    
    for _, station in stations_df.iterrows():
        daily_traffic = np.random.randint(5, 30) * (station['total_ports'] / 4)
        for day in range(30):
            current_date = start_date + timedelta(days=day)
            vol_mult = 0.6 if current_date.weekday() >= 5 else 1.2
            for _ in range(int(daily_traffic * vol_mult * np.random.uniform(0.8, 1.2))):
                hour = int(np.random.choice(np.arange(24), p=hourly_probs))
                conn_time = current_date.replace(hour=hour, minute=np.random.randint(0, 60), second=0, microsecond=0)
                dur = max(0.2, min(np.random.lognormal(0.5, 0.5), 8.0))
                sessions.append({
                    "session_id": f"SES-{np.random.randint(100000, 999999)}",
                    "station_id": station['station_id'],
                    "connection_time": conn_time,
                    "duration_hours": round(dur, 2)
                })
                
    sessions_df = pd.DataFrame(sessions)
    sessions_df['date_hour'] = sessions_df['connection_time'].dt.floor('h')
    
    # Hourly Aggregation
    hourly_df = sessions_df.groupby(['station_id', 'date_hour']).agg(
        arrivals=('session_id', 'count'),
        avg_duration=('duration_hours', 'mean')
    ).reset_index()
    
    all_hours = pd.date_range(start=sessions_df['connection_time'].min().floor('h'), 
                              end=sessions_df['connection_time'].max().floor('h'), freq='h')
    idx = pd.MultiIndex.from_product([stations_df['station_id'], all_hours], names=['station_id', 'date_hour'])
    hourly_df = hourly_df.set_index(['station_id', 'date_hour']).reindex(idx, fill_value=0).reset_index()
    
    return stations_df, hourly_df

# ==========================================
# 2. CONGESTION & SIMULATION LOGIC
# ==========================================
def calculate_congestion(station_id, hourly_df, stations_df, current_hour):
    ports = stations_df[stations_df['station_id'] == station_id].iloc[0]['total_ports']
    df = hourly_df.copy()
    df['hour'] = df['date_hour'].dt.hour
    hist = df[(df['station_id'] == station_id) & (df['hour'] == current_hour)]
    
    arr_rate = hist['arrivals'].mean() if not hist.empty else 0
    avg_dur = hist['avg_duration'].replace(0, np.nan).mean()
    if pd.isna(avg_dur) or avg_dur == 0: avg_dur = 1.0
    
    srv_rate = 1.0 / avg_dur
    util = min(arr_rate / (ports * srv_rate) if (ports * srv_rate) > 0 else 0, 0.99)
    
    prob_wait = (util ** ports) / (1 - util + 0.001)
    wait_mins = max(0, round((prob_wait * (1 / (ports * srv_rate - arr_rate + 0.001))) * 60))
    
    cat = "Low" if util < 0.5 else "Moderate" if util < 0.75 else "High" if util < 0.9 else "Severe"
    return {"category": cat, "utilization": round(util * 100, 1), "wait_mins": wait_mins, "rate": round(arr_rate, 2)}

class ChargingSimulator:
    def __init__(self, ports): self.ports = ports
    def simulate(self, arrivals, avg_dur_mins):
        events, queue, wait_times, served, avail = [], [], [], 0, self.ports
        for i, (t, res) in enumerate(arrivals): heapq.heappush(events, (t, 1, i, res))
        
        while events:
            curr_t, e_type, e_id, is_res = heapq.heappop(events)
            if e_type == 1:
                if avail > 0:
                    avail -= 1; wait_times.append(0); served += 1
                    heapq.heappush(events, (curr_t + max(10, np.random.normal(avg_dur_mins, 15)), 2, e_id, None))
                else: heapq.heappush(queue, (0 if is_res else 1, curr_t, e_id))
            elif e_type == 2:
                if queue:
                    _, q_t, q_id = heapq.heappop(queue)
                    wait_times.append(curr_t - q_t); served += 1
                    heapq.heappush(events, (curr_t + max(10, np.random.normal(avg_dur_mins, 15)), 2, q_id, None))
                else: avail += 1
        return {"served": served, "avg_wait": round(np.mean(wait_times), 1) if wait_times else 0, "q_len": len(queue)}

@st.cache_resource
def train_forecast_model(hourly_df):
    df = hourly_df.copy()
    df['hour'] = df['date_hour'].dt.hour
    df['is_weekend'] = df['date_hour'].dt.dayofweek.isin([5, 6]).astype(int)
    df['lag_24'] = df.groupby('station_id')['arrivals'].shift(24)
    df = df.dropna()
    model = XGBRegressor(n_estimators=50, random_state=42).fit(df[['hour', 'is_weekend', 'lag_24']], df['arrivals'])
    return model

# ==========================================
# 3. STREAMLIT UI
# ==========================================
stations_df, hourly_df = load_and_generate_data()
model = train_forecast_model(hourly_df)

st.title("⚡ EV Charging Congestion & Reservation Intelligence")

col1, col2 = st.columns(2)
sel_station = col1.selectbox("Select Station", stations_df['station_id'].unique())
sel_hour = col2.slider("Time of Day (Hour)", 0, 23, 12)

t1, t2, t3, t4, t5 = st.tabs(["Overview", "Map", "Forecasting", "Reservations", "Simulator"])

with t1:
    m = calculate_congestion(sel_station, hourly_df, stations_df, sel_hour)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stations", len(stations_df))
    c2.metric("Est. Wait Time", f"{m['wait_mins']} mins")
    c3.metric("Congestion", m['category'])
    c4.metric("Utilization", f"{m['utilization']}%")
    
    st.plotly_chart(px.line(hourly_df.groupby(hourly_df['date_hour'].dt.hour)['arrivals'].mean().reset_index(), 
                            x='date_hour', y='arrivals', title="Network Average Arrivals by Hour"), use_container_width=True)

with t2:
    st.markdown("Geospatial view of congestion at the selected hour.")
    fmap = folium.Map(location=[stations_df['latitude'].mean(), stations_df['longitude'].mean()], zoom_start=12)
    for _, r in stations_df.iterrows():
        sm = calculate_congestion(r['station_id'], hourly_df, stations_df, sel_hour)
        color = "green" if sm['category'] == "Low" else "orange" if sm['category'] == "Moderate" else "red"
        folium.CircleMarker([r['latitude'], r['longitude']], radius=8, color=color, fill=True, 
                            popup=f"{r['name']} - Wait: {sm['wait_mins']}m").add_to(fmap)
    st_folium(fmap, width=1000, height=500)

with t3:
    st.plotly_chart(px.line(hourly_df[hourly_df['station_id'] == sel_station].tail(168), 
                            x='date_hour', y='arrivals', title=f"7-Day Demand History: {sel_station}"), use_container_width=True)

with t4:
    with st.form("res_form"):
        r_time = st.time_input("Arrival Time")
        if st.form_submit_button("Evaluate Reservation Slot"):
            m = calculate_congestion(sel_station, hourly_df, stations_df, r_time.hour)
            if m['category'] in ["High", "Severe"]: st.success(f"High congestion ({m['wait_mins']}m wait). Reserving will bypass the queue.")
            else: st.info(f"Slot is optimal. Walk-in wait is minimal ({m['wait_mins']}m).")

with t5:
    c1, c2, c3 = st.columns(3)
    s_ports = c1.number_input("Ports", 1, 20, 4)
    s_arrs = c2.number_input("Hourly Arrivals", 1, 50, 6)
    s_res = c3.slider("% Reservations", 0, 100, 20)
    
    if st.button("Run Simulation"):
        sim = ChargingSimulator(s_ports)
        arrs = [(t, np.random.rand() < (s_res/100.0)) for t in np.sort(np.random.uniform(0, 60, s_arrs))]
        res = sim.simulate(arrs, 45)
        r1, r2, r3 = st.columns(3)
        r1.metric("Served", res['served'], f"out of {s_arrs}")
        r2.metric("Avg Wait", f"{res['avg_wait']} mins")
        r3.metric("Unserved (Queue)", res['q_len'])