import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# 1. Page Configuration
st.set_page_config(
    page_title="PH Industry Insights",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Load Your Clean Dataset
@st.cache_data
def load_data():
    df = pd.read_csv("dashboard.csv")
    df['Timeline'] = df['MONTH'] + " " + df['YEAR'].astype(str)
    month_order = {'January': 1, 'April': 2, 'July': 3, 'October': 4}
    df['Month_Num'] = df['MONTH'].map(month_order)
    df = df.sort_values(by=['YEAR', 'Month_Num'])
    df['Time_Index'] = range(len(df))
    return df

df = load_data()

# Industry Mapping
def get_major_sector(industry):
    agriculture_keywords = ['Agriculture', 'Forestry', 'Fishing', 'Crop', 'Livestock']
    if any(keyword in industry for keyword in agriculture_keywords):
        return 'Agriculture'
    
    industry_keywords = [
        'Manufacturing', 'Construction', 'Mining', 'Quarrying', 
        'Electricity', 'Gas', 'Steam', 'Air Conditioning', 
        'Water Supply', 'Sewerage', 'Waste Management'
    ]
    if any(keyword in industry for keyword in industry_keywords):
        return 'Industry'
    
    return 'Services'

# Add major sector column to dataframe
df['Major_Sector'] = df['Industry'].apply(get_major_sector)

# Premium UI Theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp{
    background:
    radial-gradient(circle at top left, rgba(59,130,246,0.20), transparent 35%),
    radial-gradient(circle at top right, rgba(139,92,246,0.20), transparent 35%),
    linear-gradient(135deg,#0f172a 0%,#111827 50%,#020617 100%);
}

.main .block-container{
    padding-top: 1.5rem;
    max-width: 95%;
}

div[data-testid="stMetric"]{
    background: rgba(255,255,255,0.05)!important;
    border: 1px solid rgba(255,255,255,0.10)!important;
    border-radius: 20px!important;
    padding: 18px!important;
    backdrop-filter: blur(18px);
    box-shadow: 0 8px 30px rgba(0,0,0,.25);
}

section[data-testid="stSidebar"]{
    background: rgba(10,15,30,.92)!important;
    border-right: 1px solid rgba(255,255,255,.08);
}

div[data-testid="stMetric"]:hover{
    transform: translateY(-4px);
    transition: .25s ease;
}

h1{
    font-weight:700!important;
    letter-spacing:-1px;
}

.stPlotlyChart{
    background: rgba(255,255,255,.04);
    border: 1px solid rgba(255,255,255,.08);
    border-radius: 20px;
    padding: 12px;
    backdrop-filter: blur(12px);
}

[data-testid="stDataFrame"]{
    background: rgba(255,255,255,.04);
    border-radius: 20px;
    overflow:hidden;
}

hr{
    border-color: rgba(255,255,255,.08)!important;
}
</style>
""", unsafe_allow_html=True)

# 4. Sidebar Navigation Controls
st.sidebar.title("🎛️ Navigation")
st.sidebar.markdown("---")

nav_options = {
    "dashboard": "📈 Dashboard Overview",
    "employment_sector": "👥 Employment by Sector",
    "wage_analysis": "💰 Wage Analysis by Industry",
    "forecasting": "🔮 Employment Forecasting"
    
    
}

if 'navigation' not in st.session_state:
    st.session_state.navigation = "dashboard"

for nav_key, nav_label in nav_options.items():
    if st.sidebar.button(nav_label, key=nav_key, use_container_width=True):
        st.session_state.navigation = nav_key
        
st.markdown("""
<style>
/* Increase sidebar button padding and font size for nav buttons */
section[data-testid="stSidebar"] button[kind="secondary"],
section[data-testid="stSidebar"] button {
    padding-top: 0.9rem !important;
    padding-bottom: 0.9rem !important;
    font-size: 1.05rem !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("📌 **Data Source:** Philippine Statistcis Authority")

# 5. Main Content
if st.session_state.navigation == "dashboard":
    st.title("📊 National Industry Labor & Wage Dashboard")
    st.markdown("---")
    
    # ============ QUICK STATS SECTION ============
    st.markdown("### 🎯 Quick Stats Overview")
    
    total_industries = len(df['Industry'].unique())
    total_records = len(df)
    latest_year = df['YEAR'].max()
    date_range = f"{df['YEAR'].min()} - {df['YEAR'].max()}"
    
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    with qcol1:
        st.metric(label="🏭 Total Industries", value=total_industries, help="Number of unique industry sectors tracked")
    with qcol2:
        st.metric(label="📅 Total Records", value=total_records, help="Total quarterly data points collected")
    with qcol3:
        st.metric(label="📆 Latest Data Year", value=latest_year, help="Most recent year in the dataset")
    with qcol4:
        st.metric(label="🕒 Date Range", value=date_range, help="Time span covered by the data")
    
    st.markdown("---")
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    overall_avg_wage = df['Average_Daily_Pay'].mean()
    overall_avg_employment = df['Employed_Persons'].mean()
    highest_paying_industry = df.groupby('Industry')['Average_Daily_Pay'].mean().idxmax()
    highest_wage_value = df.groupby('Industry')['Average_Daily_Pay'].mean().max()
    
    with col_insight1:
        st.metric(label="💡 Overall Avg Daily Wage", value=f"₱{overall_avg_wage:.2f}")
    with col_insight2:
        st.metric(label="👥 Overall Avg Employment", value=f"{overall_avg_employment:,.0f}K")
    with col_insight3:
        st.metric(label="🏆 Highest Paying Industry", value=highest_paying_industry, delta=f"₱{highest_wage_value:.2f}")
    
    st.markdown("---")
    
    # ============ HIERARCHICAL INDUSTRY SELECTOR BELOW QUICK STATS ============
    st.markdown("### 🎯 Select Industry for Analysis")
    
    sel_col1, sel_col2 = st.columns(2)
    with sel_col1:
        major_sectors = ["All Major Sectors"] + sorted(list(df['Major_Sector'].unique()))
        selected_major_sector = st.selectbox("Select Major Sector:", major_sectors, key="dashboard_major_select")
        
    with sel_col2:
        if selected_major_sector == "All Major Sectors":
            industries_in_sector = ["All Industries"]
        else:
            industries_in_sector = ["All " + selected_major_sector + " Combined"] + sorted(list(df[df['Major_Sector'] == selected_major_sector]['Industry'].unique()))
        selected_industry = st.selectbox("Select Specific Industry Sector:", industries_in_sector, key="dashboard_ind_select")
    
    st.markdown("---")
    
    # Base filtering logic 
    if selected_major_sector == "All Major Sectors":
        if selected_industry == "All Industries":
            filtered_df = df.groupby('Timeline', as_index=False).agg({'Employed_Persons': 'sum', 'Average_Daily_Pay': 'mean', 'YEAR': 'first', 'Month_Num': 'first'})
            display_title = "All Industries"
        else:
            filtered_df = df[df['Industry'] == selected_industry]
            display_title = selected_industry
    else:
        if selected_industry.startswith("All "):
            filtered_df = df[df['Major_Sector'] == selected_major_sector].groupby('Timeline', as_index=False).agg({'Employed_Persons': 'sum', 'Average_Daily_Pay': 'mean', 'YEAR': 'first', 'Month_Num': 'first'})
            display_title = f"All {selected_major_sector} Combined"
        else:
            filtered_df = df[df['Industry'] == selected_industry]
            display_title = selected_industry
            
    filtered_df = filtered_df.sort_values(by=['YEAR', 'Month_Num'])

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    if not filtered_df.empty:
        latest_data = filtered_df.iloc[-1]
        avg_emp = filtered_df['Employed_Persons'].mean()
        avg_wage = filtered_df['Average_Daily_Pay'].mean()
        emp_growth = ((filtered_df['Employed_Persons'].iloc[-1] - filtered_df['Employed_Persons'].iloc[0]) / filtered_df['Employed_Persons'].iloc[0] * 100) if len(filtered_df) > 1 else 0
        wage_growth = ((filtered_df['Average_Daily_Pay'].iloc[-1] - filtered_df['Average_Daily_Pay'].iloc[0]) / filtered_df['Average_Daily_Pay'].iloc[0] * 100) if len(filtered_df) > 1 else 0
        
        with col1:
            st.metric(label="Latest Employment", value=f"{latest_data['Employed_Persons']:.0f}K", delta=f"{emp_growth:.1f}% overall")
        with col2:
            st.metric(label="Latest Daily Wage", value=f"₱{latest_data['Average_Daily_Pay']:.2f}", delta=f"{wage_growth:.1f}% overall")
        with col3:
            st.metric(label="Avg Employment", value=f"{avg_emp:.0f}K")
        with col4:
            st.metric(label="Avg Daily Wage", value=f"₱{avg_wage:.2f}")
    
    st.markdown(" ")
    
    # Visualizations
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("📈 Employment Trend Over Time")
        fig_emp = px.line(filtered_df, x='Timeline', y='Employed_Persons', markers=True, template='plotly_dark', title=f"{display_title} - Employment Trend")
        fig_emp.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        fig_emp.update_traces(line=dict(color='#00ff88', width=2), marker=dict(size=8))
        st.plotly_chart(fig_emp, use_container_width=True)
    
    with chart_col2:
        st.subheader("💰 Average Daily Pay Growth")
        fig_wage = px.bar(filtered_df, x='Timeline', y='Average_Daily_Pay', template='plotly_dark', color='Average_Daily_Pay', color_continuous_scale='Viridis', title=f"{display_title} - Wage Trend")
        fig_wage.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wage, use_container_width=True)
    
    st.subheader("📊 Employment vs. Wage Correlation")
    fig_scatter = px.scatter(filtered_df, x='Employed_Persons', y='Average_Daily_Pay', color='Timeline', template='plotly_dark', title=f"Correlation in {display_title}", labels={'Employed_Persons': 'Employment Count (K)', 'Average_Daily_Pay': 'Daily Wage (₱)'})
    fig_scatter.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

elif st.session_state.navigation == "forecasting":
    # ============ FORECASTING TAB (ANNUAL + HIERARCHICAL) ============
    st.title("🔮 Annual Employment Forecasting Engine")
    st.markdown("Predict annual employment trajectories based on year-on-year industrial data trends using Linear Regression models.")
    st.markdown("---")
    
    # 1. Selection UI
    f_col1, f_col2 = st.columns(2)
    with f_col1:
        f_major_sectors = ["All Major Sectors"] + sorted(list(df['Major_Sector'].unique()))
        selected_f_major = st.selectbox("Select Target Major Sector:", f_major_sectors, key="forecasting_major_select")
        
    with f_col2:
        if selected_f_major == "All Major Sectors":
            f_industries = ["All Industries Combined"]
        else:
            f_industries = [f"All {selected_f_major} Combined"] + sorted(list(df[df['Major_Sector'] == selected_f_major]['Industry'].unique()))
        selected_f_industry = st.selectbox("Select Target Specific Industry:", f_industries, key="forecasting_ind_select")
        
    col_param1, col_param2 = st.columns(2)
    with col_param1:
        forecast_years = st.slider("Number of years to forecast ahead:", min_value=1, max_value=5, value=3, help="Forecast horizon configured annually.")
    with col_param2:
        show_confidence = st.checkbox("Show Prediction Interval Bounds", value=True)
        
    # 2. Extract Data & Transform to Annual Aggregates
    if selected_f_major == "All Major Sectors":
        if selected_f_industry == "All Industries Combined":
            forecast_base_df = df.copy()
            forecast_title = "All Industries Combined"
        else:
            forecast_base_df = df[df['Industry'] == selected_f_industry]
            forecast_title = selected_f_industry
    else:
        if selected_f_industry.startswith("All "):
            forecast_base_df = df[df['Major_Sector'] == selected_f_major]
            forecast_title = f"All {selected_f_major} Combined"
        else:
            forecast_base_df = df[df['Industry'] == selected_f_industry]
            forecast_title = selected_f_industry

    # Convert structural timeline records to yearly averages/sums
    annual_df = forecast_base_df.groupby('YEAR', as_index=False).agg({'Employed_Persons': 'mean'}) # tracking annualized average volumes
    annual_df = annual_df.sort_values(by='YEAR').reset_index(drop=True)
    annual_df['Year_Index'] = range(len(annual_df))
    
    if len(annual_df) >= 3:
        # Train linear model
        X_annual = annual_df['Year_Index'].values.reshape(-1, 1)
        y_annual = annual_df['Employed_Persons'].values
        
        model_annual = LinearRegression()
        model_annual.fit(X_annual, y_annual)
        
        # Performance Evaluation Metrics
        y_pred_annual = model_annual.predict(X_annual)
        mae_annual = mean_absolute_error(y_annual, y_pred_annual)
        r2_annual = r2_score(y_annual, y_pred_annual)
        
        # Calculate future projections
        last_year_index = annual_df['Year_Index'].max()
        last_actual_year = int(annual_df['YEAR'].max())
        
        future_indices = np.arange(last_year_index + 1, last_year_index + forecast_years + 1).reshape(-1, 1)
        predictions_annual = model_annual.predict(future_indices)
        future_years = [str(last_actual_year + i) for i in range(1, forecast_years + 1)]
        
        # Build out structural output prediction frame
        forecast_output_df = pd.DataFrame({
            'YEAR': future_years,
            'Forecasted_Employment': predictions_annual,
            'Lower_Bound': predictions_annual * 0.92,
            'Upper_Bound': predictions_annual * 1.08
        })
        
        # Metrics Display block
        st.markdown("### 📊 Model Quality Metrics (Annual Evaluation)")
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.metric("Model R² Fitness Score", f"{r2_annual:.3f}", help="Variance alignment score.")
        with m_col2:
            st.metric("Mean Absolute Error (MAE)", f"{mae_annual:.1f}K", help="Average annual historical error gap.")
        with m_col3:
            st.metric("Forecast Window", f"{forecast_years} Years", help="Operational horizon range.")
            
        # Chart visualization pipeline
        st.markdown("### 📈 Historical and Projected Structural Run-rates")
        fig_f_annual = make_subplots(rows=1, cols=1)
        
        # Actuals Trace
        fig_f_annual.add_trace(go.Scatter(
            x=annual_df['YEAR'].astype(str), y=annual_df['Employed_Persons'],
            mode='lines+markers', name='Historical Annual Avg',
            line=dict(color='#00ff88', width=3), marker=dict(size=8)
        ))
        
        # Forecasts Trace
        fig_f_annual.add_trace(go.Scatter(
            x=forecast_output_df['YEAR'], y=forecast_output_df['Forecasted_Employment'],
            mode='lines+markers', name='Projected Forecast Line',
            line=dict(color='#ff6b6b', width=3, dash='dash'), marker=dict(size=9, symbol='diamond')
        ))
        
        if show_confidence:
            fig_f_annual.add_trace(go.Scatter(
                x=pd.concat([forecast_output_df['YEAR'], forecast_output_df['YEAR'][::-1]]),
                y=pd.concat([forecast_output_df['Upper_Bound'], forecast_output_df['Lower_Bound'][::-1]]),
                fill='toself', fillcolor='rgba(255, 107, 107, 0.15)',
                line=dict(color='rgba(255,255,255,0)'), name='Proportional Target Range'
            ))
            
        fig_f_annual.update_layout(
            template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis_title="Calendar Year", yaxis_title="Employed Population Volume (K)", hovermode='x unified'
        )
        st.plotly_chart(fig_f_annual, use_container_width=True)
        
        # Output Context Tables
        st.markdown("### 📋 Annual Projections Details")
        st.dataframe(
            forecast_output_df.style.format({
                'Forecasted_Employment': '{:.1f}K',
                'Lower_Bound': '{:.1f}K',
                'Upper_Bound': '{:.1f}K'
            }), use_container_width=True
        )
        
        # Summary analysis block
        st.markdown("### 📈 Strategic Growth Dynamics")
        baseline_historical_val = annual_df['Employed_Persons'].iloc[-1]
        terminal_projection_val = predictions_annual[-1]
        overall_delta_rate = ((terminal_projection_val - baseline_historical_val) / baseline_historical_val) * 100
        
        st.info(f"📊 **Dynamic Projection Summary:** Structural demand profile for **{forecast_title}** indicates an estimated change trajectory shifting by **{overall_delta_rate:+.2f}%** over the upcoming **{forecast_years} year(s)** horizon.")
    else:
        st.warning("⚠️ High aggregation structural failure: Insufficient historical annual data points inside your target selection criteria to run regressions safely.")

elif st.session_state.navigation == "wage_analysis":
    st.title("💰 Comparative Wage Analysis Across Industries")
    st.markdown("Compare average daily wages across different industry sectors")
    st.markdown("---")
    
    industry_list = sorted(df['Industry'].unique())
    available_years = sorted(df['YEAR'].unique())
    selected_year = st.selectbox("Select Year for Analysis:", available_years, index=len(available_years)-1)
    
    year_data = df[df['YEAR'] == selected_year]
    wage_by_industry = year_data.groupby('Industry')['Average_Daily_Pay'].mean().sort_values(ascending=True).reset_index()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏆 Top 5 Highest Paying Industries")
        top_5 = wage_by_industry.nlargest(5, 'Average_Daily_Pay')
        fig_top = px.bar(top_5, x='Average_Daily_Pay', y='Industry', orientation='h', template='plotly_dark', color='Average_Daily_Pay', color_continuous_scale='Greens', title=f"Highest Paying Industries ({selected_year})")
        fig_top.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_top, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Bottom 5 Lowest Paying Industries")
        bottom_5 = wage_by_industry.nsmallest(5, 'Average_Daily_Pay')
        fig_bottom = px.bar(bottom_5, x='Average_Daily_Pay', y='Industry', orientation='h', template='plotly_dark', color='Average_Daily_Pay', color_continuous_scale='Reds', title=f"Lowest Paying Industries ({selected_year})")
        fig_bottom.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_bottom, use_container_width=True)
    
    st.subheader(f"📊 Complete Industry Wage Comparison ({selected_year})")
    fig_all_wages = px.bar(wage_by_industry, x='Industry', y='Average_Daily_Pay', template='plotly_dark', color='Average_Daily_Pay', color_continuous_scale='Viridis', title=f"Average Daily Wage by Industry - {selected_year}", labels={'Average_Daily_Pay': 'Daily Wage (₱)', 'Industry': 'Industry Sector'})
    fig_all_wages.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', xaxis_tickangle=-45)
    st.plotly_chart(fig_all_wages, use_container_width=True)
    
    st.subheader("📈 Wage Trends Over Time (Select Multiple Industries)")
    selected_industries_wage = st.multiselect("Choose industries to compare:", industry_list, default=['Manufacturing', 'Construction'])
    
    if selected_industries_wage:
        wage_trend_data = df[df['Industry'].isin(selected_industries_wage)]
        fig_wage_trend = px.line(wage_trend_data, x='Timeline', y='Average_Daily_Pay', color='Industry', template='plotly_dark', markers=True, title="Wage Trends Comparison")
        fig_wage_trend.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_wage_trend, use_container_width=True)

elif st.session_state.navigation == "employment_sector":
    st.title("👥 Employment Analysis by Sector")
    st.markdown("Compare employment levels across different industry sectors")
    st.markdown("---")
    
    available_years = sorted(df['YEAR'].unique())
    col1, col2 = st.columns(2)
    with col1:
        selected_year_emp = st.selectbox("Select Year:", available_years, key="emp_year_selector")
    with col2:
        selected_quarter = st.selectbox("Select Quarter:", ['January', 'April', 'July', 'October'], key="quarter_selector")
    
    quarter_data = df[(df['YEAR'] == selected_year_emp) & (df['MONTH'] == selected_quarter)]
    
    if not quarter_data.empty:
        emp_by_industry = quarter_data.sort_values('Employed_Persons', ascending=True)
        
        st.subheader(f"📊 Employment Distribution - {selected_quarter} {selected_year_emp}")
        fig_emp_dist = px.bar(emp_by_industry, x='Employed_Persons', y='Industry', orientation='h', template='plotly_dark', color='Employed_Persons', color_continuous_scale='Blues', title=f"Number of Employed Persons by Industry ({selected_quarter} {selected_year_emp})", labels={'Employed_Persons': 'Number of Employed Persons (Thousands)', 'Industry': 'Industry Sector'})
        fig_emp_dist.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=800)
        st.plotly_chart(fig_emp_dist, use_container_width=True)
        
        st.subheader("🥧 Employment Market Share")
        top_10_industries = emp_by_industry.nlargest(10, 'Employed_Persons')
        fig_pie = px.pie(top_10_industries, values='Employed_Persons', names='Industry', template='plotly_dark', title=f"Top 10 Industries by Employment ({selected_quarter} {selected_year_emp})", hole=0.3)
        fig_pie.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig_pie, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; opacity: 0.7;'>📊 Data Source: <a href='https://openstat.psa.gov.ph/Database/Labor-and-Employment' target='_blank'> Philippine Statistics Authority - Labor and Employment</a> | Dashboard built with Streamlit</p>", unsafe_allow_html=True)

