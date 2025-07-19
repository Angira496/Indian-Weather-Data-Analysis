import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import statsmodels.api as sm
import matplotlib.dates as mdates

st.set_page_config(
    page_title="Indian Weather Data Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def load_data(path):
    return pd.read_csv(path)


DATA_PATH = "modified_data.csv"
try:
    df = load_data(DATA_PATH)
except Exception as e:
    st.error(f"Failed to load data: {e}")
    st.stop()

df["date"] = pd.to_datetime(df["date"], errors="coerce")

st.sidebar.title("🔎 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Analysis", "Outcomes"])

if page == "Home":
    st.title("🌦️ Indian Weather Data Analysis")
    st.markdown(
        "###### A deep dive into 10 years of daily weather across 8 major Indian cities",
        unsafe_allow_html=True,
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Cities Studied", "8")
    k2.metric("Total Records", f"{len(df):,}")
    k3.metric("Time Span", "2009–2019")
    k4.metric("Key Parameters", "27")

    st.markdown("---")

    col_img, col_text = st.columns([1, 2])
    with col_img:
        st.image("pexels-jplenio-1118873.jpg", use_container_width=True)
    with col_text:
        st.markdown(
            """
            ### Project Overview  
            Analyze and visualize weather patterns across coastal, inland, and plateau cities to uncover:
            - **Seasonal Trends:** Monsoons, heatwaves, winter chills  
            - **Regional Contrasts:** Humidity, UV exposure, air quality  
            - **Actionable Insights:** Agriculture planning, urban resilience, public health strategies  
            """
        )

    with st.expander("📈 Importance & Context", expanded=False):
        st.markdown(
            """
            - **Geographic Impact:** From Mumbai’s coastal humidity to Delhi’s dry heat, weather shapes  
              agriculture, infrastructure, and health.  
            - **Climate Challenges:** Unpredictable monsoons, extreme temperatures, air pollution episodes.  
            - **Applications:**  
              • Crop scheduling in Punjab/Kanpur  
              • Flood & heat‐wave resilience in metros  
              • Air quality management in Bengaluru/Hyderabad  
            """
        )

    with st.expander("📦 Dataset & Methodology", expanded=False):
        st.markdown(
            """
            **Dataset**  
            • Source: Kaggle “Historical Weather Data for Indian Cities”  
            • 771,264 daily records · 28 attributes · 0 missing values  

            **Methodology**  
            1. **Cleaning & Merging:** Standardize city labels, split date/time, remove partial‐year 2020  
            2. **Exploratory Analysis:** Descriptive stats, correlations, seasonality, outliers  
            3. **Visualizations:** Heatmaps, time series, comparative bar/line plots  
            """
        )

    st.markdown("---")
    st.caption(
        "Use the sidebar to navigate to the Dataset, Analysis, or Outcomes pages."
    )


elif page == "Dataset":
    st.title("📊 Dataset Preview & Basic EDA")

    total_rows, total_cols = df.shape
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Records", f"{total_rows:,}")
    m2.metric("Total Columns", f"{total_cols}")
    m3.metric("Time Span", "2009–2019")

    st.markdown("---")

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Full Dataset (CSV)",
        data=csv,
        file_name="modified_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.markdown("---")

    tab_scope, tab_eda = st.tabs(["📋 Scope & Coverage", "🔍 Basic EDA"])

    with tab_scope:
        st.markdown(
            """
            **Cities:** Pune, Bombay (Mumbai), Delhi, Hyderabad, Jaipur, Kanpur, Nagpur, Bengaluru  
            **Time Period:** 2009–2019 (11 years of daily records)  

            **Key Parameters:**  
            - Temperature (high/low)  
            - Humidity & Wind/Gust Speed  
            - Precipitation & Cloud Cover  
            - UV Index & Air Quality  

            **Source & Structure:**  
            - **Name:** Historical Weather Data for Indian Cities  
            - **Origin:** Kaggle  
            - **Records:** 771,264  
            - **Columns:** 28  
            - **Completeness:** Zero missing values  

            **Highlights:**  
            - 8 cities spanning diverse climatic zones  
            - 11 critical daily weather variables  
            - Ready-to-use with no gap-filling needed
            """
        )
        with st.expander("🛠️ Data Preparation Pseudocode", expanded=False):
            st.code(
                """
# 1. Add 'city' column to each CSV, save updated versions
# 2. Merge updated CSVs into merged_data.csv
# 3. Split 'date_time' into 'date' & 'time'
# 4. Drop all 2020 rows (only 10 days)
                """,
                language="python",
            )

    with tab_eda:
        st.subheader("1. Data Preview & Dimensions")
        c1, c2 = st.columns([2, 1])

        with c1:
            st.markdown("**Top 15 Rows**")
            st.code("df.head(15)")
            st.dataframe(df.head(15), use_container_width=True)

        with c2:
            st.markdown("**Shape**")
            st.code("df.shape")
            st.write(df.shape)

            st.markdown("---")

            st.markdown("**Dtypes**")
            st.code("df.dtypes")
            st.write(df.dtypes)

        st.markdown("---")

        s1, s2 = st.columns(2)
        with s1:
            st.subheader("2. Missing Values")
            st.code("df.isnull().sum()")
            st.write(df.isnull().sum())
        with s2:
            st.subheader("3. Unique Values")
            st.code("df.nunique()")
            st.write(df.nunique())

        st.markdown("---")

        st.subheader("4. Statistical Summaries")
        num_tab, all_tab = st.tabs(["Numeric Only", "All Columns"])
        with num_tab:
            st.code("df.describe()")
            st.write(df.describe())
        with all_tab:
            st.code('df.describe(include="all")')
            st.write(df.describe(include="all"))

        st.markdown("---")

        st.subheader("5. Interactive Exploration")

        ie1, ie2 = st.columns(2)

        filtered_columns = [col for col in df.columns if col != "date_time"]

        with ie1:
            st.markdown("**📊 Value Counts**")
            col_vc = st.selectbox("Choose column", filtered_columns, key="vc_col")

            vc = df[col_vc].value_counts().reset_index()
            vc.columns = [col_vc, "Count"]

            fig_vc = px.bar(
                vc,
                x=col_vc,
                y="Count",
                title=f"Value Counts for '{col_vc}'",
                template="plotly_dark",
                color="Count",
                color_continuous_scale="Viridis",
            )
            fig_vc.update_layout(
                xaxis_title="", yaxis_title="Frequency", showlegend=False
            )
            st.plotly_chart(fig_vc, use_container_width=True)

        with ie2:
            st.markdown("**📦 Outlier Boxplot**")
            numeric_cols = df.select_dtypes(
                include=["int64", "float64"]
            ).columns.tolist()
            col_box = st.selectbox("Numeric column", numeric_cols, key="box_col")

            fig_box = px.box(
                df,
                x=col_box,
                title=f"Boxplot of '{col_box}'",
                template="plotly_dark",
                color_discrete_sequence=["cyan"],
            )
            st.plotly_chart(fig_box, use_container_width=True)

        st.markdown("---")

        st.subheader("6. Feature Correlation Heatmap")

        numeric_df = df.select_dtypes(include=["float64", "int64"])

        corr_matrix = numeric_df.corr()

        fig_corr = go.Figure(
            data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale="Viridis",
                colorbar=dict(title="Correlation"),
                zmin=-1,
                zmax=1,
            )
        )

        fig_corr.update_layout(
            title="📊 Correlation Matrix (Numeric Features)",
            template="plotly_dark",
            height=800,
            xaxis=dict(tickangle=45, side="bottom"),
            yaxis=dict(autorange="reversed"),
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.markdown("---")

        st.subheader("7. Sample & Column Details")
        sample_size = st.slider(
            "Select sample size", min_value=5, max_value=100, value=10, step=5
        )
        st.dataframe(df.sample(sample_size), use_container_width=True)

        st.markdown("**Column Descriptions:**")
        col = st.selectbox(
            "Pick a column to see its description", df.columns, key="col_desc"
        )
        st.write(
            f"**`{col}`** — dtype: `{df[col].dtype}`;  unique: {df[col].nunique()} values"
        )

        st.markdown("---")

        st.subheader("8. Data Quality Checks")
        dq1, dq2 = st.columns(2)
        with dq1:
            st.markdown("**Zero-value counts**")
            zero_counts = (
                (df.select_dtypes(include=["float64", "int64"]) == 0)
                .sum()
                .sort_values(ascending=False)
            )
            st.bar_chart(zero_counts[zero_counts > 0], use_container_width=True)
        with dq2:
            st.markdown("**Duplicate rows**")
            dup = int(df.duplicated().sum())
            st.write(f"Found **{dup}** duplicate rows")
            if dup:
                if st.button("Show duplicates"):
                    st.dataframe(
                        df[df.duplicated(keep=False)], use_container_width=True
                    )

        st.markdown("---")

        st.subheader("9. Export Filtered Subset")
        cities = df["city"].unique().tolist()
        chosen_city = st.multiselect("Filter by city", options=cities, default=cities)
        subset = df[df["city"].isin(chosen_city)]
        csv_subset = subset.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download Filtered Data",
            data=csv_subset,
            file_name=f"weather_{'_'.join(chosen_city)}.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.markdown("---")
        st.caption("End of Dataset Page")


elif page == "Analysis":
    st.title("Full Exploratory Data Analysis : -")

    analysis_topic = st.sidebar.radio(
        "Analysis Topics",
        [
            "1. Temperature Trends",
            "2. Visibility and Humidity Analysis",
            "3. Wind & Gust Analysis",
            "4. Precipitation Overview",
            "5. UV Index Variations",
            "6. Cloud Cover & Solar Potential",
            "7. Pressure Insights",
            "8. Moon illumination",
        ],
    )

    if analysis_topic == "1. Temperature Trends":
        st.header("Temperature Trends : - ")

        st.subheader("1. Average Temperature by City")
        avg_temp_by_city = df.groupby("city")["tempC"].mean().reset_index()
        avg_temp_by_city.columns = ["City", "Average Temperature (°C)"]
        st.dataframe(avg_temp_by_city)

        st.subheader("2. Maximum and Minimum Temperatures by City")

        max_temp = df.loc[df.groupby("city")["maxtempC"].idxmax()][
            ["city", "date", "time", "maxtempC"]
        ]
        min_temp = df.loc[df.groupby("city")["mintempC"].idxmin()][
            ["city", "date", "time", "mintempC"]
        ]

        max_temp = max_temp.rename(
            columns={
                "city": "City",
                "date": "Max Temp Date",
                "time": "Max Temp Time",
                "maxtempC": "Max Temperature (°C)",
            }
        )
        min_temp = min_temp.rename(
            columns={
                "city": "City",
                "date": "Min Temp Date",
                "time": "Min Temp Time",
                "mintempC": "Min Temperature (°C)",
            }
        )

        st.markdown("**Maximum Temperature Records**")
        st.dataframe(max_temp)

        st.markdown("**Minimum Temperature Records**")
        st.dataframe(min_temp)

        st.subheader("3. Temperature Comparison Across Cities")

        df["tempC"] = pd.to_numeric(df["tempC"], errors="coerce")
        df["mintempC"] = pd.to_numeric(df["mintempC"], errors="coerce")
        df["maxtempC"] = pd.to_numeric(df["maxtempC"], errors="coerce")

        city_data = (
            df.groupby("city")
            .agg(
                Average=("tempC", "mean"),
                Minimum=("mintempC", "min"),
                Maximum=("maxtempC", "max"),
            )
            .reset_index()
        )

        melted = city_data.melt(
            id_vars="city",
            value_vars=["Average", "Minimum", "Maximum"],
            var_name="Temperature Type",
            value_name="Temperature (°C)",
        )

        fig = px.bar(
            melted,
            x="city",
            y="Temperature (°C)",
            color="Temperature Type",
            barmode="group",
            title="Temperature Comparison for Each City - Overall",
            labels={"city": "City"},
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("4. Yearly & Range Temperature Stats for a City")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        cities = df["city"].unique().tolist()
        selected_city = st.selectbox("Choose a city", cities)

        mode = st.radio("Choose mode", ["Specific Year", "Year Range"])

        if mode == "Specific Year":
            year = st.number_input(
                "Select Year",
                min_value=int(df["date"].dt.year.min()),
                max_value=int(df["date"].dt.year.max()),
                value=int(df["date"].dt.year.min()),
                step=1,
            )
            year_data = df[(df["date"].dt.year == year) & (df["city"] == selected_city)]
            if year_data.empty:
                st.warning(f"No data for {selected_city} in {year}.")
            else:
                month_order = [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ]
                stats = (
                    year_data.assign(month=year_data["date"].dt.month_name())
                    .groupby("month")
                    .agg(
                        avg=("tempC", "mean"),
                        mn=("mintempC", "min"),
                        mx=("maxtempC", "max"),
                    )
                    .reindex(month_order)
                    .dropna(how="all")
                    .reset_index()
                )

                fig_spec = px.line(
                    stats,
                    x="month",
                    y=["avg", "mn", "mx"],
                    title=f"{selected_city} Temperatures in {year}",
                    labels={
                        "value": "Temperature (°C)",
                        "month": "Month",
                        "variable": "Stat",
                    },
                    markers=True,
                )
                fig_spec.update_traces(
                    selector=dict(name="avg"), line_dash="solid", marker_symbol="circle"
                )
                fig_spec.update_traces(
                    selector=dict(name="mn"), line_dash="dash", marker_symbol="square"
                )
                fig_spec.update_traces(
                    selector=dict(name="mx"), line_dash="dot", marker_symbol="diamond"
                )
                fig_spec.update_xaxes(categoryorder="array", categoryarray=month_order)

                st.plotly_chart(fig_spec, use_container_width=True)

        else:
            start_year = st.number_input(
                "Start Year",
                min_value=int(df["date"].dt.year.min()),
                max_value=int(df["date"].dt.year.max()),
                value=int(df["date"].dt.year.min()),
                step=1,
            )
            end_year = st.number_input(
                "End Year",
                min_value=int(df["date"].dt.year.min()),
                max_value=int(df["date"].dt.year.max()),
                value=int(df["date"].dt.year.max()),
                step=1,
            )
            if start_year > end_year:
                st.error("Start year must be ≤ end year.")
            else:
                range_data = df[
                    (df["date"].dt.year.between(start_year, end_year))
                    & (df["city"] == selected_city)
                ]
                if range_data.empty:
                    st.warning(
                        f"No data for {selected_city} between {start_year}–{end_year}."
                    )
                else:
                    yearly = (
                        range_data.assign(year=range_data["date"].dt.year)
                        .groupby("year")
                        .agg(
                            avg=("tempC", "mean"),
                            mn=("mintempC", "min"),
                            mx=("maxtempC", "max"),
                        )
                        .reindex(range(start_year, end_year + 1), fill_value=None)
                        .reset_index()
                    )

                    fig_range = px.line(
                        yearly,
                        x="year",
                        y=["avg", "mn", "mx"],
                        title=f"{selected_city} Temperature {start_year}–{end_year}",
                        labels={
                            "value": "Temperature (°C)",
                            "year": "Year",
                            "variable": "Stat",
                        },
                        markers=True,
                    )
                    fig_range.update_traces(
                        selector=dict(name="avg"),
                        line_dash="solid",
                        marker_symbol="circle",
                    )
                    fig_range.update_traces(
                        selector=dict(name="mn"),
                        line_dash="dash",
                        marker_symbol="square",
                    )
                    fig_range.update_traces(
                        selector=dict(name="mx"),
                        line_dash="dot",
                        marker_symbol="diamond",
                    )
                    fig_range.update_xaxes(dtick=1)

                    st.plotly_chart(fig_range, use_container_width=True)

        st.subheader("5. Heat Index & Wind Chill Comparison")

        heat_stats = (
            df.assign(
                HeatIndexC=pd.to_numeric(df["HeatIndexC"], errors="coerce"),
                WindChillC=pd.to_numeric(df["WindChillC"], errors="coerce"),
            )
            .groupby("city")
            .agg({"HeatIndexC": "max", "WindChillC": "min"})
            .reset_index()
        )

        st.dataframe(heat_stats)

        heatmap_data = heat_stats.melt(
            id_vars="city", var_name="Condition", value_name="Value"
        ).pivot(index="city", columns="Condition", values="Value")

        fig_heat = px.imshow(
            heatmap_data,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            labels={"x": "Condition", "y": "City", "color": "Value (°C)"},
            text_auto=True,
            aspect="auto",
            title="Max Heat Index & Min Wind Chill by City",
        )

        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("6. Heat Index & Wind Chill by Month/Year for a City : ")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month_name()
        df["year"] = df["date"].dt.year
        df["HeatIndexC"] = pd.to_numeric(df["HeatIndexC"], errors="coerce")
        df["WindChillC"] = pd.to_numeric(df["WindChillC"], errors="coerce")

        city = st.selectbox("Select City", df["city"].unique())
        mode = st.radio("View by", ["Monthly (Single Year)", "Yearly (Range)"])

        if mode == "Monthly (Single Year)":
            year = st.number_input(
                "Select Year",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=int(df["year"].min()),
                step=1,
                key="heatmap_select_year",
            )
            data_year = df[(df["city"] == city) & (df["year"] == year)]
            if data_year.empty:
                st.warning(f"No data for {city} in {year}.")
            else:
                all_months = [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ]
                hi = (
                    data_year.pivot_table(
                        index="month", values="HeatIndexC", aggfunc="max"
                    )
                    .reindex(all_months)
                    .reset_index()
                )
                wc = (
                    data_year.pivot_table(
                        index="month", values="WindChillC", aggfunc="min"
                    )
                    .reindex(all_months)
                    .reset_index()
                )

                fig_hi = px.imshow(
                    [hi["HeatIndexC"].tolist()],
                    x=hi["month"],
                    y=[f"HeatIndex (max)"],
                    labels=dict(x="Month", y="", color="°C"),
                    title=f"Heat Index in {city} ({year})",
                    aspect="auto",
                )
                st.plotly_chart(fig_hi, use_container_width=True)

                fig_wc = px.imshow(
                    [wc["WindChillC"].tolist()],
                    x=wc["month"],
                    y=[f"Wind Chill (min)"],
                    labels=dict(x="Month", y="", color="°C"),
                    title=f"Wind Chill in {city} ({year})",
                    aspect="auto",
                )
                st.plotly_chart(fig_wc, use_container_width=True)

        else:
            start_y = st.number_input(
                "Start Year",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=int(df["year"].min()),
                step=1,
                key="heatmap_start_year",
            )
            end_y = st.number_input(
                "End Year",
                min_value=int(df["year"].min()),
                max_value=int(df["year"].max()),
                value=int(df["year"].max()),
                step=1,
                key="heatmap_end_year",
            )
            if start_y > end_y:
                st.error("Start Year must be ≤ End Year.")
            else:
                data_range = df[
                    (df["city"] == city) & (df["year"].between(start_y, end_y))
                ]
                if data_range.empty:
                    st.warning(f"No data for {city} between {start_y}–{end_y}.")
                else:
                    hi_y = (
                        data_range.pivot_table(
                            index="year", values="HeatIndexC", aggfunc="max"
                        )
                        .sort_index()
                        .reset_index()
                    )
                    wc_y = (
                        data_range.pivot_table(
                            index="year", values="WindChillC", aggfunc="mean"
                        )
                        .sort_index()
                        .reset_index()
                    )

                    fig_hi_y = px.imshow(
                        [hi_y["HeatIndexC"].tolist()],
                        x=hi_y["year"],
                        y=[f"HeatIndex (max)"],
                        labels=dict(x="Year", y="", color="°C"),
                        title=f"Heat Index in {city} ({start_y}–{end_y})",
                        aspect="auto",
                    )
                    st.plotly_chart(fig_hi_y, use_container_width=True)

                    fig_wc_y = px.imshow(
                        [wc_y["WindChillC"].tolist()],
                        x=wc_y["year"],
                        y=[f"Wind Chill (avg)"],
                        labels=dict(x="Year", y="", color="°C"),
                        title=f"Wind Chill in {city} ({start_y}–{end_y})",
                        aspect="auto",
                    )
                    st.plotly_chart(fig_wc_y, use_container_width=True)

        st.subheader("7. Average Daytime vs. Nighttime Temperatures")

        day_night = (
            df.groupby("city")
            .agg(Daytime=("maxtempC", "mean"), Nighttime=("mintempC", "mean"))
            .reset_index()
            .sort_values("Daytime", ascending=False)
        )

        fig_dn = px.line(
            day_night,
            x="city",
            y=["Daytime", "Nighttime"],
            title="Average Day vs Night Temperatures by City",
            labels={"value": "Temperature (°C)", "city": "City", "variable": "Period"},
            markers=True,
        )

        fig_dn.update_traces(
            selector={"name": "Daytime"}, line_dash="dash", marker_symbol="circle"
        )
        fig_dn.update_traces(
            selector={"name": "Nighttime"}, line_dash="dot", marker_symbol="square"
        )

        fig_dn.update_xaxes(tickangle=45)

        fig_dn.update_yaxes(showgrid=True)

        st.plotly_chart(fig_dn, use_container_width=True)

        st.subheader("8. Yearly Temperature Fluctuation")

        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["date"].dt.year

        city_input = st.selectbox(
            "Select city for fluctuation plot", df["city"].unique(), key="fluct_city"
        )
        start_year_fl = st.number_input(
            "Start year",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=int(df["year"].min()),
            step=1,
            key="fluct_start",
        )
        end_year_fl = st.number_input(
            "End year",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=int(df["year"].max()),
            step=1,
            key="fluct_end",
        )

        mask = (
            (df["city"].str.strip().str.lower() == city_input.strip().lower())
            & (df["year"] >= start_year_fl)
            & (df["year"] <= end_year_fl)
        )
        df_city_years = df.loc[mask]

        if df_city_years.empty:
            st.warning(
                f"No data for {city_input} between {start_year_fl}–{end_year_fl}."
            )
        else:
            yearly_temp = df_city_years.groupby("year")["tempC"].mean().reset_index()

            fig_fluct = px.line(
                yearly_temp,
                x="year",
                y="tempC",
                title=f"Yearly Avg Temp in {city_input} ({start_year_fl}–{end_year_fl})",
                labels={"year": "Year", "tempC": "Average Temp (°C)"},
                markers=True,
            )
            fig_fluct.update_traces(line_color="green", marker=dict(color="green"))
            fig_fluct.update_xaxes(dtick=1)
            fig_fluct.update_yaxes(showgrid=True)

            st.plotly_chart(fig_fluct, use_container_width=True)

        st.subheader("9. Heat Stress Analysis")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["maxtempC"] = pd.to_numeric(df["maxtempC"], errors="coerce")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        sel_year = st.selectbox(
            "Select Year for Heat Stress Analysis", years, key="heatstress_year"
        )

        df_year = df[df["date"].dt.year == sel_year].copy()

        if df_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:
            daily = df_year.groupby(df_year["date"].dt.date, as_index=False).agg(
                mean_maxtemp=("maxtempC", "mean")
            )
            total_days = daily.shape[0]

            moderate = daily[
                (daily["mean_maxtemp"] >= 40) & (daily["mean_maxtemp"] <= 42)
            ].shape[0]
            high = daily[
                (daily["mean_maxtemp"] > 42) & (daily["mean_maxtemp"] <= 45)
            ].shape[0]
            extreme = daily[
                (daily["mean_maxtemp"] > 45) & (daily["mean_maxtemp"] <= 48)
            ].shape[0]
            none = total_days - (moderate + high + extreme)

            pie_df = pd.DataFrame(
                {
                    "Heat Stress Level": [
                        "Moderate (40–42°C)",
                        "High (42–45°C)",
                        "Extreme (45–48°C)",
                        "None",
                    ],
                    "Days": [moderate, high, extreme, none],
                }
            )

            fig_pie = px.pie(
                pie_df,
                names="Heat Stress Level",
                values="Days",
                title=f"Heat Stress Day Distribution in {sel_year} (n={total_days} days)",
                hole=0.0,
                template="plotly_dark",
            )
            fig_pie.update_traces(pull=[0.1, 0.1, 0.1, 0.0], textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("10. Heat Risk Bubble Chart During Summer Months")

        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["HeatIndexC"] = pd.to_numeric(df["HeatIndexC"], errors="coerce")
        df["maxtempC"] = pd.to_numeric(df["maxtempC"], errors="coerce")
        df["WindChillC"] = pd.to_numeric(df["WindChillC"], errors="coerce")

        city_risk = st.selectbox("Select city", df["city"].unique(), key="risk_city")
        year_risk = st.number_input(
            "Select year for summer analysis",
            min_value=int(df["year"].min()),
            max_value=int(df["year"].max()),
            value=int(df["year"].min()),
            step=1,
            key="risk_year",
        )

        summer_months = [5, 6, 7]
        mask = (
            (df["city"] == city_risk)
            & (df["year"] == year_risk)
            & (df["month"].isin(summer_months))
        )
        df_summer = df.loc[mask].copy()

        if df_summer.empty:
            st.warning(f"No data for {city_risk} in summer months of {year_risk}.")
        else:
            conditions = [
                df_summer["maxtempC"] < 40,
                (df_summer["maxtempC"] >= 40) & (df_summer["maxtempC"] < 42),
                (df_summer["maxtempC"] >= 42) & (df_summer["maxtempC"] < 45),
                df_summer["maxtempC"] >= 45,
            ]
            choices = [
                "No Heat Stress",
                "Moderate Heat Stress",
                "High Heat Stress",
                "Extreme Heat Stress",
            ]
            df_summer["Risk"] = np.select(conditions, choices, default="No Heat Stress")
            df_summer["month_name"] = df_summer["date"].dt.month_name()

            agg = df_summer.groupby(["city", "Risk", "month_name"], as_index=False)[
                "maxtempC"
            ].max()

            color_map = {
                "No Heat Stress": "cyan",
                "Moderate Heat Stress": "yellow",
                "High Heat Stress": "magenta",
                "Extreme Heat Stress": "red",
            }

            fig_risk = px.scatter(
                agg,
                x="city",
                y="maxtempC",
                size="maxtempC",
                color="Risk",
                color_discrete_map=color_map,
                hover_data=["month_name"],
                title=f"Summer Heat Risk in {city_risk} ({year_risk})",
                template="plotly_dark",
            )

            fig_risk.update_traces(marker=dict(line=dict(width=1, color="white")))

            st.plotly_chart(fig_risk, use_container_width=True)

        st.subheader("11. Peak Heat Hours on a Specific Date")

        peak_city = st.selectbox("Select city", df["city"].unique(), key="peak_city")
        peak_date = st.date_input(
            "Select date",
            min_value=df["date"].min().date(),
            max_value=df["date"].max().date(),
            key="peak_date",
        )

        df_peak = df[
            (df["city"] == peak_city) & (df["date"].dt.date == peak_date)
        ].copy()

        if df_peak.empty:
            st.warning(f"No data for {peak_city} on {peak_date}.")
        else:
            df_peak["hour"] = pd.to_datetime(
                df_peak["time"], format="%I:%M:%S %p"
            ).dt.hour

            fig_peak = px.line(
                df_peak,
                x="hour",
                y="FeelsLikeC",
                title=f"FeelsLike Temp by Hour in {peak_city} on {peak_date}",
                labels={"hour": "Hour", "FeelsLikeC": "FeelsLike (°C)"},
                markers=True,
            )
            fig_peak.update_traces(line_color="magenta", marker=dict(color="magenta"))
            fig_peak.update_yaxes(showgrid=True)

            st.plotly_chart(fig_peak, use_container_width=True)

            st.subheader("12. Seasonal Temperature Trends Over Years")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month

            seasons = {
                "Summer": [3, 4, 5],
                "Monsoon": [6, 7, 8, 9],
                "Autumn": [10, 11],
                "Winter": [12, 1, 2],
            }

            city_season = st.selectbox(
                "Select city", df["city"].unique(), key="season_city"
            )
            season_sel = st.selectbox(
                "Select season", list(seasons.keys()), key="season_name"
            )

            df_city = df[df["city"] == city_season].copy()
            if df_city.empty:
                st.warning(f"No data for {city_season}.")
            else:
                months = seasons[season_sel]
                df_season = df_city[df_city["month"].isin(months)]
                if df_season.empty:
                    st.warning(f"No {season_sel} data for {city_season}.")
                else:
                    season_stats = (
                        df_season.groupby("year")
                        .agg(avg_max=("maxtempC", "mean"), avg_min=("mintempC", "mean"))
                        .reset_index()
                        .dropna()
                    )

                    max_year = season_stats.loc[season_stats["avg_max"].idxmax()]
                    min_year = season_stats.loc[season_stats["avg_min"].idxmin()]

                    st.markdown(
                        f"- **Highest Avg Max Temp:** {int(max_year['year'])} ({max_year['avg_max']:.2f}°C)  \n"
                        f"- **Lowest Avg Min Temp:** {int(min_year['year'])} ({min_year['avg_min']:.2f}°C)"
                    )

                    fig = go.Figure()

                    fig.add_trace(
                        go.Scatter(
                            x=season_stats["year"],
                            y=season_stats["avg_max"],
                            mode="lines+markers",
                            name="Avg Max Temp",
                            line=dict(color="orange"),
                            marker=dict(symbol="circle"),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=season_stats["year"],
                            y=season_stats["avg_min"],
                            mode="lines+markers",
                            name="Avg Min Temp",
                            line=dict(color="blue"),
                            marker=dict(symbol="circle"),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[int(max_year["year"])],
                            y=[max_year["avg_max"]],
                            mode="markers",
                            name=f"Peak Max ({int(max_year['year'])})",
                            marker=dict(color="red", size=10),
                        )
                    )

                    fig.add_trace(
                        go.Scatter(
                            x=[int(min_year["year"])],
                            y=[min_year["avg_min"]],
                            mode="markers",
                            name=f"Lowest Min ({int(min_year['year'])})",
                            marker=dict(color="green", size=10),
                        )
                    )

                    fig.update_layout(
                        title=f"{city_season} – {season_sel} Temps Over Years",
                        xaxis_title="Year",
                        yaxis_title="Temperature (°C)",
                        xaxis=dict(dtick=1, showgrid=True),
                        yaxis=dict(showgrid=True),
                        legend=dict(title="Legend"),
                    )

                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("13. Correlation Matrix of Weather Factors")

            weather_factors = df[
                [
                    "maxtempC",
                    "mintempC",
                    "FeelsLikeC",
                    "HeatIndexC",
                    "WindChillC",
                    "windspeedKmph",
                    "humidity",
                ]
            ]

            correlation_matrix = weather_factors.corr()

            fig_corr = px.imshow(
                correlation_matrix,
                text_auto=".2f",
                aspect="auto",
                origin="lower",
                labels=dict(
                    x="Weather Factor", y="Weather Factor", color="Correlation"
                ),
                title="Correlation Matrix of Weather Factors",
            )

            st.plotly_chart(fig_corr, use_container_width=True)

    elif analysis_topic == "2. Visibility and Humidity Analysis":
        st.header("Visibility, Cloud Cover and Humidity Analysis : ")

        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce")
        df["cloudcover"] = pd.to_numeric(df["cloudcover"], errors="coerce")

        st.subheader("1. Correlation: Humidity vs Visibility")
        corr_hv = df[["humidity", "visibility"]].corr().loc["humidity", "visibility"]
        st.write(f"**Pearson correlation coefficient:** {corr_hv:.2f}")

        fig_hv = px.imshow(
            df[["humidity", "visibility"]].corr(),
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            labels={"x": "", "y": "", "color": "Correlation"},
            title="Humidity–Visibility Correlation",
        )
        fig_hv.update_layout(width=400, height=400)
        st.plotly_chart(fig_hv, use_container_width=True)

        st.subheader("2. Top 5 Cities by Visibility")
        city_visibility = (
            df.groupby("city")["visibility"]
            .mean()
            .reset_index()
            .rename(columns={"visibility": "Avg Visibility"})
        )
        top_high = city_visibility.nlargest(5, "Avg Visibility")
        top_low = city_visibility.nsmallest(5, "Avg Visibility")

        fig_high = px.bar(
            top_high,
            x="city",
            y="Avg Visibility",
            title="Top 5 Cities by Highest Visibility",
            labels={"city": "City"},
            color="Avg Visibility",
        )
        fig_high.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_high, use_container_width=True)

        fig_low = px.bar(
            top_low,
            x="city",
            y="Avg Visibility",
            title="Top 5 Cities by Lowest Visibility",
            labels={"city": "City"},
            color="Avg Visibility",
        )
        fig_low.update_layout(xaxis_tickangle=45, showlegend=False)
        st.plotly_chart(fig_low, use_container_width=True)

        st.subheader("3. Average Humidity by City")
        avg_humidity = (
            df.groupby("city")["humidity"]
            .mean()
            .reset_index()
            .rename(columns={"humidity": "Avg Humidity"})
        )
        fig_avg = px.bar(
            avg_humidity,
            x="city",
            y="Avg Humidity",
            title="Average Humidity by City",
            labels={"city": "City"},
        )
        fig_avg.update_layout(xaxis_tickangle=45, yaxis_title="Avg Humidity (%)")
        st.plotly_chart(fig_avg, use_container_width=True)

        st.subheader("4. Humidity & Cloud Cover Extremes by City")
        city_stats = df.groupby("city")[["humidity", "cloudcover"]].agg(["max", "min"])
        city_stats.columns = [
            "max_humidity",
            "min_humidity",
            "max_cloudcover",
            "min_cloudcover",
        ]
        city_stats = city_stats.reset_index()

        fig_extremes = px.bar(
            city_stats,
            x="city",
            y=["max_humidity", "min_humidity", "max_cloudcover", "min_cloudcover"],
            barmode="group",
            title="Max/Min Humidity & Cloud Cover by City",
            labels={"value": "Value", "variable": "Metric", "city": "City"},
        )
        fig_extremes.update_layout(
            xaxis_tickangle=45, yaxis_title="Value", legend_title="Metric"
        )
        st.plotly_chart(fig_extremes, use_container_width=True)

        st.subheader("5. Fog Formation Analysis")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce")
        df["cloudcover"] = pd.to_numeric(df["cloudcover"], errors="coerce")

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month_name()

        visibility_threshold = 5
        humidity_threshold = 70
        df["fog"] = (
            (df["visibility"] < visibility_threshold)
            & (df["humidity"] > humidity_threshold)
        ).astype(int)

        mode = st.radio("Analysis Mode", ["Single Month", "Year Range"], key="fog_mode")

        if mode == "Single Month":
            years = sorted(df["year"].dropna().unique().astype(int))
            year = st.selectbox("Select Year", years, key="fog_year")
            months = df[df["year"] == year]["month"].unique().tolist()
            month = st.selectbox("Select Month", months, key="fog_month")

            df_filtered = df[(df["year"] == year) & (df["month"] == month)]
            title_txt = f"{month}, {year}"
        else:
            years = sorted(df["year"].dropna().unique().astype(int))
            start_year = st.number_input(
                "Start Year",
                min_value=years[0],
                max_value=years[-1],
                value=years[0],
                key="fog_start",
            )
            end_year = st.number_input(
                "End Year",
                min_value=years[0],
                max_value=years[-1],
                value=years[-1],
                key="fog_end",
            )
            df_filtered = df[df["year"].between(start_year, end_year)]
            title_txt = f"{start_year}–{end_year}"

        if df_filtered.empty:
            st.warning(f"No data for {title_txt}.")
        else:
            city_avg = (
                df_filtered.groupby("city")[
                    ["cloudcover", "humidity", "visibility", "fog"]
                ]
                .mean()
                .reset_index()
            )
            city_avg["fog"] *= 100

            city_melted = city_avg.melt(
                id_vars="city",
                value_vars=["cloudcover", "humidity", "visibility", "fog"],
                var_name="Parameter",
                value_name="Percentage",
            )

            import plotly.express as px

            fig_fog = px.bar(
                city_melted,
                x="city",
                y="Percentage",
                color="Parameter",
                barmode="stack",
                title=f"Cloud Cover, Humidity, Visibility & Fog by City ({title_txt})",
                labels={"Percentage": "Percentage (%)", "city": "City"},
                color_discrete_map={
                    "cloudcover": "blue",
                    "humidity": "green",
                    "visibility": "orange",
                    "fog": "gray",
                },
                template="plotly_dark",
            )
            fig_fog.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_fog, use_container_width=True)

            st.markdown(
                "> **Insight:** Fog occurs when visibility drops below 5 km and humidity exceeds 70%. "
                "High humidity with low visibility leads to increased fog frequency, impacting travel and safety."
            )

        st.subheader("6. Visibility Trend by City Over Years")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        years = sorted(df["year"].dropna().unique().astype(int))
        start_year = st.number_input(
            "Start Year",
            min_value=years[0],
            max_value=years[-1],
            value=years[0],
            key="vis_start",
        )
        end_year = st.number_input(
            "End Year",
            min_value=years[0],
            max_value=years[-1],
            value=years[-1],
            key="vis_end",
        )

        df_vis = df[df["year"].between(start_year, end_year)].copy()
        if df_vis.empty:
            st.warning(f"No data between {start_year}–{end_year}.")
        else:
            vis_trend = (
                df_vis.groupby(["year", "city"], observed=True)["visibility"]
                .mean()
                .reset_index()
            )

            import plotly.express as px

            fig_vis = px.line(
                vis_trend,
                x="year",
                y="visibility",
                color="city",
                markers=True,
                title=f"Average Visibility ({start_year}–{end_year})",
                labels={
                    "visibility": "Avg Visibility (km)",
                    "year": "Year",
                    "city": "City",
                },
                template="plotly_dark",
            )
            st.plotly_chart(fig_vis, use_container_width=True)

        st.subheader("7. Airport Disruption Likelihood by City")

        year_input = st.selectbox("Select Year", years, key="aviation_year")
        df_air = df[df["year"] == year_input].copy()

        if df_air.empty:
            st.warning(f"No data for {year_input}.")
        else:
            df_air["disruption"] = (
                (df_air["visibility"] <= 5) & (df_air["cloudcover"] >= 70)
            ).astype(int)

            df_air_unique = df_air.drop_duplicates(subset=["city", "date"])

            disruptions = (
                df_air_unique.groupby("city")["disruption"]
                .sum()
                .reset_index(name="disruption_days")
                .sort_values("disruption_days", ascending=False)
            )

            fig_dis = px.bar(
                disruptions,
                x="disruption_days",
                y="city",
                orientation="h",
                title=f"Days of Potential Airport Disruption in {year_input}",
                labels={"disruption_days": "Disruption Days", "city": "City"},
                color="disruption_days",
                color_continuous_scale="OrRd",
                template="plotly_dark",
            )
            fig_dis.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig_dis, use_container_width=True)

            total_disp = int(disruptions["disruption_days"].sum())
            top_city = disruptions.iloc[0]["city"]
            top_days = int(disruptions.iloc[0]["disruption_days"])
            st.markdown(
                f"**Insight:** Across all cities in {year_input}, there were **{total_disp}** potential disruption days."
                f" The highest was in **{top_city}** with **{top_days}** days."
            )

        st.subheader("8. Airport Disruption Likelihood Across Cities Over Years")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["visibility"] = pd.to_numeric(df["visibility"], errors="coerce")
        df["cloudcover"] = pd.to_numeric(df["cloudcover"], errors="coerce")

        low_vis = 5
        high_cc = 70
        df["disruption_likelihood"] = (
            (df["visibility"] <= low_vis) & (df["cloudcover"] >= high_cc)
        ).astype(int)

        df_unique = df.drop_duplicates(subset=["city", "date"])

        disruption_data = (
            df_unique.groupby(["city", "year"], observed=True)["disruption_likelihood"]
            .sum()
            .reset_index()
            .rename(columns={"disruption_likelihood": "disruption_days"})
        )

        disruption_summary = (
            disruption_data.groupby("city")["disruption_days"]
            .sum()
            .reset_index()
            .rename(columns={"disruption_days": "Total Disruptions"})
            .sort_values("Total Disruptions", ascending=False)
        )

        fig_disrupt = px.bar(
            disruption_data,
            x="year",
            y="disruption_days",
            color="city",
            barmode="group",
            title="Airport Disruption Days by City & Year",
            labels={
                "disruption_days": "Days with Disruptions",
                "year": "Year",
                "city": "City",
            },
            template="plotly_dark",
        )
        fig_disrupt.update_xaxes(tickmode="linear", dtick=1)
        st.plotly_chart(fig_disrupt, use_container_width=True)

        st.markdown(
            "> **Insight:** Low visibility (≤5 km) combined with high cloud cover (≥70%) "
            "can lead to flight delays or cancellations, with certain cities experiencing "
            "more frequent disruption days over the years."
        )

    elif analysis_topic == "3. Wind & Gust Analysis":
        st.header("Wind & Gust Speed Analysis")

        st.subheader("1. Wind Speed Over the Months")

        selected_city = st.selectbox(
            "Select City", df["city"].unique(), key="wind_city"
        )

        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month_name()
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

        df_city = df[df["city"] == selected_city]

        df_wind = (
            df_city.groupby(["year", "month"], observed=False)
            .agg(
                avg_wind=("windspeedKmph", "mean"),
                min_wind=("windspeedKmph", "min"),
                max_wind=("windspeedKmph", "max"),
            )
            .reset_index()
        )

        years = sorted(df_wind["year"].dropna().unique().astype(int))
        init_year = years[0]
        df_init = df_wind[df_wind["year"] == init_year]

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=df_init["month"],
                    y=df_init["avg_wind"],
                    mode="lines+markers",
                    name="Average",
                ),
                go.Scatter(
                    x=df_init["month"],
                    y=df_init["min_wind"],
                    mode="lines+markers",
                    name="Minimum",
                ),
                go.Scatter(
                    x=df_init["month"],
                    y=df_init["max_wind"],
                    mode="lines+markers",
                    name="Maximum",
                ),
            ]
        )

        steps = []
        for yr in years:
            df_year = df_wind[df_wind["year"] == yr]
            steps.append(
                dict(
                    label=str(yr),
                    method="update",
                    args=[
                        {
                            "y": [
                                df_year["avg_wind"].tolist(),
                                df_year["min_wind"].tolist(),
                                df_year["max_wind"].tolist(),
                            ]
                        },
                        {"title": f"Wind Speed in {selected_city} – {yr}"},
                    ],
                )
            )

        fig.update_layout(
            sliders=[
                dict(
                    active=0,
                    currentvalue={"prefix": "Year: "},
                    pad={"t": 50},
                    steps=steps,
                )
            ],
            title=f"Wind Speed in {selected_city} – {init_year}",
            xaxis_title="Month",
            yaxis_title="Wind Speed (Kmph)",
            template="plotly_dark",
            height=600,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("2. Pressure & Wind Correlation Heatmap")

        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df["windspeedKmph"] = pd.to_numeric(df["windspeedKmph"], errors="coerce")
        df["WindGustKmph"] = pd.to_numeric(df["WindGustKmph"], errors="coerce")

        corr_matrix = df[["pressure", "windspeedKmph", "WindGustKmph"]].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu",
            title="Pressure vs. Wind Speed & Gust Correlation",
            template="plotly_dark",
        )
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("3. Wind Direction Distribution by Year")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        sel_year = st.selectbox("Select Year", years, key="wd_year")

        df_year = df[df["date"].dt.year == sel_year].copy()

        if df_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:

            def categorize_wind_direction(deg):
                if (deg >= 337.5) or (deg < 22.5):
                    return "North"
                elif deg < 67.5:
                    return "Northeast"
                elif deg < 112.5:
                    return "East"
                elif deg < 157.5:
                    return "Southeast"
                elif deg < 202.5:
                    return "South"
                elif deg < 247.5:
                    return "Southwest"
                elif deg < 292.5:
                    return "West"
                else:
                    return "Northwest"

            df_year["dir_sector"] = df_year["winddirDegree"].apply(
                categorize_wind_direction
            )

        daily_dir = (
            df_year.groupby(df_year["date"].dt.date)["dir_sector"]
            .agg(lambda x: x.value_counts().idxmax())
            .reset_index(name="dominant_dir")
        )

        total_days = daily_dir.shape[0]

        pie_df = (
            daily_dir["dominant_dir"]
            .value_counts()
            .rename_axis("Wind Direction")
            .reset_index(name="Days")
        )

        fig_wd = px.pie(
            pie_df,
            names="Wind Direction",
            values="Days",
            title=f"Dominant Wind Direction by Day in {sel_year} (n={total_days})",
            hole=0.3,
            template="plotly_dark",
        )
        fig_wd.update_traces(textinfo="percent+label")
        st.plotly_chart(fig_wd, use_container_width=True)

        st.subheader("4. Wind Direction Funnel: Days Count vs Avg Windspeed (%)")

        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

        available_years = sorted(df["date"].dt.year.dropna().unique().astype(int))

        selected_year = st.selectbox(
            "Select Year for Wind Funnel Analysis",
            available_years,
            key="wind_funnel_year",
        )

        df_year = df[df["date"].dt.year == selected_year].copy()

        if df_year.empty:
            st.warning(f"No data available for {selected_year}")
        else:
            df_year.loc[:, "winddirDegree_binned"] = pd.cut(
                df_year["winddirDegree"],
                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                labels=[
                    "North",
                    "North-East",
                    "East",
                    "South-East",
                    "South",
                    "South-West",
                    "West",
                    "North-West",
                ],
                include_lowest=True,
            )

            df_wind_freq = (
                df_year.groupby(["date", "winddirDegree_binned"], observed=True)
                .size()
                .reset_index(name="count")
            )

            df_dominant_wind = df_wind_freq.loc[
                df_wind_freq.groupby("date")["count"].idxmax()
            ]

            df_grouped = (
                df_dominant_wind.groupby("winddirDegree_binned", observed=True)
                .agg(days_count=("date", "nunique"), total_windspeed=("count", "sum"))
                .reset_index()
            )

            total_windspeed_sum = df_grouped["total_windspeed"].sum()
            df_grouped["windspeed_percentage"] = (
                df_grouped["total_windspeed"] / total_windspeed_sum
            ) * 100

            df_count = df_grouped[["winddirDegree_binned", "days_count"]].copy()
            df_count["type"] = "Days Count"

            df_avg_wind = df_grouped[
                ["winddirDegree_binned", "windspeed_percentage"]
            ].copy()
            df_avg_wind.columns = ["winddirDegree_binned", "days_count"]
            df_avg_wind["type"] = "Avg Windspeed (%)"

            df_funnel = pd.concat([df_count, df_avg_wind], axis=0)

            fig_funnel = px.funnel(
                df_funnel,
                x="days_count",
                y="winddirDegree_binned",
                color="type",
                title=f"Wind Direction Funnel for {selected_year}: Days Count vs Avg Windspeed (%)",
                labels={
                    "days_count": "Value",
                    "winddirDegree_binned": "Wind Direction",
                },
                template="plotly_dark",
            )

            st.plotly_chart(fig_funnel, use_container_width=True)

        st.subheader("5. Average Windspeed by Wind Direction & Month")

        df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")

        available_cities = sorted(df["city"].dropna().unique())
        available_years = sorted(df["date"].dt.year.dropna().unique().astype(int))

        selected_city = st.selectbox(
            "Select City", available_cities, key="avg_wind_city"
        )
        selected_year = st.selectbox(
            "Select Year", available_years, key="avg_wind_year"
        )

        df_city_year = df[
            (df["city"] == selected_city) & (df["date"].dt.year == selected_year)
        ].copy()

        if df_city_year.empty:
            st.warning(
                f"No windspeed data available for {selected_city} in {selected_year}"
            )
        else:
            df_city_year["month"] = df_city_year["date"].dt.month

            df_city_year["winddirDegree_binned"] = pd.cut(
                df_city_year["winddirDegree"],
                bins=[0, 45, 90, 135, 180, 225, 270, 315, 360],
                labels=[
                    "North",
                    "North-East",
                    "East",
                    "South-East",
                    "South",
                    "South-West",
                    "West",
                    "North-West",
                ],
                include_lowest=True,
            )

            df_grouped = (
                df_city_year.groupby(["month", "winddirDegree_binned"], observed=True)
                .agg(avg_windspeed=("windspeedKmph", "mean"))
                .reset_index()
            )

            df_grouped["avg_windspeed"] = df_grouped["avg_windspeed"].fillna(0)

            fig_avg_wind = px.line(
                df_grouped,
                x="month",
                y="avg_windspeed",
                color="winddirDegree_binned",
                markers=True,
                height=700,
                title=f"Average Windspeed for {selected_city} in {selected_year} by Wind Direction & Month",
                labels={
                    "avg_windspeed": "Average Windspeed (Kmph)",
                    "winddirDegree_binned": "Wind Direction",
                    "month": "Month",
                },
                category_orders={
                    "winddirDegree_binned": [
                        "North",
                        "North-East",
                        "East",
                        "South-East",
                        "South",
                        "South-West",
                        "West",
                        "North-West",
                    ]
                },
            )

            fig_avg_wind.update_traces(
                marker=dict(size=8, line=dict(width=1)), line=dict(width=2)
            )

            fig_avg_wind.update_layout(
                xaxis=dict(
                    tickmode="array",
                    tickvals=list(range(1, 13)),
                    ticktext=[
                        "Jan",
                        "Feb",
                        "Mar",
                        "Apr",
                        "May",
                        "Jun",
                        "Jul",
                        "Aug",
                        "Sep",
                        "Oct",
                        "Nov",
                        "Dec",
                    ],
                ),
                template="plotly_dark",
            )

            st.plotly_chart(fig_avg_wind, use_container_width=True)

    elif analysis_topic == "4. Precipitation Overview":
        st.header("Precipitation Overview")

        df["precipMM"] = pd.to_numeric(df["precipMM"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        st.subheader("1. Precipitation Summary Statistics")
        precip_stats = df["precipMM"].describe()
        st.write(precip_stats)

        st.subheader("2. Days with Zero vs Non‑Zero Precipitation")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        sel_year = st.selectbox("Select Year", years, key="precip_year")

        df_year = df[df["date"].dt.year == sel_year].copy()

        if df_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:
            daily_precip = df_year.groupby(df_year["date"].dt.date, as_index=False).agg(
                avg_precip=("precipMM", "mean")
            )

            total_days = daily_precip.shape[0]
            zero_count = int((daily_precip["avg_precip"] == 0).sum())
            nonzero_count = int((daily_precip["avg_precip"] > 0).sum())

            pie_df = pd.DataFrame(
                {
                    "Category": ["Zero Precipitation", "Non‑Zero Precipitation"],
                    "Count": [zero_count, nonzero_count],
                }
            )

            fig_pie = px.pie(
                pie_df,
                names="Category",
                values="Count",
                title=f"Zero vs Non‑Zero Precipitation Days in {sel_year} (n={total_days})",
                hole=0.4,
                template="plotly_dark",
            )
            fig_pie.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_pie, use_container_width=True)

        st.subheader("3. Daily Precipitation Over Time")

        max_row = df.loc[df["precipMM"].idxmax()]
        min_row = df.loc[df["precipMM"].idxmin()]

        fig_ts = px.line(
            df.sort_values("date"),
            x="date",
            y="precipMM",
            title="Time Series of Daily Precipitation",
            labels={"date": "Date", "precipMM": "Precipitation (mm)"},
        )
        fig_ts.add_trace(
            go.Scatter(
                x=[max_row["date"]],
                y=[max_row["precipMM"]],
                mode="markers+text",
                marker=dict(color="red", size=10),
                text=["Max"],
                textposition="top center",
                name="Maximum",
            )
        )
        fig_ts.add_trace(
            go.Scatter(
                x=[min_row["date"]],
                y=[min_row["precipMM"]],
                mode="markers+text",
                marker=dict(color="green", size=10),
                text=["Min"],
                textposition="bottom center",
                name="Minimum",
            )
        )
        fig_ts.update_yaxes(showgrid=True)
        st.plotly_chart(fig_ts, use_container_width=True)

        st.subheader("4. Average Precipitation by City")

        avg_precip_by_city = (
            df.groupby("city")["precipMM"]
            .mean()
            .reset_index()
            .rename(columns={"city": "City", "precipMM": "Avg Precipitation (mm)"})
        )

        fig_bar = px.bar(
            avg_precip_by_city,
            x="City",
            y="Avg Precipitation (mm)",
            title="Average Daily Precipitation by City",
            labels={"Avg Precipitation (mm)": "Avg Precipitation (mm)"},
        )
        fig_bar.update_xaxes(tickangle=45)
        fig_bar.update_yaxes(showgrid=True)
        st.plotly_chart(fig_bar, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["precipMM"] = pd.to_numeric(df["precipMM"], errors="coerce")

        st.subheader("5. Max Temp vs. Precipitation for a City")
        city_rel = st.selectbox(
            "Select city", df["city"].unique(), key="precip_rel_city"
        )
        df_rel = df[df["city"] == city_rel].dropna(subset=["maxtempC", "precipMM"])

        if df_rel.empty:
            st.warning(f"No data for {city_rel}.")
        else:
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=df_rel["maxtempC"],
                    y=df_rel["precipMM"],
                    mode="markers",
                    marker=dict(size=6, opacity=0.6),
                    name="Data points",
                    hovertemplate="Max Temp: %{x}°C<br>Precip: %{y} mm",
                )
            )

            x = df_rel["maxtempC"].to_numpy()
            y = df_rel["precipMM"].to_numpy()
            slope, intercept = np.polyfit(x, y, 1)
            x_fit = np.array([x.min(), x.max()])
            y_fit = slope * x_fit + intercept

            fig.add_trace(
                go.Scatter(
                    x=x_fit,
                    y=y_fit,
                    mode="lines",
                    line=dict(color="red"),
                    name="Regression line",
                    hoverinfo="skip",
                )
            )

            fig.update_layout(
                title=f"Max Temp vs. Precipitation in {city_rel}",
                xaxis_title="Max Temperature (°C)",
                yaxis_title="Precipitation (mm)",
                template="plotly_white",
            )

            st.plotly_chart(fig, use_container_width=True)

            corr_val = df_rel[["maxtempC", "precipMM"]].corr().iloc[0, 1]
            if corr_val > 0:
                st.write(
                    f"Positive correlation (**{corr_val:.2f}**) → higher temps may bring more rain."
                )
            elif corr_val < 0:
                st.write(
                    f"Negative correlation (**{corr_val:.2f}**) → higher temps may bring less rain."
                )
            else:
                st.write(f"No significant correlation (**{corr_val:.2f}**).")

        st.subheader("6. City‑wise Precipitation Over Years")
        df_yearly = df.drop_duplicates().copy()
        df_yearly["year"] = df_yearly["date"].dt.year
        city_year_data = (
            df_yearly.groupby(["city", "year"])["precipMM"].sum().reset_index()
        )
        years = sorted(city_year_data["year"].unique())

        fig_precip = go.Figure()
        for city in city_year_data["city"].unique():
            cdf = city_year_data[city_year_data["city"] == city]
            fig_precip.add_trace(go.Bar(x=cdf["year"], y=cdf["precipMM"], name=city))
        fig_precip.update_layout(
            title="Precipitation by City Over Years",
            xaxis_title="Year",
            yaxis_title="Total Precipitation (mm)",
            barmode="group",
            xaxis=dict(tickmode="array", tickvals=years),
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig_precip, use_container_width=True)

        st.subheader("7. Precipitation Distribution by Season (2009–2019)")

        def get_season(date):
            m = date.month
            if m in [3, 4, 5]:
                return "Summer"
            elif m in [6, 7, 8, 9]:
                return "Monsoon"
            elif m in [10, 11]:
                return "Autumn"
            else:
                return "Winter"

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["season"] = df["date"].apply(get_season)

        seasonal_precip = df.groupby(["year", "season"])["precipMM"].sum().reset_index()

        total_by_year = seasonal_precip.groupby("year")["precipMM"].transform("sum")
        seasonal_precip["percentage"] = (
            seasonal_precip["precipMM"] / total_by_year * 100
        )

        years = sorted(seasonal_precip["year"].dropna().unique().astype(int))
        colors = ["#2c3e50", "#e74c3c", "#3498db", "#27ae60"]

        fig = go.Figure()

        for i, yr in enumerate(years):
            data_year = seasonal_precip[seasonal_precip["year"] == yr]
            fig.add_trace(
                go.Pie(
                    labels=data_year["season"],
                    values=data_year["percentage"],
                    name=str(yr),
                    marker=dict(colors=colors),
                    textinfo="percent+label",
                    hole=0.4,
                    visible=(i == 0),
                )
            )

        updatemenu = [
            dict(
                buttons=[
                    dict(
                        label=str(yr),
                        method="update",
                        args=[
                            {"visible": [j == i for j in range(len(years))]},
                            {"title": f"Precipitation by Season in {yr}"},
                        ],
                    )
                    for i, yr in enumerate(years)
                ],
                direction="down",
                x=0.5,
                y=1.15,
                xanchor="center",
                yanchor="top",
            )
        ]

        fig.update_layout(
            title_text=f"Precipitation by Season in {years[0]}",
            updatemenus=updatemenu,
            template="plotly_dark",
            showlegend=False,
            height=500,
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("8. Flash Flood Risk (≥50 mm/day)")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year

        daily_precip = (
            df.groupby(["city", "date", "year"])["precipMM"].sum().reset_index()
        )

        flash_risk = daily_precip[daily_precip["precipMM"] >= 50]

        if flash_risk.empty:
            st.write("No flash flood risk records (≥50 mm in a single day).")
        else:
            bar_df = (
                flash_risk.groupby("city")["precipMM"]
                .sum()
                .reset_index()
                .rename(columns={"precipMM": "Total Precipitation (mm)"})
            )
            fig_bar = px.bar(
                bar_df,
                x="city",
                y="Total Precipitation (mm)",
                title="Cities with Flash Flood Risk",
                labels={"city": "City"},
                color="city",
            )
            fig_bar.update_layout(xaxis_tickangle=90, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

            fig_scatter = px.scatter(
                flash_risk,
                x="date",
                y="precipMM",
                color="city",
                symbol="year",
                title="Flash Flood Risk Over Time",
                labels={"date": "Date", "precipMM": "Precipitation (mm)"},
            )
            fig_scatter.update_xaxes(
                range=["2009-01-01", "2019-12-31"], tickformat="%Y", tickangle=45
            )
            st.plotly_chart(fig_scatter, use_container_width=True)

            st.subheader("Sample Flash Flood Risk Records")
            st.dataframe(
                flash_risk[["city", "date", "precipMM"]]
                .sample(10, random_state=1)
                .reset_index(drop=True)
            )
            st.write(f"**Total flash flood risk records:** {len(flash_risk)}")

        st.subheader("9. Correlation Heatmap: PrecipMM vs Other Factors")

        relevant_columns = [
            "precipMM",
            "maxtempC",
            "mintempC",
            "humidity",
            "tempC",
            "windspeedKmph",
            "cloudcover",
            "pressure",
            "DewPointC",
        ]

        corr_matrix = df[relevant_columns].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            labels={"x": "", "y": "", "color": "Correlation"},
            x=relevant_columns,
            y=relevant_columns,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            title="Correlation Heatmap: Precipitation vs Other Factors",
        )

        fig_corr.update_layout(width=700, height=700, xaxis_side="top")

        st.plotly_chart(fig_corr, use_container_width=True)

    elif analysis_topic == "5. UV Index Variations":
        st.header("UV Index Variations")

        df["uvIndex"] = pd.to_numeric(df["uvIndex"], errors="coerce")

        st.subheader("1. Cities with Maximum UV Index :")
        max_uv = (
            df.groupby("city")["uvIndex"]
            .max()
            .reset_index()
            .sort_values("uvIndex", ascending=False)
        )

        fig_uv_bar = px.bar(
            max_uv.head(10),
            x="city",
            y="uvIndex",
            text="uvIndex",
            color="uvIndex",
            color_continuous_scale="Reds",
            title="Top 10 Cities by Maximum UV Index",
            labels={"uvIndex": "Max UV Index", "city": "City"},
            template="plotly_dark",
        )
        fig_uv_bar.update_layout(
            xaxis_tickangle=-45,
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_uv_bar, use_container_width=True)

        st.subheader("2. UV Risk Levels – One Category per Day by Year")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        sel_year = st.selectbox("Select Year", years, key="uv_year")

        df_year = df[df["date"].dt.year == sel_year].copy()

        if df_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:

            def uv_category(u):
                if u >= 11:
                    return "11+ (Extremely High)"
                if u >= 8:
                    return "8-10 (Very High)"
                if u >= 6:
                    return "6-7 (High)"
                if u >= 3:
                    return "3-5 (Medium)"
                if u >= 1:
                    return "1-2 (Low)"
                return "Unknown"

            df_year["uv_category"] = df_year["uvIndex"].apply(uv_category)
            df_year["day"] = df_year["date"].dt.date

            daily_uv = (
                df_year.groupby("day")["uv_category"]
                .agg(lambda x: x.value_counts().idxmax())
                .reset_index(name="daily_category")
            )
            total_days = daily_uv.shape[0]

            counts = daily_uv["daily_category"].value_counts()
            uv_counts = pd.DataFrame(
                {"UV Category": counts.index, "Count": counts.values}
            )

            daily_max_uv = (
                df_year.groupby("day")["uvIndex"].max().reset_index(name="max_uv")
            )
            merged = daily_uv.merge(daily_max_uv, on="day")
            avg = merged.groupby("daily_category")["max_uv"].mean()
            uv_avg = pd.DataFrame(
                {"UV Category": avg.index, "Avg UV Index": avg.values}
            )

            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig_uv = make_subplots(
                rows=1,
                cols=2,
                specs=[[{"type": "domain"}, {"type": "domain"}]],
                subplot_titles=["Days per UV Category", "Avg UV Index per Category"],
            )

            fig_uv.add_trace(
                go.Pie(
                    labels=uv_counts["UV Category"],
                    values=uv_counts["Count"],
                    name="Counts",
                ),
                row=1,
                col=1,
            )
            fig_uv.add_trace(
                go.Pie(
                    labels=uv_avg["UV Category"],
                    values=uv_avg["Avg UV Index"],
                    name="Average UV",
                ),
                row=1,
                col=2,
            )

            fig_uv.update_layout(
                template="plotly_dark",
                title_text=f"UV Levels in {sel_year} (n={total_days} days)",
                annotations=[
                    dict(text="Counts", x=0.18, y=0.5, showarrow=False),
                    dict(text="Avg UV", x=0.82, y=0.5, showarrow=False),
                ],
                showlegend=False,
            )

            st.plotly_chart(fig_uv, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["uvIndex"] = pd.to_numeric(df["uvIndex"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.strftime("%b")

        def uv_category(u):
            if u >= 11:
                return "Extremely High (11+)"
            elif 8 <= u <= 10:
                return "Very High (8-10)"
            elif 6 <= u <= 7:
                return "High (6-7)"
            elif 3 <= u <= 5:
                return "Medium (3-5)"
            else:
                return "Low (1-2)"

        df["uv_category"] = df["uvIndex"].apply(uv_category)

        import plotly.graph_objects as go

        st.subheader("3. UV Index Levels by City & Month")

        fig3 = go.Figure()
        years = sorted(df["year"].dropna().unique().astype(int))
        months = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        for year in years:
            dfy = df[df["year"] == year]
            for city in dfy["city"].unique():
                series = (
                    dfy[dfy["city"] == city]
                    .groupby("month")["uvIndex"]
                    .mean()
                    .reindex(months)
                )
                fig3.add_trace(
                    go.Scatter(
                        x=series.index,
                        y=series.values,
                        mode="lines+markers",
                        name=f"{city} ({year})",
                        visible=False,
                    )
                )
        for trace in fig3.data:
            if f"({years[0]})" in trace.name:
                trace.visible = True

        buttons = []
        for yr in years:
            vis = [f"({yr})" in t.name for t in fig3.data]
            buttons.append(
                dict(label=str(yr), method="update", args=[{"visible": vis}])
            )
        fig3.update_layout(
            updatemenus=[dict(active=0, buttons=buttons, x=1.1, y=1)],
            title="Average UV Index by City & Month",
            xaxis_title="Month",
            yaxis_title="Avg UV Index",
            template="plotly_dark",
            height=500,
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.subheader(
            "4. Count of Days by UV Index Risk Levels for a Particular Year : "
        )

        year_sel = st.selectbox("Select Year", years, key="uv_count_year")
        dfy2 = df[df["year"] == year_sel].drop_duplicates(subset=["city", "date"])

        counts = (
            dfy2.groupby(["city", "uv_category"]).size().reset_index(name="day_count")
        )

        fig4 = go.Figure()
        for city in counts["city"].unique():
            cdf = counts[counts["city"] == city]
            fig4.add_trace(go.Bar(x=cdf["uv_category"], y=cdf["day_count"], name=city))
        fig4.update_layout(
            barmode="group",
            title=f"Count of Days by UV Level in {year_sel}",
            xaxis_title="UV Risk Level",
            yaxis_title="Days Count",
            template="plotly_dark",
            xaxis=dict(tickangle=-45),
            height=500,
        )
        st.plotly_chart(fig4, use_container_width=True)

    elif analysis_topic == "6. Cloud Cover & Solar Potential":
        st.header("Cloud Cover & Solar Potential")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month
        df["month_name"] = df["date"].dt.month_name()

        st.subheader("1. Monthly Average Cloud Cover")
        monthly_cloud = (
            df.groupby("month_name", sort=False)["cloudcover"]
            .mean()
            .reindex(
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ]
            )
            .reset_index()
            .rename(columns={"month_name": "Month", "cloudcover": "Avg Cloud Cover"})
        )

        fig_mc = px.bar(
            monthly_cloud,
            x="Month",
            y="Avg Cloud Cover",
            title="Average Cloud Cover by Month",
            labels={"Avg Cloud Cover": "Cloud Cover (%)"},
        )
        fig_mc.update_layout(xaxis_tickangle=45)
        st.plotly_chart(fig_mc, use_container_width=True)

        st.subheader("2. Avg Sun Hours & Cloud Cover by City")
        city_summary = (
            df.groupby("city")[["sunHour", "cloudcover"]]
            .mean()
            .reset_index()
            .rename(
                columns={
                    "city": "City",
                    "sunHour": "Avg Sun Hours",
                    "cloudcover": "Avg Cloud Cover",
                }
            )
        )

        fig = px.line(
            city_summary,
            x="City",
            y=["Avg Sun Hours", "Avg Cloud Cover"],
            title="Average Sun Hours & Cloud Cover by City",
            markers=True,
            labels={"value": "Average", "variable": "Metric", "City": "City"},
        )

        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(showgrid=True)

        st.plotly_chart(fig, use_container_width=True)

        st.subheader("3. Correlation: Sun Hours vs. Cloud Cover")

        corr_matrix = df[["sunHour", "cloudcover"]].corr()

        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            title="Pearson Correlation Matrix",
            labels={"x": "", "y": "", "color": "Correlation"},
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )

        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("4. Top 5 Cities by Average Sun Hours")
        top_sun = city_summary.nlargest(5, "Avg Sun Hours").reset_index(drop=True)
        st.dataframe(top_sun)

        st.subheader("5. Solar Availability by City in a Year")
        year_input = st.number_input(
            "Select Year",
            min_value=int(df["date"].dt.year.min()),
            max_value=int(df["date"].dt.year.max()),
            value=int(df["date"].dt.year.min()),
            step=1,
        )
        df_year = df[df["date"].dt.year == year_input]

        if df_year.empty:
            st.warning(f"No data for {year_input}.")
        else:
            solar_by_city = (
                df_year.groupby("city")[["sunHour", "cloudcover"]]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "city": "City",
                        "sunHour": "Avg Sun Hours",
                        "cloudcover": "Avg Cloud Cover",
                    }
                )
            )
            st.dataframe(solar_by_city)

            fig_sc = px.scatter(
                solar_by_city,
                x="Avg Sun Hours",
                y="Avg Cloud Cover",
                text="City",
                labels={
                    "Avg Sun Hours": "Avg Sun Hours",
                    "Avg Cloud Cover": "Avg Cloud Cover (%)",
                },
                title=f"Sun Hours vs. Cloud Cover ({year_input})",
            )
            fig_sc.update_traces(textposition="top center")
            fig_sc.update_layout(
                yaxis=dict(tickformat=".1f"), xaxis=dict(tickformat=".1f")
            )
            st.plotly_chart(fig_sc, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date"].dt.month

        st.subheader("6. Cloud Cover Impact on Solar Availability")

        city_cc = st.selectbox(
            "Select city for cloud cover analysis", df["city"].unique(), key="cc_city"
        )
        city_data = df[df["city"] == city_cc].copy()

        if city_data.empty:
            st.warning(f"No data for {city_cc}.")
        else:
            city_data["cloudcover_bins"] = pd.cut(
                city_data["cloudcover"],
                bins=[0, 20, 50, 80, 100],
                labels=[
                    "Low (0–20%)",
                    "Medium (21–50%)",
                    "High (51–80%)",
                    "Very High (81–100%)",
                ],
            )
            cc_analysis = (
                city_data.groupby("cloudcover_bins", observed=True)["sunHour"]
                .mean()
                .reset_index()
                .rename(columns={"sunHour": "Avg Sun Hours"})
            )

            fig_cc = px.bar(
                cc_analysis,
                x="cloudcover_bins",
                y="Avg Sun Hours",
                title=f"Avg Sun Hours by Cloud Cover in {city_cc}",
                labels={
                    "cloudcover_bins": "Cloud Cover Level",
                    "Avg Sun Hours": "Avg Sun Hours",
                },
            )
            fig_cc.update_layout(xaxis_tickangle=45, yaxis_title="Avg Sun Hours")
            st.plotly_chart(fig_cc, use_container_width=True)

        st.subheader("7. Solar Availability Across Cities in a Year")

        year_solar = st.number_input(
            "Select year",
            min_value=int(df["date"].dt.year.min()),
            max_value=int(df["date"].dt.year.max()),
            value=int(df["date"].dt.year.min()),
            step=1,
            key="solar_year2",
        )
        df_year = df[df["date"].dt.year == year_solar]

        if df_year.empty:
            st.warning(f"No data for {year_solar}.")
        else:
            city_solar = (
                df_year.groupby("city")[["sunHour", "cloudcover"]]
                .mean()
                .reset_index()
                .rename(
                    columns={
                        "city": "City",
                        "sunHour": "Avg Sun Hours",
                        "cloudcover": "Avg Cloud Cover",
                    }
                )
            )

            fig_se = make_subplots(specs=[[{"secondary_y": True}]])

            fig_se.add_trace(
                go.Scatter(
                    x=city_solar["City"],
                    y=city_solar["Avg Sun Hours"],
                    mode="lines+markers",
                    name="Avg Sun Hours",
                    marker=dict(symbol="circle", size=8),
                    line=dict(color="orange"),
                ),
                secondary_y=False,
            )

            fig_se.add_trace(
                go.Scatter(
                    x=city_solar["City"],
                    y=city_solar["Avg Cloud Cover"],
                    mode="lines+markers",
                    name="Avg Cloud Cover",
                    marker=dict(symbol="square", size=8),
                    line=dict(color="blue"),
                ),
                secondary_y=True,
            )

            fig_se.update_layout(
                title_text=f"Solar Availability by City in {year_solar}",
                xaxis=dict(title="City", tickangle=45),
                legend=dict(
                    orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5
                ),
            )
            fig_se.update_yaxes(title_text="Avg Sun Hours", secondary_y=False)
            fig_se.update_yaxes(title_text="Avg Cloud Cover (%)", secondary_y=True)

            st.plotly_chart(fig_se, use_container_width=True)

        st.subheader("8. Monthly & Yearly Trends for a City")

        city_trend = st.selectbox("Select city", df["city"].unique(), key="trend_city")
        city_df = df[df["city"] == city_trend].copy()

        if city_df.empty:
            st.warning(f"No data for {city_trend}.")
        else:
            city_df["date"] = pd.to_datetime(city_df["date"], errors="coerce")
            city_df["month"] = city_df["date"].dt.month
            city_df["year"] = city_df["date"].dt.year

            monthly = (
                city_df.groupby("month")[["sunHour", "cloudcover"]].mean().reset_index()
            )
            yearly = (
                city_df.groupby("year")[["sunHour", "cloudcover"]].mean().reset_index()
            )

            fig_monthly = px.line(
                monthly,
                x="month",
                y=["sunHour", "cloudcover"],
                markers=True,
                title=f"Monthly Trends in {city_trend}",
                labels={"month": "Month", "value": "Average", "variable": "Metric"},
            )
            fig_monthly.for_each_trace(
                lambda trace: trace.update(
                    name=(
                        "Avg Sun Hours"
                        if trace.name == "sunHour"
                        else "Avg Cloud Cover (%)"
                    ),
                    line=dict(color=("yellow" if trace.name == "sunHour" else "blue")),
                    marker=dict(
                        symbol=("circle" if trace.name == "sunHour" else "square")
                    ),
                )
            )
            month_abbr = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            fig_monthly.update_xaxes(
                tickmode="array", tickvals=list(range(1, 13)), ticktext=month_abbr
            )
            fig_monthly.update_yaxes(showgrid=True)
            st.plotly_chart(fig_monthly, use_container_width=True)

            fig_yearly = px.line(
                yearly,
                x="year",
                y=["sunHour", "cloudcover"],
                markers=True,
                title=f"Yearly Trends in {city_trend}",
                labels={"year": "Year", "value": "Average", "variable": "Metric"},
            )
            fig_yearly.for_each_trace(
                lambda trace: trace.update(
                    name=(
                        "Avg Sun Hours"
                        if trace.name == "sunHour"
                        else "Avg Cloud Cover (%)"
                    ),
                    line=dict(color=("yellow" if trace.name == "sunHour" else "blue")),
                    marker=dict(
                        symbol=("circle" if trace.name == "sunHour" else "square")
                    ),
                )
            )
            fig_yearly.update_xaxes(dtick=1)
            fig_yearly.update_yaxes(showgrid=True)
            st.plotly_chart(fig_yearly, use_container_width=True)

    elif analysis_topic == "7. Pressure Insights":
        st.header("Pressure Analysis : ")

        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df["tempC"] = pd.to_numeric(df["tempC"], errors="coerce")
        df["humidity"] = pd.to_numeric(df["humidity"], errors="coerce")
        df["windspeedKmph"] = pd.to_numeric(df["windspeedKmph"], errors="coerce")
        df["precipMM"] = pd.to_numeric(df["precipMM"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        st.subheader("1. Average Pressure by City")
        avg_pressure = (
            df.groupby("city")["pressure"]
            .mean()
            .reset_index()
            .rename(columns={"pressure": "Avg Pressure (hPa)", "city": "City"})
        )

        fig_pressure_city = px.bar(
            avg_pressure,
            x="City",
            y="Avg Pressure (hPa)",
            title="Average Pressure by City",
            labels={"Avg Pressure (hPa)": "Avg Pressure (hPa)"},
        )
        fig_pressure_city.update_xaxes(tickangle=45)
        st.plotly_chart(fig_pressure_city, use_container_width=True)

        st.subheader("2. Correlations with Pressure")
        corr_df = df[
            ["pressure", "tempC", "humidity", "windspeedKmph", "precipMM"]
        ].corr()
        fig_corr = px.imshow(
            corr_df,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            labels={"x": "", "y": "", "color": "Correlation"},
            title="Correlation Matrix with Pressure",
        )
        fig_corr.update_layout(width=600, height=500)
        st.plotly_chart(fig_corr, use_container_width=True)

        st.subheader("3. Average Pressure by Month")
        df["month"] = df["date"].dt.month
        monthly_pressure = (
            df.groupby("month")["pressure"].mean().reset_index().sort_values("month")
        )
        monthly_pressure["Month"] = monthly_pressure["month"].apply(
            lambda m: pd.to_datetime(str(m), format="%m").strftime("%b")
        )
        monthly_pressure = monthly_pressure.rename(
            columns={"pressure": "Avg Pressure (hPa)"}
        )

        fig_month = px.line(
            monthly_pressure,
            x="Month",
            y="Avg Pressure (hPa)",
            title="Average Pressure by Month",
            markers=True,
            labels={"Avg Pressure (hPa)": "Avg Pressure (hPa)"},
        )
        fig_month.update_xaxes(
            categoryorder="array",
            categoryarray=[
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ],
        )
        st.plotly_chart(fig_month, use_container_width=True)

        st.subheader("7. Pressure vs Wind Speed Colored by Wind Direction")

        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df["windspeedKmph"] = pd.to_numeric(df["windspeedKmph"], errors="coerce")
        df["winddirDegree"] = pd.to_numeric(df["winddirDegree"], errors="coerce")

        fig7 = px.scatter(
            df,
            x="pressure",
            y="windspeedKmph",
            color="winddirDegree",
            color_continuous_scale="RdBu",
            labels={
                "pressure": "Pressure (hPa)",
                "windspeedKmph": "Wind Speed (Kmph)",
                "winddirDegree": "Wind Direction (°)",
            },
            title="Pressure vs. Wind Speed with Wind Direction Gradient",
            template="plotly_dark",
        )
        fig7.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("8. Pressure Trends by City, Season & Year Range")

        def get_season(dt):
            m = dt.month
            if m in [3, 4, 5]:
                return "Summer"
            if m in [6, 7, 8, 9]:
                return "Monsoon"
            if m in [10, 11]:
                return "Autumn"
            return "Winter"

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["season"] = df["date"].apply(get_season)
        df["precipMM"] = pd.to_numeric(df["precipMM"], errors="coerce")

        city_list = sorted(df["city"].unique())
        season_list = ["Summer", "Monsoon", "Autumn", "Winter"]
        year_min, year_max = int(df["date"].dt.year.min()), int(
            df["date"].dt.year.max()
        )

        city_sel = st.selectbox("Select City", city_list, key="press_city")
        season_sel = st.selectbox("Select Season", season_list, key="press_season")
        start_year = st.number_input(
            "Start Year",
            min_value=year_min,
            max_value=year_max,
            value=year_min,
            step=1,
            key="press_start",
        )
        end_year = st.number_input(
            "End Year",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            step=1,
            key="press_end",
        )

        mask = (
            (df["city"] == city_sel)
            & (df["season"] == season_sel)
            & (df["date"].dt.year.between(start_year, end_year))
        )
        df_filt = df.loc[mask].copy()

        if df_filt.empty:
            st.warning(
                f"No data for {city_sel}, {season_sel}, {start_year}–{end_year}."
            )
        else:
            df_filt["precipitation"] = df_filt["precipMM"].apply(
                lambda x: "Precipitation" if x > 0 else "No Precipitation"
            )
            df_filt["Year"] = df_filt["date"].dt.year

            fig8 = px.scatter(
                df_filt,
                x="date",
                y="pressure",
                color="precipitation",
                hover_data={"Year": True, "precipMM": True, "date": False},
                labels={"pressure": "Pressure (hPa)", "date": "Date"},
                title=f"Pressure Trends in {city_sel} during {season_sel} ({start_year}–{end_year})",
                template="plotly_dark",
            )
            fig8.add_scatter(
                x=df_filt["date"],
                y=df_filt["pressure"],
                mode="lines",
                name="Pressure Trend",
                line=dict(color="blue"),
            )
            fig8.update_xaxes(
                dtick="M12",
                tickformat="%Y",
                tick0=f"{start_year}-01-01",
                range=[f"{start_year}-01-01", f"{end_year}-12-31"],
            )
            fig8.update_layout(
                xaxis_title="Year", yaxis_title="Pressure (hPa)", hovermode="x unified"
            )
            st.plotly_chart(fig8, use_container_width=True)

        st.subheader("9. Sun Hours vs Pressure (Cyclone Conditions)")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")
        df["WindGustKmph"] = pd.to_numeric(df["WindGustKmph"], errors="coerce")
        df["cloudcover"] = pd.to_numeric(df["cloudcover"], errors="coerce")
        df["sunHour"] = pd.to_numeric(df["sunHour"], errors="coerce")

        city_list = sorted(df["city"].unique())
        year_min = int(df["date"].dt.year.min())
        year_max = int(df["date"].dt.year.max())
        city_sel = st.selectbox("Select City", city_list, key="cyclone_city")
        start_year = st.number_input(
            "Start Year",
            min_value=year_min,
            max_value=year_max,
            value=year_min,
            key="cyclone_start",
        )
        end_year = st.number_input(
            "End Year",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            key="cyclone_end",
        )

        df_cyc = df[
            (df["city"] == city_sel)
            & (df["date"].dt.year.between(start_year, end_year))
        ].copy()
        if df_cyc.empty:
            st.warning(f"No data for {city_sel} {start_year}-{end_year}.")
        else:
            df_cyc["Cyclone"] = (
                (df_cyc["pressure"] < 1000)
                & (df_cyc["WindGustKmph"] > 50)
                & (df_cyc["cloudcover"] > 60)
            ).map({True: "Cyclone", False: "No Cyclone"})
            fig9 = px.scatter(
                df_cyc,
                x="sunHour",
                y="pressure",
                color="Cyclone",
                hover_data=["WindGustKmph", "cloudcover", "precipMM", "date"],
                labels={"sunHour": "Sun Hours", "pressure": "Pressure (hPa)"},
                title=f"Sun Hours vs Pressure in {city_sel} ({start_year}-{end_year})",
                template="plotly_dark",
            )
            st.plotly_chart(fig9, use_container_width=True)

            cyclones = df_cyc[df_cyc["Cyclone"] == "Cyclone"]
            if not cyclones.empty:
                st.subheader("Cyclone Occurrences")
                st.dataframe(
                    cyclones[
                        ["date", "pressure", "WindGustKmph", "cloudcover", "precipMM"]
                    ]
                )
            else:
                st.info("No cyclone conditions detected in this period.")

        st.subheader("10. Seasonal Pressure Distribution")

        def get_season(dt):
            m = dt.month
            if m in [3, 4, 5]:
                return "Summer"
            if m in [6, 7, 8, 9]:
                return "Monsoon"
            if m in [10, 11]:
                return "Autumn"
            return "Winter"

        def classify_pressure(p):
            if p < 900:
                return "Extremely Low"
            elif p < 980:
                return "Very Low"
            elif p < 990:
                return "Low"
            elif p < 1005:
                return "Monsoon Low"
            elif p < 1010:
                return "Heat Low"
            elif p < 1015:
                return "Moderate High"
            elif p < 1020:
                return "Winter High"
            else:
                return "Anticyclonic High"

        df["season"] = df["date"].apply(get_season)
        df["Pressure Category"] = df["pressure"].apply(classify_pressure)

        city_sel2 = st.selectbox("City for Distribution", city_list, key="dist_city")
        start_y2 = st.number_input(
            "Start Year for Distribution",
            min_value=year_min,
            max_value=year_max,
            value=year_min,
            key="dist_start",
        )
        end_y2 = st.number_input(
            "End Year for Distribution",
            min_value=year_min,
            max_value=year_max,
            value=year_max,
            key="dist_end",
        )

        df_dist = df[
            (df["city"] == city_sel2) & (df["date"].dt.year.between(start_y2, end_y2))
        ].copy()
        if df_dist.empty:
            st.warning(f"No data for {city_sel2} {start_y2}-{end_y2}.")
        else:
            pct = df_dist["Pressure Category"].value_counts(normalize=True) * 100
            fig10 = px.pie(
                names=pct.index,
                values=pct.values,
                title=f"Pressure Categories in {city_sel2} ({start_y2}-{end_y2})",
                template="plotly_dark",
            )
            st.plotly_chart(fig10, use_container_width=True)

            df_dist["Year"] = df_dist["date"].dt.year
            summary = (
                df_dist.groupby(["Year", "season", "Pressure Category"])
                .size()
                .reset_index(name="days")
            )
            summary["pct"] = summary.groupby(["Year", "season"])["days"].transform(
                lambda x: x / x.sum() * 100
            )

            st.subheader("Seasonal % of Low vs High Pressure Days")
            st.dataframe(summary)

        st.subheader("11. Pressure Distribution Comparison Between Two Cities")

        def classify_pressure(p):
            if p < 900:
                return "Extremely Low"
            elif p < 980:
                return "Very Low"
            elif p < 990:
                return "Low"
            elif p < 1005:
                return "Monsoon Low"
            elif p < 1010:
                return "Heat Low"
            elif p < 1015:
                return "Moderate High"
            elif p < 1020:
                return "Winter High"
            else:
                return "Anticyclonic High"

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["pressure"] = pd.to_numeric(df["pressure"], errors="coerce")

        cities = sorted(df["city"].unique())
        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        city1 = st.selectbox("City 1", cities, key="cmp_city1")
        city2 = st.selectbox("City 2", cities, key="cmp_city2")
        start_y = st.selectbox("Start Year", years, key="cmp_start")
        end_y = st.selectbox("End Year", years, key="cmp_end")

        mask = df["city"].isin([city1, city2]) & df["date"].dt.year.between(
            start_y, end_y
        )
        df_cmp = df.loc[mask].copy()
        df_cmp["Pressure Category"] = df_cmp["pressure"].apply(classify_pressure)

        def get_pct_df(city):
            pct = (
                df_cmp[df_cmp["city"] == city]["Pressure Category"]
                .value_counts(normalize=True)
                .mul(100)
                .rename_axis("Category")
                .reset_index(name="Pct")
            )
            return pct

        pct1 = get_pct_df(city1)
        pct2 = get_pct_df(city2)

        import plotly.graph_objects as go

        fig = go.Figure()

        fig.add_trace(
            go.Pie(
                labels=pct1["Category"],
                values=pct1["Pct"],
                name=city1,
                domain={"x": [0, 0.48]},
                hole=0.3,
            )
        )
        fig.add_trace(
            go.Pie(
                labels=pct2["Category"],
                values=pct2["Pct"],
                name=city2,
                domain={"x": [0.52, 1]},
                hole=0.3,
            )
        )

        fig.update_layout(
            title_text=f"Pressure Distribution: {city1} vs {city2} ({start_y}–{end_y})",
            annotations=[
                dict(text=city1, x=0.20, y=0.5, font_size=14, showarrow=False),
                dict(text=city2, x=0.80, y=0.5, font_size=14, showarrow=False),
            ],
            template="plotly_dark",
        )

        st.plotly_chart(fig, use_container_width=True)

    elif analysis_topic == "8. Moon illumination":
        st.header("Moon illumination Analysis : ")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        st.subheader("1. Average Moon Illumination")
        avg_moon = df["moon_illumination"].mean()
        st.write(f"**Average Moon Illumination:** {avg_moon:.2f}%")

        st.subheader("2. City‑wise Average Moon Illumination")
        city_moon = (
            df.groupby("city")["moon_illumination"]
            .mean()
            .reset_index()
            .rename(columns={"moon_illumination": "Avg Moon Illumination (%)"})
        )
        st.dataframe(city_moon)

        st.subheader("3. Dates of Highest & Lowest Moon Illumination")
        max_idx = df["moon_illumination"].idxmax()
        min_idx = df["moon_illumination"].idxmin()
        max_date = df.loc[max_idx, "date"]
        min_date = df.loc[min_idx, "date"]
        st.write(f"- **Date of highest illumination:** {max_date.date()}")
        st.write(f"- **Date of lowest illumination:** {min_date.date()}")

        st.subheader("4. Number of Unique Moon Phases")
        unique_phases = df["moonrise"].nunique()
        st.write(f"**Unique Moon Phase Entries:** {unique_phases}")

        st.subheader("5. Solar & Lunar Time Series Trends")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["sunset_dt"] = pd.to_datetime(
            df["sunset"], format="%I:%M %p", errors="coerce"
        )
        df["moonrise_dt"] = pd.to_datetime(
            df["moonrise"], format="%I:%M %p", errors="coerce"
        )
        df["moonset_dt"] = pd.to_datetime(
            df["moonset"], format="%I:%M %p", errors="coerce"
        )

        df["sunset_hour"] = df["sunset_dt"].dt.hour + df["sunset_dt"].dt.minute / 60
        df["moonrise_hour"] = (
            df["moonrise_dt"].dt.hour + df["moonrise_dt"].dt.minute / 60
        )
        df["moonset_hour"] = df["moonset_dt"].dt.hour + df["moonset_dt"].dt.minute / 60

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month_name()

        sunhour_trends = df.groupby(["year", "month"])["sunHour"].mean().reset_index()
        sunset_trends = (
            df.groupby(["year", "month"])["sunset_hour"].mean().reset_index()
        )
        moonrise_trends = (
            df.groupby(["year", "month"])["moonrise_hour"].mean().reset_index()
        )
        moonset_trends = (
            df.groupby(["year", "month"])["moonset_hour"].mean().reset_index()
        )

        st.markdown("**Average Sun Hours by Month & Year**")
        fig1 = px.pie(
            sunhour_trends,
            names="month",
            values="sunHour",
            title="Avg Sun Hours (2009–2019)",
            hole=0.4,
            template="plotly_dark",
        )
        st.plotly_chart(fig1, use_container_width=True)

        st.markdown("**Average Sunset Hour by Month & Year**")
        fig2 = px.pie(
            sunset_trends,
            names="month",
            values="sunset_hour",
            title="Avg Sunset Hour (2009–2019)",
            hole=0.4,
            template="plotly_dark",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Average Moonrise Hour by Month & Year**")
        fig3 = px.pie(
            moonrise_trends,
            names="month",
            values="moonrise_hour",
            title="Avg Moonrise Hour (2009–2019)",
            hole=0.4,
            template="plotly_dark",
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("**Average Moonset Hour by Month & Year**")
        fig4 = px.pie(
            moonset_trends,
            names="month",
            values="moonset_hour",
            title="Avg Moonset Hour (2009–2019)",
            hole=0.4,
            template="plotly_dark",
        )
        st.plotly_chart(fig4, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        st.subheader("6. Monthly Moon Illumination Heatmap")

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month_name()
        month_order = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        df["month"] = pd.Categorical(df["month"], categories=month_order, ordered=True)

        lunar_phase = (
            df.groupby(["year", "month"])["moon_illumination"]
            .mean()
            .unstack()
            .reindex(columns=month_order)
        )

        fig6 = px.imshow(
            lunar_phase.T,
            x=lunar_phase.index,
            y=month_order,
            labels={"x": "Year", "y": "Month", "color": "Avg Illumination"},
            color_continuous_scale="YlGnBu",
            title="Avg Monthly Moon Illumination (2009–2019)",
            aspect="auto",
            template="plotly_dark",
        )
        st.plotly_chart(fig6, use_container_width=True)

        st.subheader("7. Lunar vs Weather Correlations")

        df["sunset_dt"] = pd.to_datetime(
            df["sunset"], format="%I:%M %p", errors="coerce"
        )
        df["moonrise_dt"] = pd.to_datetime(
            df["moonrise"], format="%I:%M %p", errors="coerce"
        )
        df["moonset_dt"] = pd.to_datetime(
            df["moonset"], format="%I:%M %p", errors="coerce"
        )
        df["sunset_hour"] = df["sunset_dt"].dt.hour + df["sunset_dt"].dt.minute / 60
        df["moonrise_hour"] = (
            df["moonrise_dt"].dt.hour + df["moonrise_dt"].dt.minute / 60
        )
        df["moonset_hour"] = df["moonset_dt"].dt.hour + df["moonset_dt"].dt.minute / 60

        lunar_weather = df[
            ["moon_illumination", "moonrise_hour", "moonset_hour", "tempC", "precipMM"]
        ].dropna()
        corr_lunar = lunar_weather.corr()

        fig7 = px.imshow(
            corr_lunar,
            text_auto=True,
            color_continuous_scale="YlGnBu",
            title="Lunar-Weather Correlation",
            labels={"x": "Variable", "y": "Variable", "color": "Corr"},
            template="plotly_dark",
        )
        st.plotly_chart(fig7, use_container_width=True)

        st.subheader("8. Avg Illumination by Lunar Phase for Selected Year & Month")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        sel_year = st.selectbox("Select Year", years, key="illum_year")
        sel_month = st.selectbox("Select Month", months, key="illum_month")

        df_sel = df[
            (df["date"].dt.year == sel_year) & (df["date"].dt.month_name() == sel_month)
        ].copy()

        if df_sel.empty:
            st.warning(f"No data for {sel_month} {sel_year}.")
        else:

            def get_lunar_phase(illum):
                if pd.isna(illum):
                    return "Unknown"
                if illum == 0:
                    return "New Moon"
                if illum < 50:
                    return "Waxing Crescent"
                if illum == 50:
                    return "First Quarter"
                if illum < 100:
                    return "Waxing Gibbous"
                if illum == 100:
                    return "Full Moon"
                return "Other"

            df_sel["lunar_phase"] = df_sel["moon_illumination"].apply(get_lunar_phase)

            phase_avg = (
                df_sel.groupby("lunar_phase")["moon_illumination"]
                .mean()
                .reset_index(name="Avg Illumination")
            )

            fig8 = px.pie(
                phase_avg,
                names="lunar_phase",
                values="Avg Illumination",
                title=f"Avg Moon Illumination by Phase: {sel_month} {sel_year}",
                hole=0.4,
                template="plotly_dark",
            )
            st.plotly_chart(fig8, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["sunset_dt"] = pd.to_datetime(
            df["sunset"], format="%I:%M %p", errors="coerce"
        )
        df["sunrise_dt"] = pd.to_datetime(
            df["sunrise"], format="%I:%M %p", errors="coerce"
        )
        df["moon_illumination"] = pd.to_numeric(
            df["moon_illumination"], errors="coerce"
        )

        df["sunset_hour"] = df["sunset_dt"].dt.hour + df["sunset_dt"].dt.minute / 60
        df["sunrise_hour"] = df["sunrise_dt"].dt.hour + df["sunrise_dt"].dt.minute / 60

        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month

        st.subheader("9. Day & Night Duration Over the Years")

        df["day_duration"] = df["sunset_hour"] - df["sunrise_hour"]
        df["night_duration"] = 24 - df["day_duration"]

        years = sorted(df["year"].dropna().unique().astype(int))
        sel_year = st.selectbox("Select Year", years, key="dn_year")

        df_year = df[df["year"] == sel_year]
        if df_year.empty:
            st.warning(f"No data for {sel_year}.")
        else:
            grouped = (
                df_year.groupby("month")
                .agg(
                    avg_day=("day_duration", "mean"),
                    avg_night=("night_duration", "mean"),
                )
                .reset_index()
            )

            month_names = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            fig9 = go.Figure()
            fig9.add_trace(
                go.Scatter(
                    x=grouped["month"],
                    y=grouped["avg_day"],
                    mode="lines+markers",
                    name="Day Duration",
                )
            )
            fig9.add_trace(
                go.Scatter(
                    x=grouped["month"],
                    y=grouped["avg_night"],
                    mode="lines+markers",
                    name="Night Duration",
                )
            )
            fig9.update_layout(
                title=f"Day/Night Duration in {sel_year}",
                xaxis=dict(
                    tickmode="array", tickvals=list(range(1, 13)), ticktext=month_names
                ),
                yaxis_title="Hours",
                template="plotly_dark",
            )
            st.plotly_chart(fig9, use_container_width=True)

        st.subheader("10. Tidal Types by Moon Phase")

        tidal_year = st.selectbox(
            "Select Year for Tidal Analysis", years, key="tide_year"
        )
        df_tide = df[df["year"] == tidal_year].copy()

        if df_tide.empty:
            st.warning(f"No data for {tidal_year}.")
        else:

            def get_moon_phase(illum):
                if np.isnan(illum):
                    return "Unknown"
                if illum == 0:
                    return "New Moon"
                if illum < 50:
                    return "Waxing Crescent"
                if illum == 50:
                    return "First Quarter"
                if illum < 100:
                    return "Waxing Gibbous"
                if illum == 100:
                    return "Full Moon"
                return "Other"

            def get_tide_type(phase):
                if phase in ["Full Moon", "New Moon"]:
                    return "Spring Tide"
                if phase in ["First Quarter", "Last Quarter"]:
                    return "Neap Tide"
                return "Normal Tide"

            df_tide["moon_phase"] = df_tide["moon_illumination"].apply(get_moon_phase)
            df_tide["tide_type"] = df_tide["moon_phase"].apply(get_tide_type)

            fig10 = px.line(
                df_tide,
                x="date",
                y="moon_illumination",
                color="tide_type",
                title=f"Tidal Types in {tidal_year}",
                labels={
                    "moon_illumination": "Illumination (%)",
                    "tide_type": "Tide Type",
                },
                template="plotly_dark",
            )
            fig10.update_traces(mode="markers+lines")
            fig10.update_layout(height=600)
            st.plotly_chart(fig10, use_container_width=True)

        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["moon_illumination"] = pd.to_numeric(
            df["moon_illumination"], errors="coerce"
        )
        df["sunrise_dt"] = pd.to_datetime(
            df["sunrise"], format="%I:%M %p", errors="coerce"
        )
        df["sunset_dt"] = pd.to_datetime(
            df["sunset"], format="%I:%M %p", errors="coerce"
        )

        df["sunrise_hour"] = df["sunrise_dt"].dt.hour + df["sunrise_dt"].dt.minute / 60
        df["sunset_hour"] = df["sunset_dt"].dt.hour + df["sunset_dt"].dt.minute / 60

        st.subheader("11. Moon Phase Distribution by Year")

        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        years = sorted(df["date"].dt.year.dropna().unique().astype(int))
        sel_year_phase = st.selectbox(
            "Select Year for Phase Distribution", years, key="phase_year_dist"
        )

        df_year = df[df["date"].dt.year == sel_year_phase].copy()

        if df_year.empty:
            st.warning(f"No data available for {sel_year_phase}.")
        else:

            def get_lunar_phase(illum):
                if pd.isna(illum):
                    return "Unknown"
                if illum == 0:
                    return "New Moon"
                if illum < 50:
                    return "Waxing Crescent"
                if illum == 50:
                    return "First Quarter"
                if illum < 100:
                    return "Waxing Gibbous"
                if illum == 100:
                    return "Full Moon"
                return "Other"

            df_year["moon_phase"] = df_year["moon_illumination"].apply(get_lunar_phase)
            df_year["day"] = df_year["date"].dt.date

            daily_phase = df_year.groupby("day")["moon_phase"].agg(
                lambda x: x.value_counts().idxmax()
            )

            pie_df = (
                daily_phase.value_counts()
                .rename_axis("Moon Phase")
                .reset_index(name="Days")
            )

            total_days = pie_df["Days"].sum()

            fig_phase = px.pie(
                pie_df,
                names="Moon Phase",
                values="Days",
                title=f"Moon Phase Distribution in {sel_year_phase} (n={total_days} days)",
                hole=0.4,
                template="plotly_dark",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_phase.update_traces(textinfo="percent+label")
            st.plotly_chart(fig_phase, use_container_width=True)

        st.subheader("12. Daily Moon Phases for a City & Month")

        months = [
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        ]
        sel_month = st.selectbox("Select Month", months, key="phase_month")
        cities = sorted(df["city"].dropna().unique())
        sel_city = st.selectbox("Select City", cities, key="phase_city")

        df_dm = df[
            (df["date"].dt.year == sel_year)
            & (df["date"].dt.month_name() == sel_month)
            & (df["city"] == sel_city)
        ].copy()

        if df_dm.empty:
            st.warning(f"No data for {sel_city} in {sel_month} {sel_year}.")
        else:

            def get_moon_symbol(illum):
                if pd.isna(illum):
                    return "●"
                if illum == 0:
                    return "●"
                if illum < 50:
                    return "◔"
                if illum == 50:
                    return "◑"
                if illum < 100:
                    return "◐"
                if illum == 100:
                    return "○"
                return "●"

            df_dm["phase_symbol"] = df_dm["moon_illumination"].apply(get_moon_symbol)

            jitter = np.random.uniform(-1, 1, size=len(df_dm))
            df_dm["illum_jitter"] = df_dm["moon_illumination"] + jitter

            fig_dm = px.scatter(
                df_dm,
                x="date",
                y="illum_jitter",
                symbol="phase_symbol",
                title=f"Daily Moon Phases in {sel_city}, {sel_month} {sel_year}",
                labels={"illum_jitter": "Moon Illumination (%)"},
                template="plotly_dark",
                width=900,
                height=500,
            )
            fig_dm.update_traces(
                marker=dict(size=12, color="lightblue"), showlegend=False
            )

            legend_txt = "<br>".join(
                [
                    "● New Moon",
                    "◔ Waxing Crescent",
                    "◑ First Quarter",
                    "◐ Waxing Gibbous",
                    "○ Full Moon",
                ]
            )
            fig_dm.add_annotation(
                xref="paper",
                yref="paper",
                x=0.85,
                y=0.95,
                text=legend_txt,
                showarrow=False,
                bgcolor="rgba(255,255,255,0.7)",
                bordercolor="white",
                font=dict(color="black", size=10),
            )

            st.plotly_chart(fig_dm, use_container_width=True)


elif page == "Outcomes":
    st.title("🔮 Key Outcomes & Insights")

    o1, o2, o3, o4 = st.columns(4)

    with o1:
        st.markdown(
            """
            **🌡 Avg Temp Δ North vs South**  
            ↑ 5.2 °C
            """
        )

    with o2:
        st.markdown(
            """
            **☀️ Peak UV Cities**  
            Nagpur, Hyderabad
            """
        )

    with o3:
        st.markdown(
            """
            **⛈️ Monsoon Risk Areas**  
            Mumbai, Hyderabad
            """
        )

    with o4:
        st.markdown(
            """
            **💨 Max Wind Zones**  
            Delhi, Jaipur
            """
        )

    st.markdown("---")

    st.subheader("1. Temperature & Heat Stress")
    st.write(
        """
        • **Extreme North:** Delhi & Jaipur saw severe summers (>40 °C) and winter chills.  
        • **Stable South:** Bengaluru & Hyderabad maintained moderate, stable temperatures year‑round.  
        • **Risk Peak:** Delhi experienced the highest heat stress in June, with several days above 45 °C.
        """
    )

    st.subheader("2. UV Exposure")
    st.write(
        """
        • **Highest Risk:** Nagpur and Hyderabad peaked above UV index 11 during April–June.  
        • **Lowest:** Pune consistently stayed below UV index 8.  
        • **Moderate:** Delhi and Jaipur experienced mid‑range UV levels year‑round.
        """
    )

    st.subheader("3. Solar Energy Potential")
    st.write(
        """
        • **High Yield:** Pune and Hyderabad averaged over 9 sun‑hours/day outside monsoon.  
        • **Monsoon Dip:** Mumbai’s cloud cover (June–Sept) reduced solar yield by ~25 %.  
        • **Consistent:** Hyderabad showed two annual peaks (March, October) ideal for solar farms.
        """
    )

    st.subheader("4. Precipitation & Flood Risks")
    st.write(
        """
        • **Max Rainfall:** Mumbai recorded up to 245 mm in a single monsoon day.  
        • **Flash Flood Hotspots:** Mumbai & Hyderabad face highest 1‑hour totals (>50 mm).  
        • **Monsoon Dominance:** Over 80 % of annual rain falls during Jun–Sept in Jaipur & Mumbai.
        """
    )

    st.subheader("5. Wind & Storms")
    st.write(
        """
        • **Strongest Gusts:** Delhi and Jaipur saw gusts exceeding 75 km/h during pre‑monsoon storms.  
        • **Pressure‐Wind Link:** Kanpur and Nagpur exhibit low‐pressure days coinciding with peak wind events.  
        • **Dominant Directions:** NW storms in Jaipur vs. westerlies in Mumbai.
        """
    )

    st.subheader("6. Fog & Visibility")
    st.write(
        """
        • **Severe Fog:** Delhi (Dec–Jan) often dropped below 500 m visibility, disrupting flights.  
        • **Low Risk:** Bengaluru enjoyed clear visibility year‑round.  
        • **Winter Haze:** Kanpur struggled with reduced visibility (700–1,000 m) in December.
        """
    )

    st.subheader("7. Pressure & Storm Conditions")
    st.write(
        """
        • **Deep Pressure Drops:** Kanpur’s monsoon lows (<990 hPa) aligned with torrential rains.  
        • **Storm Troughs:** Hyderabad experienced its lowest pressure in July, triggering local storms.
        """
    )

    st.subheader("8. Lunar Influence")
    st.write(
        """
        • **Tidal Impact:** Full and new moons exacerbate coastal flooding in Mumbai during monsoon.  
        • **Temperature Shifts:** Nagpur showed minor cooling (~1 °C) around full‑moon nights.
        """
    )

    st.markdown("---")

    st.subheader("Limitations")
    st.write(
        """
        - **Excluded 2020 Data:** Only 15 days available → insufficient for trend analysis.  
        - **Snowfall Column:** All zeros → no snowfall insights.  
        - **Time Format:** AM/PM conversion added extra preprocessing steps.  
        - **Data Gaps:** Occasional missing visibility/humidity impacted certain analyses.
        """
    )

    st.subheader("Future Scope")
    st.write(
        """
        - **Predictive Models:** Implement ML-based forecasting for heatwaves and monsoons.  
        - **Interactive Dashboards:** Extend this Streamlit app with user‐driven scenario simulations.  
        - **Expanded Coverage:** Include post‑2019 data and additional Tier‑II cities.  
        - **Advanced Analytics:** Develop multi‑variable risk indices combining heat, pollution, and rainfall.
        """
    )

    st.subheader("Conclusion")
    st.write(
        """
        Regional contrasts highlight North’s extreme temperature swings vs. South’s stability.  
        Critical risks—heatwaves, floods, fog—demand targeted resilience strategies.  
        Renewable potential peaks in Pune/Hyderabad offer avenues for solar energy expansion.  
        These insights can guide agriculture planning, urban flood defenses, and public health policies.
        """
    )

    st.markdown("---")

    st.caption(
        "🌐  Data-driven insights to guide agriculture, resilience & public health planning."
    )


st.markdown("---")
st.markdown(
    """<p style="text-align: center; font-size:12px; color:gray;">
    © All Rights Reserved. Developed by Angira, Reekparna, Pratyush
    </p>""",
    unsafe_allow_html=True,
)
