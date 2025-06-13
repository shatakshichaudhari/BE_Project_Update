import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ortools.linear_solver import pywraplp
from fpdf import FPDF
import io
import os
from PIL import Image # For getting image dimensions if needed, or for more robust image handling
# from scipy.stats import pearsonr # For correlation, if you want to implement it more formally

def fig_to_bytesio(fig):
    # ... (keep as is) ...
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches='tight', dpi=150)
    img_buffer.seek(0)
    return img_buffer

# --- Helper function for calculating PDF line breaks after an image ---
def get_image_height_for_pdf(image_buffer, target_width_mm):
    # ... (keep as is) ...
    try:
        image_buffer.seek(0)
        img = Image.open(image_buffer)
        width_px, height_px = img.size
        aspect_ratio = height_px / width_px
        height_mm = aspect_ratio * target_width_mm
        return height_mm
    except Exception:
        return 70 # Default height if something goes wrong

def main():
    st.set_page_config(page_title="Rail Project Dashboard", layout="wide")

    # Initialize session state variables if they don't exist
    if 'optimization_results' not in st.session_state:
        st.session_state.optimization_results = {}
    if 'ppg_results' not in st.session_state:
        st.session_state.ppg_results = {}

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["🏠 Introduction", "📊 Optimization Analysis", "📈 PPG Dataset Analysis", "📄 Download Reports", "🔧 Future Page"])

    if page == "🏠 Introduction":
        introduction_page()
    elif page == "📊 Optimization Analysis":
        optimization_page()
    elif page == "📈 PPG Dataset Analysis":
        ppg_analysis_page()
    elif page == "📄 Download Reports":
        download_reports_page()
    elif page == "🔧 Future Page":
        future_page()

def introduction_page():
    st.title("🚆 Welcome to the Railway Optimization System")

    st.write("## 🔍 What is this application about?")
    st.write(
        "- This app is designed to **optimize railway resource allocation** by analyzing labor costs, efficiency, and fatigue levels."
        "\n- It uses **data-driven insights** and **AI-based models** to improve **scheduling, cost management, and worker productivity**."
        "\n- **Goal:** To create a **smarter, safer, and more efficient** railway workforce management system."
    )

    st.write("---")

    st.write("## 👥 Who is this for?")
    st.write(
        "- **Railway Project Managers** 🏗️ - To efficiently allocate labor and reduce operational costs."
        "\n- **Railway Authorities & Engineers** 🛤️ - To make informed decisions using real-time data."
        "\n- **Policy Makers & Analysts** 📊 - To ensure compliance, efficiency, and safety improvements."
    )

    st.write("---")

    st.write("## 🕒 When is Railway Optimization Needed?")
    st.write(
        "- When managing **large-scale railway projects** with multiple workforce categories."
        "\n- When **cost efficiency** and **timely execution** are crucial."
        "\n- When **worker fatigue and safety** need to be monitored and improved."
    )

    st.write("---")

    st.write("## 📍 Where does this optimization apply?")
    st.write(
        "- **Railway Construction & Maintenance** 🚄 - Managing workers efficiently for repairs, track laying, and station maintenance."
        "\n- **Operational Workforce Planning** 👷‍♂️ - Ensuring the right number of workers are assigned to different shifts."
        "\n- **Cost Reduction Strategies** 💰 - Identifying the most cost-effective labor allocation."
    )

    st.write("---")

    st.write("## ❓ Why is this important?")
    st.write(
        "- ✅ **Saves Costs**: Reduces unnecessary spending on overstaffing."
        "\n- ✅ **Prevents Worker Fatigue**: Uses **PPG (Photoplethysmography) Data** to track and analyze worker exhaustion."
        "\n- ✅ **Improves Project Efficiency**: Ensures the **right workers** are assigned at the **right time**."
        "\n- ✅ **Enhances Decision-Making**: Provides clear **data visualization & reports** for better insights."
    )

    st.write("---")

    st.write("## ⚙️ How does it work?")
    st.write(
        "- 🛠️ **Optimization Engine:** Uses AI to calculate **optimal labor allocation** based on cost & efficiency."
        "\n- 📊 **Data Visualization:** Interactive **charts** provide insights into labor trends and fatigue levels."
        "\n- 🩺 **Fatigue Analysis:** Integrates **PPG data** to measure and prevent worker exhaustion."
        "\n- 📄 **PDF Reports:** Generate **detailed downloadable reports** for project tracking."
    )

    st.write("---")

    st.success("🎯 **Get Started!** Use the sidebar to navigate through different sections and explore the features.")


def optimization_page():
    st.title("📊 Optimization Analysis")
    st.write("Upload labor cost and project requirement datasets to optimize workforce allocation.")
    
    labor_file = st.file_uploader("Upload Labor Costs & Availability CSV", type=["csv"], key="opt_labor_file")
    project_file = st.file_uploader("Upload Project Requirements CSV", type=["csv"], key="opt_project_file")
    
    # Clear previous results if files change or page reloads without new analysis
    # This logic might need refinement based on exact desired behavior
    if not labor_file or not project_file:
        st.session_state.optimization_results = {}


    if labor_file and project_file:
        try:
            labor_df = pd.read_csv(labor_file)
            project_df = pd.read_csv(project_file)

            labor_df.columns = labor_df.columns.str.strip()
            project_df.columns = project_df.columns.str.strip()
            
            st.session_state.optimization_results['labor_df_orig'] = labor_df.copy()
            st.session_state.optimization_results['project_df_orig'] = project_df.copy()

            selected_project = st.selectbox("Select Project", project_df['Project Name'].unique())
            project_data_selected = project_df[project_df['Project Name'] == selected_project]
            
            selected_labor_types = st.multiselect("Select Labor Types", labor_df['Labor Type'].unique(), default=labor_df['Labor Type'].unique())
            labor_data_selected = labor_df[labor_df['Labor Type'].isin(selected_labor_types)]
            
            if labor_data_selected.empty or project_data_selected.empty:
                st.warning("Please select valid project and labor types resulting in non-empty data.")
                return

            max_hours_slider_max = int(labor_data_selected['Maximum Available Hours'].max()) if not labor_data_selected.empty else 200
            max_hours_selected = st.slider("Max Hours per Worker", min_value=50, max_value=max_hours_slider_max, value=max_hours_slider_max, step=50)
            
            costs = labor_data_selected['Cost per Hour (₹)'].values.astype(float)
            max_hours = np.minimum(labor_data_selected['Maximum Available Hours'].values.astype(float), max_hours_selected)
            max_hours = np.nan_to_num(max_hours, nan=0.0)
            labor_types_actual = labor_data_selected['Labor Type'].values
            required_hours = project_data_selected['Total Required Hours'].sum()
            num_workers = len(labor_types_actual)

            if num_workers == 0:
                st.error("No labor types selected or available for optimization.")
                return
            
            solver = pywraplp.Solver.CreateSolver('SCIP')
            worker_hours = [solver.NumVar(0.0, float(max_hours[i]), f'worker_{i}') for i in range(num_workers)]
            
            solver.Add(solver.Sum(worker_hours) == required_hours)
            min_allocation = required_hours / num_workers if num_workers > 0 else 0
            for i in range(num_workers):
                solver.Add(worker_hours[i] >= min_allocation * 0.7)
                solver.Add(worker_hours[i] <= max_hours[i])
            
            solver.Minimize(solver.Sum(worker_hours[i] * costs[i] for i in range(num_workers)))
            
            status = solver.Solve()

            if status == pywraplp.Solver.OPTIMAL:
                allocation = [round(worker_hours[i].solution_value(), 2) for i in range(num_workers)]
                optimized_cost = round(sum(allocation[i] * costs[i] for i in range(num_workers)), 2)
                allocation_df = pd.DataFrame([allocation], columns=labor_types_actual, index=["Allocated Hours"])
                
                st.subheader("✅ Optimized Labor Allocation")
                st.dataframe(allocation_df.style.format(precision=2))
                st.metric(label="💰 Optimized Total Cost", value=f"₹{optimized_cost:,.2f}")
                
                # Store results for report
                st.session_state.optimization_results['allocation_df'] = allocation_df
                st.session_state.optimization_results['optimized_cost'] = optimized_cost
                st.session_state.optimization_results['selected_project_name'] = selected_project
                st.session_state.optimization_results['project_data_selected'] = project_data_selected
                st.session_state.optimization_results['labor_data_selected'] = labor_data_selected
                st.session_state.optimization_results['required_hours_for_project'] = required_hours


                st.subheader("📊 Labor Allocation Distribution")
                fig_alloc, ax_alloc = plt.subplots(figsize=(10, 6))
                sns.barplot(x=labor_types_actual, y=allocation, palette="coolwarm", ax=ax_alloc, edgecolor="black")
                ax_alloc.set_xlabel("Labor Type", fontsize=12)
                ax_alloc.set_ylabel("Assigned Hours", fontsize=12)
                ax_alloc.set_title("Optimized Labor Hours Allocation", fontsize=14, fontweight='bold')
                ax_alloc.grid(axis='y', linestyle='--', alpha=0.7)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()
                st.pyplot(fig_alloc)
                st.session_state.optimization_results['allocation_chart_buffer'] = fig_to_bytesio(fig_alloc)
                
                st.markdown("### 🔍 Insights from the Graph:")
                st.markdown("- **Visual Representation:** Shows how labor hours are distributed across different categories.")
                st.markdown("- **Optimization Impact:** Balances workload efficiently based on available labor and project demand.")
                st.markdown("- **Changes with Slider:** Adjusting max hours influences allocation dynamically.")
                
                st.subheader("📈 Cost Distribution by Labor Type (Based on Optimized Allocation)")
                allocated_costs = [alloc * cost for alloc, cost in zip(allocation, costs)]
                fig_cost, ax_cost = plt.subplots(figsize=(8, 6))
                # Filter out zero-cost labor types for pie chart clarity
                non_zero_costs_data = [(lt, ac) for lt, ac in zip(labor_types_actual, allocated_costs) if ac > 0]
                if non_zero_costs_data:
                    pie_labels = [item[0] for item in non_zero_costs_data]
                    pie_values = [item[1] for item in non_zero_costs_data]
                    wedges, texts, autotexts = ax_cost.pie(pie_values, labels=pie_labels, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("Set2"), textprops={'fontsize': 10})
                    for text in autotexts:
                        text.set_fontsize(12)
                        text.set_fontweight('bold')
                    ax_cost.set_title("Cost Breakdown by Labor Type (Optimized)", fontsize=14, fontweight='bold')
                    st.pyplot(fig_cost)
                    st.session_state.optimization_results['cost_dist_chart_buffer'] = fig_to_bytesio(fig_cost)
                else:
                    st.warning("No costs to display in pie chart (all allocated costs are zero).")

                st.markdown("### 🔍 Key Takeaways from Cost Distribution:")
                st.markdown("- **Expense Breakdown:** Highlights which labor types contribute most to total optimized cost.")
                st.markdown("- **Cost Efficiency:** Helps identify expensive labor types that might need further attention in future projects.")
                st.markdown("- **Optimization Effect:** Shows the financial impact of the optimized labor allocation.")
            else:
                st.error("Optimization failed. Try adjusting constraints or ensure data is valid.")
                st.session_state.optimization_results = {} # Clear results on failure
        except Exception as e:
            st.error(f"An error occurred during optimization: {e}")
            st.session_state.optimization_results = {}


def ppg_analysis_page():
    st.title("📈 PPG Dataset Analysis")
    st.write("Upload the PPG fatigue dataset to analyze worker fatigue levels and efficiency impact.")
    
    ppg_file = st.file_uploader("Upload PPG Fatigue Dataset CSV", type=["csv"], key="ppg_file_uploader")

    if not ppg_file:
        st.session_state.ppg_results = {}

    if ppg_file:
        try:
            ppg_df_orig = pd.read_csv(ppg_file)
            ppg_df = ppg_df_orig.copy()
            ppg_df.columns = ppg_df.columns.str.strip()
            
            metric_column = 'PPG_Level' # Make sure this column exists
            if metric_column not in ppg_df.columns:
                st.error(f"❌ '{metric_column}' column not found! Available columns: " + ", ".join(ppg_df.columns))
                st.session_state.ppg_results = {}
                return
            
            # Convert metric_column to numeric, coercing errors
            ppg_df[metric_column] = pd.to_numeric(ppg_df[metric_column], errors='coerce')
            ppg_df.dropna(subset=[metric_column], inplace=True) # Remove rows where conversion failed

            if ppg_df.empty:
                st.error(f"No valid numeric data in '{metric_column}' after cleaning.")
                st.session_state.ppg_results = {}
                return

            min_val, max_val = int(ppg_df[metric_column].min()), int(ppg_df[metric_column].max())
            selected_range = st.slider("Filter by PPG Level", min_value=min_val, max_value=max_val, value=(min_val, max_val))
            filtered_df = ppg_df[(ppg_df[metric_column] >= selected_range[0]) & (ppg_df[metric_column] <= selected_range[1])]
            
            st.session_state.ppg_results['filtered_df'] = filtered_df
            st.session_state.ppg_results['selected_ppg_range'] = selected_range

            st.subheader("📊 PPG Level Distribution")
            fig_dist, ax_dist = plt.subplots(figsize=(10, 6))
            sns.histplot(filtered_df[metric_column], bins=20, kde=True, color="royalblue", edgecolor="black", ax=ax_dist)
            ax_dist.set_title("PPG Level Distribution Among Workers", fontsize=14, fontweight='bold')
            ax_dist.set_xlabel("PPG Level", fontsize=12)
            ax_dist.set_ylabel("Number of Workers", fontsize=12)
            ax_dist.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            st.pyplot(fig_dist)
            st.session_state.ppg_results['ppg_dist_chart_buffer'] = fig_to_bytesio(fig_dist)
            
            st.markdown("### 🔍 Key Insights:")
            st.markdown("- **Overall PPG Trends:** Identifies distribution of PPG Levels across workers.")
            st.markdown("- **Clusters & Peaks:** Helps find patterns in worker stress levels.")
            st.markdown("- **Optimization Factor:** Can assist in workload balancing based on physiological stress.")
            
            if 'Activity_Level' in filtered_df.columns:
                filtered_df['Activity_Level'] = pd.to_numeric(filtered_df['Activity_Level'], errors='coerce')
                activity_df_cleaned = filtered_df.dropna(subset=['Activity_Level', metric_column])

                if not activity_df_cleaned.empty:
                    st.subheader("📈 PPG Level vs. Activity Level")
                    fig_scatter, ax_scatter = plt.subplots(figsize=(10, 6))
                    sns.scatterplot(x=activity_df_cleaned[metric_column], y=activity_df_cleaned['Activity_Level'], 
                                    hue=activity_df_cleaned['Activity_Level'], palette="coolwarm", 
                                    edgecolor="black", alpha=0.8, ax=ax_scatter)
                    ax_scatter.set_title("PPG Level vs. Worker Activity", fontsize=14, fontweight='bold')
                    ax_scatter.set_xlabel("PPG Level", fontsize=12)
                    ax_scatter.set_ylabel("Activity Level", fontsize=12)
                    ax_scatter.grid(axis='both', linestyle='--', alpha=0.7)
                    plt.tight_layout()
                    st.pyplot(fig_scatter)
                    st.session_state.ppg_results['ppg_activity_scatter_buffer'] = fig_to_bytesio(fig_scatter)
                    
                    st.markdown("### 🔍 Observations:")
                    st.markdown("- **Correlation Check:** Higher PPG may indicate lower activity levels.")
                    st.markdown("- **Outliers & Trends:** Some workers may maintain high activity despite high PPG levels.")
                    st.markdown("- **Shift Planning Factor:** Can help in designing optimal work schedules.")
                else:
                    st.warning("Not enough valid data for PPG Level vs Activity Level plot after cleaning.")
            else:
                st.warning("Activity level data ('Activity_Level') is missing in the uploaded dataset.")
            
            if 'Timestamp' in filtered_df.columns:
                try:
                    temp_df_time = filtered_df.copy()
                    temp_df_time['Timestamp'] = pd.to_datetime(temp_df_time['Timestamp'], errors='coerce')
                    temp_df_time.dropna(subset=['Timestamp', metric_column], inplace=True)
                    
                    if not temp_df_time.empty:
                        time_trend = temp_df_time.groupby(temp_df_time['Timestamp'].dt.hour)[metric_column].mean()
                        
                        st.subheader("📉 PPG Level Trends Over Time")
                        fig_time, ax_time = plt.subplots(figsize=(10, 6))
                        sns.lineplot(x=time_trend.index, y=time_trend.values, marker='o', color="darkred", ax=ax_time)
                        ax_time.set_title("PPG Level Changes Throughout the Day", fontsize=14, fontweight='bold')
                        ax_time.set_xlabel("Hour of the Day (0-23)", fontsize=12)
                        ax_time.set_ylabel(f"Average {metric_column}", fontsize=12)
                        ax_time.grid(axis='both', linestyle='--', alpha=0.7)
                        plt.tight_layout()
                        st.pyplot(fig_time)
                        st.session_state.ppg_results['ppg_time_trend_buffer'] = fig_to_bytesio(fig_time)
                        
                        st.markdown("### 🔍 What This Tells Us:")
                        st.markdown("- **Peak Stress Hours:** Identifies when workers experience highest PPG levels.")
                        st.markdown("- **Shift Adjustments:** Helps in structuring better work & rest periods.")
                        st.markdown("- **Health & Performance:** Guides strategies to manage worker well-being.")
                    else:
                        st.warning("Not enough valid data for Time-based PPG trends after cleaning.")
                except Exception as e_time:
                    st.warning(f"Could not process Timestamp data for trends: {e_time}")
            else:
                st.warning("Timestamp data is missing. Unable to generate time-based PPG trends.")
        except Exception as e:
            st.error(f"An error occurred during PPG analysis: {e}")
            st.session_state.ppg_results = {}


# --- Insight Generation Functions ---
def generate_optimization_insights(opt_results):
    insights = []
    if not opt_results or 'allocation_df' not in opt_results:
        insights.append("Optimization analysis has not been performed or data is unavailable.")
        return insights

    allocation_df = opt_results.get('allocation_df')
    optimized_cost = opt_results.get('optimized_cost', 0)
    project_name = opt_results.get('selected_project_name', 'N/A')
    required_hours = opt_results.get('required_hours_for_project', 0)
    labor_data = opt_results.get('labor_data_selected') # Contains 'Cost per Hour (₹)'

    insights.append(f"**Project Focus:** {project_name}")
    insights.append(f"**Total Required Hours for Project:** {required_hours:,.2f} hours.")
    insights.append(f"**Optimized Total Cost:** ₹{optimized_cost:,.2f}.")

    if allocation_df is not None and not allocation_df.empty:
        total_allocated_hours = allocation_df.sum(axis=1).iloc[0]
        insights.append(f"**Total Hours Allocated by Optimizer:** {total_allocated_hours:,.2f} hours.")
        if abs(total_allocated_hours - required_hours) > 0.01: # Check for discrepancies
             insights.append(f"  - Note: Allocated hours ({total_allocated_hours:,.2f}) differ slightly from required ({required_hours:,.2f}) due to solver precision or constraints.")

        num_labor_types_used = allocation_df.shape[1]
        insights.append(f"**Labor Types Utilized:** {num_labor_types_used} types involved in the optimized allocation.")

        # Most Utilized Labor Type
        if not allocation_df.empty:
            max_hours_labor_type = allocation_df.T.idxmax().iloc[0]
            max_hours_value = allocation_df.T.max().iloc[0]
            insights.append(f"**Most Utilized Labor Type:** '{max_hours_labor_type}' with {max_hours_value:,.2f} hours allocated.")

        # Cost Drivers (based on allocated hours and cost per hour)
        if labor_data is not None and not labor_data.empty:
            costs_per_hour_map = pd.Series(labor_data['Cost per Hour (₹)'].values, index=labor_data['Labor Type']).to_dict()
            allocated_costs_detail = {}
            for labor_type_col in allocation_df.columns:
                allocated_value = allocation_df[labor_type_col].iloc[0]
                cost_rate = costs_per_hour_map.get(labor_type_col, 0)
                allocated_costs_detail[labor_type_col] = allocated_value * cost_rate
            
            if allocated_costs_detail:
                max_cost_driver_type = max(allocated_costs_detail, key=allocated_costs_detail.get)
                max_cost_driver_value = allocated_costs_detail[max_cost_driver_type]
                insights.append(f"**Primary Cost Driver:** '{max_cost_driver_type}' contributed ₹{max_cost_driver_value:,.2f} to the total cost.")
    else:
        insights.append("No allocation data found to generate detailed insights.")
        
    insights.append("\n**Summary:** The optimization aimed to meet project hour requirements at minimal cost, distributing work among available labor types. The above highlights key aspects of this allocation.")
    return insights


def generate_ppg_insights(ppg_results):
    insights = []
    if not ppg_results or 'filtered_df' not in ppg_results:
        insights.append("PPG analysis has not been performed or data is unavailable.")
        return insights

    filtered_df = ppg_results.get('filtered_df')
    selected_range = ppg_results.get('selected_ppg_range', ('N/A', 'N/A'))
    metric_column = 'PPG_Level' # Assuming this is consistent

    insights.append(f"**PPG Data Analyzed:** Filtered for PPG Levels between {selected_range[0]} and {selected_range[1]}.")
    
    if filtered_df is None or filtered_df.empty:
        insights.append("No PPG data available within the selected filter range.")
        return insights

    insights.append(f"**Number of Data Points Analyzed:** {len(filtered_df)} records.")
    
    avg_ppg = filtered_df[metric_column].mean()
    median_ppg = filtered_df[metric_column].median()
    min_ppg = filtered_df[metric_column].min()
    max_ppg = filtered_df[metric_column].max()
    std_ppg = filtered_df[metric_column].std()

    insights.append(f"**Overall PPG Statistics:**")
    insights.append(f"  - Average PPG Level: {avg_ppg:.2f}")
    insights.append(f"  - Median PPG Level: {median_ppg:.2f}")
    insights.append(f"  - Range of PPG Levels: {min_ppg:.2f} to {max_ppg:.2f}")
    insights.append(f"  - Standard Deviation: {std_ppg:.2f} (indicates spread of PPG values).")

    # High PPG Threshold (example: 75th percentile)
    high_ppg_threshold = filtered_df[metric_column].quantile(0.75)
    num_high_ppg = filtered_df[filtered_df[metric_column] > high_ppg_threshold].shape[0]
    percent_high_ppg = (num_high_ppg / len(filtered_df)) * 100 if len(filtered_df) > 0 else 0
    insights.append(f"**High PPG Levels:** Approximately {percent_high_ppg:.1f}% of readings were above {high_ppg_threshold:.2f} (75th percentile), potentially indicating higher stress/fatigue for this segment.")

    # Activity Level Insights
    if 'Activity_Level' in filtered_df.columns and 'ppg_activity_scatter_buffer' in ppg_results:
        # Simple correlation observation (not statistical test here for simplicity)
        # For a formal test, you'd use scipy.stats.pearsonr
        # Here, we'll just observe means for illustrative purposes
        if pd.api.types.is_numeric_dtype(filtered_df['Activity_Level']):
            median_ppg_for_split = filtered_df[metric_column].median()
            low_ppg_group_activity = filtered_df[filtered_df[metric_column] <= median_ppg_for_split]['Activity_Level'].mean()
            high_ppg_group_activity = filtered_df[filtered_df[metric_column] > median_ppg_for_split]['Activity_Level'].mean()
            insights.append(f"**PPG vs. Activity Level:**")
            insights.append(f"  - Avg. Activity for Lower PPG Group (≤{median_ppg_for_split:.2f}): {low_ppg_group_activity:.2f}")
            insights.append(f"  - Avg. Activity for Higher PPG Group (>{median_ppg_for_split:.2f}): {high_ppg_group_activity:.2f}")
            if high_ppg_group_activity < low_ppg_group_activity:
                 insights.append("  - Observation: Workers with higher PPG levels tended to have lower average activity levels.")
            elif high_ppg_group_activity > low_ppg_group_activity:
                 insights.append("  - Observation: Workers with higher PPG levels tended to have higher average activity levels (investigate further).")
            else:
                 insights.append("  - Observation: No clear difference in average activity levels between high/low PPG groups based on this split.")
        else:
            insights.append("**PPG vs. Activity Level:** 'Activity_Level' column is not numeric, cannot perform comparison.")


    # Time-Based Trends Insights
    if 'Timestamp' in filtered_df.columns and 'ppg_time_trend_buffer' in ppg_results:
        try:
            temp_df_time = filtered_df.copy()
            temp_df_time['Timestamp'] = pd.to_datetime(temp_df_time['Timestamp'], errors='coerce')
            temp_df_time.dropna(subset=['Timestamp', metric_column], inplace=True)
            if not temp_df_time.empty:
                time_trend = temp_df_time.groupby(temp_df_time['Timestamp'].dt.hour)[metric_column].mean()
                if not time_trend.empty:
                    peak_hour = time_trend.idxmax()
                    peak_ppg = time_trend.max()
                    low_hour = time_trend.idxmin()
                    low_ppg = time_trend.min()
                    insights.append(f"**Time-Based PPG Trends:**")
                    insights.append(f"  - Peak Average PPG ({peak_ppg:.2f}) observed around hour: {peak_hour}:00.")
                    insights.append(f"  - Lowest Average PPG ({low_ppg:.2f}) observed around hour: {low_hour}:00.")
                    insights.append(f"  - This suggests potential periods of increased and decreased average stress/fatigue during the day.")
        except Exception as e:
            insights.append(f"**Time-Based PPG Trends:** Error processing time trends: {e}")


    insights.append("\n**Summary:** The PPG analysis provides a snapshot of worker physiological states. These insights can help identify patterns in fatigue and inform scheduling or intervention strategies.")
    return insights


def download_reports_page():
    st.title("📄 Download Analysis Reports")
    st.write("Generate a comprehensive PDF report with optimization and PPG dataset analysis, including charts and AI-generated insights (based on descriptive statistics).")

    opt_ready = 'allocation_df' in st.session_state.get('optimization_results', {})
    ppg_ready = 'filtered_df' in st.session_state.get('ppg_results', {})

    if not opt_ready:
        st.warning("⚠️ Please run the Optimization Analysis first to include its results in the report.")
    if not ppg_ready:
        st.warning("⚠️ Please run the PPG Dataset Analysis first to include its results in the report.")

    if st.button("📥 Generate PDF Report", disabled=not (opt_ready or ppg_ready)):
        pdf = FPDF()
        font_family_name = 'DejaVuSans' # Use this consistently

        try:
            font_dir = os.path.dirname(os.path.abspath(__file__))
            regular_font_path = os.path.join(font_dir, 'DejaVuSans.ttf')
            bold_font_path = os.path.join(font_dir, 'DejaVuSans-Bold.ttf')
            italic_font_path = os.path.join(font_dir, 'DejaVuSans-Oblique.ttf')
            bold_italic_font_path = os.path.join(font_dir, 'DejaVuSans-BoldOblique.ttf')

            fonts_to_check = {
                "Regular": regular_font_path,
                "Bold": bold_font_path,
                "Italic (Oblique)": italic_font_path,
                "Bold Italic (BoldOblique)": bold_italic_font_path
            }
            missing_fonts_files = []
            for style_name, path in fonts_to_check.items():
                if not os.path.exists(path):
                    missing_fonts_files.append(f"{style_name}: {os.path.basename(path)}")

            if missing_fonts_files:
                st.error(f"The following font files are missing: {', '.join(missing_fonts_files)}")
                return

            pdf.add_font(font_family_name, '', regular_font_path, uni=True)
            pdf.add_font(font_family_name, 'B', bold_font_path, uni=True)
            pdf.add_font(font_family_name, 'I', italic_font_path, uni=True)
            pdf.add_font(font_family_name, 'BI', bold_italic_font_path, uni=True)

            # --- DEBUGGING STEP ---
            # st.write("Registered fonts in FPDF (check keys):", pdf.fonts)
            # st.write("Current font family:", pdf.font_family)
            # st.write("Current font style:", pdf.font_style)
            # --- END DEBUGGING ---

            pdf.set_font(font_family_name, '', 12) # Set default font

        except RuntimeError as e:
            st.error(f"FPDF RuntimeError during font loading: {e}. Ensure all DejaVuSans TTF files are present and you are using fpdf2.")
            return
        except Exception as e_general:
            st.error(f"An unexpected error occurred during font setup: {e_general}")
            # st.error(f"Registered fonts before error: {pdf.fonts if hasattr(pdf, 'fonts') else 'N/A'}")
            return

        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font(font_family_name, style='B', size=20) # Use variable
        pdf.cell(0, 10, "Railway Resource Optimization & Analysis Report", ln=True, align='C')
        pdf.ln(10)

        pdf_image_width = pdf.w - pdf.l_margin - pdf.r_margin - 10
        if pdf_image_width <= 0: pdf_image_width = 150

        # --- Optimization Section ---
        if opt_ready:
            pdf.set_font(font_family_name, style='B', size=16)
            pdf.cell(0, 10, "I. Optimization Analysis", ln=True)
            pdf.ln(2)

            opt_results = st.session_state.optimization_results

            pdf.set_font(font_family_name, style='B', size=12)
            pdf.multi_cell(0, 7, "Generated Insights (Descriptive Statistics & Templated NLG):", ln=1) # Assumes fpdf2
            pdf.set_font(font_family_name, size=10) # Regular style for insights
            # ... (rest of insights) ...
            opt_insights = generate_optimization_insights(opt_results) # Make sure this function is defined
            for insight_line in opt_insights:
                page_width_insights = pdf.w - 2 * pdf.l_margin
                pdf.multi_cell(page_width_insights, 6, f"- {insight_line.replace('**', '')}", border=0, align='L', fill=False, ln=1)
            pdf.ln(3)


            pdf.set_font(font_family_name, style='B', size=12)
            pdf.cell(0, 10, "Optimization Charts:", ln=1)

            alloc_chart_buffer = opt_results.get('allocation_chart_buffer')
            if alloc_chart_buffer:
                pdf.ln(2)
                # THIS IS THE LINE FROM THE TRACEBACK
                pdf.set_font(font_family_name, style='I', size=10) # Italic for chart caption
                pdf.multi_cell(0, 5, "Labor Hours Allocation Chart", ln=1, align='C') # Assumes fpdf2
                # ... (rest of chart embedding) ...
                alloc_chart_buffer.seek(0)
                img_h = get_image_height_for_pdf(alloc_chart_buffer, pdf_image_width)
                alloc_chart_buffer.seek(0)
                pdf.image(name=alloc_chart_buffer, x=pdf.l_margin + 5, w=pdf_image_width, h=img_h, type='PNG')
                pdf.ln(img_h + 3)


            cost_dist_chart_buffer = opt_results.get('cost_dist_chart_buffer')
            if cost_dist_chart_buffer:
                pdf.ln(2)
                pdf.set_font(font_family_name, style='I', size=10) # Italic for chart caption
                pdf.multi_cell(0, 5, "Cost Distribution by Labor Type Chart", ln=1, align='C') # Assumes fpdf2
                # ... (rest of chart embedding) ...
                cost_dist_chart_buffer.seek(0)
                img_h = get_image_height_for_pdf(cost_dist_chart_buffer, pdf_image_width)
                cost_dist_chart_buffer.seek(0)
                pdf.image(name=cost_dist_chart_buffer, x=pdf.l_margin + 5, w=pdf_image_width, h=img_h, type='PNG')
                pdf.ln(img_h + 3)

        else:
            pdf.set_font(font_family_name, style='I', size=12)
            pdf.multi_cell(0, 10, "Optimization Analysis data not available for this report.", ln=True) # Assumes fpdf2
        pdf.ln(5)

        # --- PPG Analysis Section (similar changes for font_family_name and styles) ---
        if ppg_ready:
            if pdf.get_y() > (pdf.h - pdf.b_margin - 70):
                 pdf.add_page()
            else:
                 pdf.ln(5)

            pdf.set_font(font_family_name, style='B', size=16)
            pdf.cell(0, 10, "II. PPG Dataset Analysis", ln=True)
            pdf.ln(2)

            ppg_results_data = st.session_state.ppg_results

            pdf.set_font(font_family_name, style='B', size=12)
            pdf.multi_cell(0, 7, "Generated Insights (Descriptive Statistics & Templated NLG):", ln=1) # Assumes fpdf2
            pdf.set_font(font_family_name, size=10)
            ppg_insights = generate_ppg_insights(ppg_results_data) # Make sure this function is defined
            for insight_line in ppg_insights:
                page_width_insights = pdf.w - 2 * pdf.l_margin
                pdf.multi_cell(page_width_insights, 6, f"- {insight_line.replace('**', '')}", border=0, align='L', fill=False, ln=1)
            pdf.ln(3)

            pdf.set_font(font_family_name, style='B', size=12)
            pdf.cell(0, 10, "PPG Analysis Charts:", ln=1)

            ppg_dist_chart = ppg_results_data.get('ppg_dist_chart_buffer')
            if ppg_dist_chart:
                pdf.ln(2)
                pdf.set_font(font_family_name, style='I', size=10)
                pdf.multi_cell(0, 5, "PPG Level Distribution Chart", ln=1, align='C') # Assumes fpdf2
                ppg_dist_chart.seek(0)
                img_h = get_image_height_for_pdf(ppg_dist_chart, pdf_image_width)
                ppg_dist_chart.seek(0)
                pdf.image(name=ppg_dist_chart, x=pdf.l_margin + 5, w=pdf_image_width, h=img_h, type='PNG')
                pdf.ln(img_h + 3)

            # ... (repeat for other PPG charts: ppg_activity_scatter, ppg_time_trend) ...
            ppg_activity_scatter = ppg_results_data.get('ppg_activity_scatter_buffer')
            if ppg_activity_scatter:
                pdf.ln(2)
                pdf.set_font(font_family_name, style='I', size=10)
                pdf.multi_cell(0, 5, "PPG Level vs. Activity Level Chart", ln=1, align='C')
                ppg_activity_scatter.seek(0)
                img_h = get_image_height_for_pdf(ppg_activity_scatter, pdf_image_width)
                ppg_activity_scatter.seek(0)
                pdf.image(name=ppg_activity_scatter, x=pdf.l_margin + 5, w=pdf_image_width, h=img_h, type='PNG')
                pdf.ln(img_h + 3)

            ppg_time_trend = ppg_results_data.get('ppg_time_trend_buffer')
            if ppg_time_trend:
                pdf.ln(2)
                pdf.set_font(font_family_name, style='I', size=10)
                pdf.multi_cell(0, 5, "PPG Level Trends Over Time Chart", ln=1, align='C')
                ppg_time_trend.seek(0)
                img_h = get_image_height_for_pdf(ppg_time_trend, pdf_image_width)
                ppg_time_trend.seek(0)
                pdf.image(name=ppg_time_trend, x=pdf.l_margin + 5, w=pdf_image_width, h=img_h, type='PNG')
                pdf.ln(img_h + 3)

        else:
            pdf.set_font(font_family_name, style='I', size=12)
            pdf.multi_cell(0, 10, "PPG Analysis data not available for this report.", ln=True) # Assumes fpdf2
        pdf.ln(5)

        pdf_output_bytes = bytes(pdf.output(dest="S"))
        st.download_button(
            label="📥 Download Generated Report as PDF",
            data=pdf_output_bytes,
            file_name="Comprehensive_Analysis_Report.pdf",
            mime="application/pdf"
        )




def future_page():
    st.title("🔧 Future Considerations & Social Impact")
    st.write("This project goes beyond traditional cost optimization by integrating worker well-being into decision-making.")
    
    st.subheader("💡 Why Combine PPG Analysis with Cost Optimization?")
    st.markdown("- **Worker Well-being First:** Ensuring railway workers remain efficient and safe by monitoring fatigue trends.")
    st.markdown("- **Reducing Accidents & Errors:** High fatigue levels are linked to reduced attention, increasing risks in railway operations.")
    st.markdown("- **Long-Term Workforce Sustainability:** Organizations investing in employee well-being see increased productivity and reduced turnover.")
    
    st.subheader("🌍 The Social Cause Behind This Initiative")
    st.markdown("- **A Step Towards Safer Railways:** Fatigue-induced errors have led to railway mishaps globally. Our system helps prevent them.")
    st.markdown("- **Balancing Efficiency with Human Needs:** Instead of treating labor purely as a cost factor, we account for physiological stress and optimize shifts accordingly.")
    st.markdown("- **Leading by Example:** This project sets a precedent for industries to integrate data-driven health insights into workforce planning.")
    
    st.subheader("🚀 A Noble Step Forward")
    st.markdown("- **Beyond Profitability:** We chose to integrate PPG analysis, knowing it would add complexity, but believing it was the right thing to do.")
    st.markdown("- **Ethical AI for Workforce Management:** By analyzing fatigue alongside cost, we ensure that optimization isn't just about money, but also about people.")
    st.markdown("- **Future Scope:** Expanding this model to include real-time fatigue alerts and AI-driven shift recommendations.")
    
    st.write("🔹 *This initiative is more than just an optimization model—it’s a commitment to safer, smarter, and more humane workforce planning.*")

if __name__ == "__main__":
    main()