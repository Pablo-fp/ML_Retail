import itertools
import streamlit as st
import pandas as pd
import numpy as np
import numpy_financial as npf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
from sklearn.multioutput import RegressorChain
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split


# Config the page tittle on the browser
st.set_page_config(
    page_title='Retail Real Estate App', layout="wide")


sidebar = st.sidebar
header = st.container()
dataset = st.container()
model = st.container()
costs = st.container()
analysis = st.container()


# Function to load the data an store it in cache


@st.cache
def GetData(filename):
    data = pd.read_csv(filename)
    return data


with sidebar:
    # Config a sidebar with link to my sites
    void, picture, void = (st.columns(3))
    profile = Image.open('Data/linkedin.png')
    picture.image(profile)

    st.markdown("""
    You can follow me in [LinkedIn](https://www.linkedin.com/in/pablo-fernandez-perez/) for more content.
     """)
    st.write('---')

with dataset:
    # Invented Retail dataset
    data = GetData('Data/Retail_Dataset.csv')

with model:

    # Spliting target variable and independent variables
    X = data[["Total_area", "Shop_window_area"]]
    y = data.drop(["Shop_name", "Total_area", "Secondary_area",
                   "Shop_window_area", "Total_cost"], axis=1)

    # st.write(X.head())
    # st.write(y.head())

    # st.write(X.shape)
    # st.write(y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=4)

    chain = RegressorChain(LinearRegression()).fit(X_train, y_train)
    # Model prediction on train data ###################################################################################################################
    y_pred = chain.predict(X_train)
    # Predicting Test data with the model ################################################################################################################
    y_test_pred = chain.predict(X_test)
    acc_rf = metrics.r2_score(y_test, y_test_pred)


with header:
    # LOAD HEADER
    image = Image.open('Data/retail_illustration.png')
    st.image(image)
    st.markdown("""
    # Real Estate App for a Retail Business 
    This simple webapp uses an invented retail dataset to build a machine learning model that predicts several of the features, including the distribution of areas inside the store and the cost of construction, presenting at the end an economical analysis of the future inversion. It´s a basic example of what could become a useful tool for expansion taking advantage of all the accumulated data within organisations.   
    (Cost and sales in $* an imaginary currency)
    """)
    with st.expander("See more of the dataset used"):
        # Load dataset´s head

        st.markdown(""" 
            ## DATASET
            I am using a fictional dataset of an hypothetical retail company with surfaces, construction costs and sales records.  
            A real dataset would have to be processed before you could use it, filtering and cleaning the data also normalising it to be able to compare it. The records would need to be clasified by type of store if needed, also construction costs would have to be updated to the current year by the accumalated inflation rates of different construction years (inflation normalization) and finally if you are comparing stores in different countries or areas with different price levels you can normalize it by using different metrics as the "Purchasing power parity index".  
            This dataset has been created multiplying a feature by several invented rates and adding some noise to each value to make them slightly different. Below you can see the first ten rows:
            """)

        st.write(data.head(10))
        # Load dataset´s shape
        n_rows = data.shape[0]
        n_columns = data.shape[1]
        shape = [n_rows, n_columns]
        st.write('This dataset has {0} rows and {1} columns.'.format(*shape))

        url = "https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.RegressorChain.html"
        st.markdown("""
        ## MODEL
        The method used to create the dataset is a succesion of correlated linear regressions. I am using a **Multiouput Regressor Chain**, if you want to know more, check out this [link](%s).
        """ % url)
        st.write('**Model Evaluation Metrics of the chained Linear Regression**')

        st.write('**R^2**:', acc_rf, '- It is a measure of the linear relationship between X and Y. It is interpreted as the proportion of the variance in the dependent variable that is predictable from the independent variable.')
        st.write('**Adjusted R^2**:', 1 - (1-metrics.r2_score(y_test, y_test_pred))
                 * (len(y_test)-1)/(len(y_test)-X_test.shape[1]-1), '- The adjusted R-squared compares the explanatory power of regression models that contain different numbers of predictors.')
        st.write('**MAE**:', metrics.mean_absolute_error(y_test, y_test_pred),
                 '- It is the mean of the absolute value of the errors. It measures the difference between two continuous variables, here actual and predicted values of y.')
        st.write('**MSE**:', metrics.mean_squared_error(y_test, y_test_pred),
                 '- The mean square error (MSE) is just like the MAE, but squares the difference before summing them all instead of using the absolute value.')
        st.write('**RMSE**:', np.sqrt(metrics.mean_squared_error(y_test, y_test_pred)),
                 '- The root-mean-square error (RMSE) is a frequently used measure of the differences between values (sample or population values) predicted by a model or an estimator and the values observed.')
        st.markdown("""
        (Notice that as the dataset is totally invented I haven´t worked on the model to improve performance, this is what I got from the raw data).
        """)
    st.write('---')


with costs:
    st.markdown("""
    ## PREDICTED DATA
    Ingress the input to obtain the predicted construction costs, surfaces and revenue of the store.
     """)

    # Dividing the screen in columns; input, surfaces, costs and earnings #################################################################################
    input_column, surfaces_column, costs_column, earnings_column = (
        st.columns([2, 2, 2, 3]))

    # Input Parameters in input_column ####################################################################################################################
    input_column.header('Input Parameters')

    # Function with sliders and user inputs for Total Area and Shop Window Area ###########################################################################
    def user_input_features():
        Total_area = input_column.slider('Total area: Total net area of the local (in m\u00B2)', int(
            X.Total_area.min()), int(X.Total_area.max()), int(600), 1)
        Shop_window_area = input_column.slider('Shop window area: Façade of the local (in m\u00B2)', int(
            30), int(X.Shop_window_area.max()), int(140), 1)

        data = {'Total_area': Total_area,
                'Shop_window_area': Shop_window_area}
        features = pd.DataFrame(data, index=[0])
        return features

    df_features = user_input_features()
    prediction = chain.predict(df_features)
    inflation_rate = input_column.slider(
        'Construction inflation rate: Compared to previous year (in %)', int(0), int(100), 0, 1)
    input_column.write('---')

    # Sliders for percentage of cost (scope of the works) ##################################################################
    with input_column.expander("Choose the scope of the works here:"):
        st.write(
            "In case the project is a partial renovation, you can choose the scope of the works by chapter in percentage in relation with a brand new store (by defect 100%).")
        demolition_scope = st.slider(
            'Demolition scope (%)', 0, 100, 100, 1)
        facilities_scope = st.slider(
            'Facilites scope (%)', 0, 100, 100, 1)
        internal_walls_scope = st.slider(
            'Internal walls scope (%)', 0, 100, 100, 1)
        carpentry_scope = st.slider(
            'Carpentry scope (%)', 0, 100, 100, 1)
        facade_scope = st.slider(
            'Façade scope (%)', 0, 100, 100, 1)
        finishes_scope = st.slider(
            'Finishes scope (%)', 0, 100, 100, 1)
        furniture_scope = st.slider(
            'Furniture scope (%)', 0, 100, 100, 1)

    # Print Surfaces in surfaces_column ###################################################################################################################
    surfaces_column.header('Estimated surfaces')

    surfaces_column.metric(label="Stockroom area:", value=str(
        int(prediction[0, 0])) + " m\u00B2")

    secondary_area = int(df_features["Total_area"] -
                         int(prediction[0, 0]) - int(prediction[0, 1]))
    surfaces_column.metric(label="Secondary area:",
                           value=str(secondary_area) + " m\u00B2")

    surfaces_column.metric(label="Sales area:", value=str(
        int(prediction[0, 1])) + " m\u00B2")

    surfaces_column.metric(label="Total area:",
                           value=str(int(df_features["Total_area"])) + " m\u00B2")

    # Print Costs in cost_column ##########################################################################################################################
    costs_column.header('Estimated costs')
    demolition_cost = int((int(prediction[0, 2]) + (
        inflation_rate * 0.01 * (int(prediction[0, 2])))) * demolition_scope * 0.01)
    costs_column.metric(label="Demolition cost:",
                        value="{:,}".format(demolition_cost) + " $*")

    facilities_cost = int(int((prediction[0, 3]) + (
        inflation_rate * 0.01 * (int(prediction[0, 3])))) * facilities_scope * 0.01)
    costs_column.metric(label="Facilities cost:",
                        value="{:,}".format(facilities_cost) + " $*")

    internal_walls_cost = int(
        int((prediction[0, 4]) + (
            inflation_rate * 0.01 * (int(prediction[0, 4])))) * internal_walls_scope * 0.01)
    costs_column.metric(label="Internal walls cost:",
                        value="{:,}".format(internal_walls_cost) + " $*")

    carpentry_cost = int((int(prediction[0, 5]) + (
        inflation_rate * 0.01 * (int(prediction[0, 5])))) * carpentry_scope * 0.01)
    costs_column.metric(label="Carpentry cost:",
                        value="{:,}".format(carpentry_cost) + " $*")

    Facade_rate = 600
    Facade_randomness = 300
    facade_cost = int(
        (int(df_features['Shop_window_area']) * Facade_rate + (int(df_features['Shop_window_area']) * Facade_rate * inflation_rate * 0.01)) * facade_scope * 0.01)
    costs_column.metric(label="Façade cost:",
                        value="{:,}".format(facade_cost) + " $*")

    finishes_cost = int((int(prediction[0, 7]) + (
        inflation_rate * 0.01 * (int(prediction[0, 7])))) * finishes_scope * 0.01)
    costs_column.metric(label="Finishes cost:",
                        value="{:,}".format(finishes_cost) + " $*")

    furniture_cost = int((int(prediction[0, 8]) + (
        inflation_rate * 0.01 * (int(prediction[0, 8])))) * furniture_scope * 0.01)
    costs_column.metric(label="Furniture cost:",
                        value="{:,}".format(furniture_cost) + " $*")

    total_cost = demolition_cost + facilities_cost + internal_walls_cost + \
        carpentry_cost + facade_cost + finishes_cost + furniture_cost
    costs_column.metric(label="Total cost:",
                        value="{:,}".format(total_cost) + " $*")

    # Print Earnings in earnings_column ####################################################################################################################
    earnings_column.header('Estimated Sales revenue')
    annual_sales = int(prediction[0, 9])
    earnings_column.metric(label="Annual sales revenue:",
                           value="{:,}".format(annual_sales) + " $*")

    # Print Graph Surfaces in earnings_column ############################################################################################################
    data_surfaces = [['Stockroom area', int(prediction[0, 0])], [
        'Secondary area', secondary_area], ['Sales area', int(prediction[0, 1])]]
    df_surfaces = pd.DataFrame(data_surfaces, columns=['Surface', 'm\u00B2'])
    fig1 = px.pie(df_surfaces, values='m\u00B2',
                  names='Surface', title='Surfaces', hole=.5,
                  color_discrete_sequence=px.colors.sequential.Brwnyl)
    fig1.update_traces(textposition='inside', textinfo='percent+label')
    fig1.update_layout(width=400, height=350)
    earnings_column.write(fig1)

    # Print Graph Costs in earnings_column #################################################################################################################
    data_costs = [['Demolition cost', demolition_cost], ['Facilities cost', facilities_cost], ['Internal walls cost', internal_walls_cost], [
        'Façade cost', facade_cost], ['Finishes cost', finishes_cost], ['Furniture cost', furniture_cost]]
    df_costs = pd.DataFrame(data_costs, columns=['Costs', '$'])
    fig2 = px.pie(df_costs, values='$',
                  names='Costs', title='Costs', hole=.5,
                  color_discrete_sequence=px.colors.sequential.Brwnyl)
    fig2.update_traces(textposition='inside', textinfo='percent+label')
    fig2.update_layout(width=400, height=350)
    earnings_column.write(fig2)

    st.write('---')

with analysis:
    st.markdown("""
    ## BUSINESS ANALYSIS
    Ingress the input to analyze the business case.""")
    # Dividing the screen in columns; input, analysis #################################################################################
    case_column, analysis_column, graphs_column = (
        st.columns([2, 2, 5]))

    # Input from the user of business cost #############################################################################################
    case_column.header('Input Parameters')
    annual_rent = case_column.number_input(
        'Insert the "Annual Rent" (in $*):', min_value=0, max_value=None, value=150000)
    annual_labor = case_column.number_input(
        'Insert the "Annual Labor Costs" (in $*):', min_value=0, max_value=None, value=200000)
    other_costs = case_column.number_input(
        'Insert "Other annual costs concept" (in $*):', min_value=0, max_value=None, value=0)
    periods = int(case_column.number_input(
        'Insert the "Period" of time you want to study (in years):', min_value=0, max_value=None, value=5))
    discount_rate = case_column.number_input(
        'Insert the "Discount Rate" or return that could be earn in alternative investments (in %):', min_value=0, max_value=None, value=6)
    case_column.write('---')

    # Modifiers of the user for costs and earnings######################################################################################
    with case_column.expander("Add some modification to the predicted construction and revenue here:"):
        st.write(
            ' Modify the estimated construction cost and annual revenue with your own amount (positive or negative in $*).')
        construction_modifier = st.number_input(
            'Construction cost modifier:', min_value=None, max_value=None, value=0)
        revenue_modifier = st.number_input(
            'Sales revenue modifier:', min_value=None, max_value=None, value=0)
    st.write('---')

    with analysis_column:
        total_construction_cost = demolition_cost + facilities_cost + internal_walls_cost + \
            facade_cost + finishes_cost + furniture_cost + \
            construction_modifier + carpentry_cost
        total_revenue = annual_sales + revenue_modifier
        annual_costs = annual_rent + annual_labor + other_costs

        # Analysis_column ####################################################################################################################
        analysis_column.header('Financial metrics')

        # ROI per year! in analysis_column ####################################################################################################################
        roi_py = round(((((((total_revenue - annual_costs) * periods) -
                           (total_construction_cost))) / (total_construction_cost)) * 100)/periods, 2)
        roi_py_message = "ROI (Return of Investment) mean per year:"
        analysis_column.metric(label=roi_py_message, value=str(roi_py) + " %")

        # ROI in analysis_column ####################################################################################################################

        roi = round((((((total_revenue - annual_costs) * periods) -
                       (total_construction_cost))) / (total_construction_cost)) * 100, 2)
        roi_message = "ROI (Return of Investment) for " + \
            str(periods) + " years:"
        analysis_column.metric(label=roi_message, value=str(roi) + " %")

        # NPV in analysis_column ####################################################################################################################
        cashflow = (total_revenue - annual_costs)
        cashflows = [cashflow for _ in range(periods)]
        cashflows.insert(0, total_construction_cost*-1)
        net_present_value = npf.npv(discount_rate*0.01, cashflows)
        npv_message = "NPV (Net Present Value) for " + str(periods) + " years:"
        analysis_column.metric(
            label=npv_message, value="{:,}".format(int(net_present_value)) + " $*")

        # IRR in analysis_column ####################################################################################################################

        internal_rate_return = round(npf.irr(cashflows)*100, 2)
        irr_message = "IRR (Internal Rate of Return) for " + \
            str(periods) + " years:"
        analysis_column.metric(
            label=irr_message, value=str(internal_rate_return) + " %")

        # BEP of the Investment ####################################################################################################################
        break_even_period = round((total_construction_cost / cashflow), 2)
        bes_message = "Break-even point achieved in:"
        analysis_column.metric(
            label=bes_message, value=str(break_even_period) + " years")

        break_even_sales = total_construction_cost + \
            (annual_costs * break_even_period)
        bes_message = "Total Sales in the period for break-even:"
        analysis_column.metric(
            label=bes_message, value="{:,}".format(break_even_sales) + " $*")

        break_even_sales_year = (total_construction_cost +
                                 (annual_costs * break_even_period))/periods
        besy_message = "Sales per year for break-even in the period:"
        analysis_column.metric(
            label=besy_message, value="{:,}".format(break_even_sales_year) + " $*")

        # Profit and loss statement - Waterfall Chart#################################################################################################################

        # Quantities for the graph => "y"
        annual_cashflow = [annual_costs*-1,
                           total_revenue, total_revenue - annual_costs]
        annual_cashflows = [annual_cashflow for _ in range(periods)]
        flat_annual_cashflows = list(itertools.chain(*annual_cashflows))
        flat_annual_cashflows.insert(0, total_construction_cost*-1)

        # Type of column for the graph => "measure"
        construction_measure = "relative"
        annual_measure = ["relative", "relative", "total"]
        annual_measures = [annual_measure for _ in range(periods)]
        flat_annual_measures = list(itertools.chain(*annual_measures))
        flat_annual_measures.insert(0, construction_measure)

        # Name of column for the graph => "x"
        construction_name = "Construction Cost"
        annual_names = [["Operational Costs Year " +
                         str(i+1), "Revenue Year " + str(i+1), "Profit Year " + str(i+1)] for i in range(periods)]
        flat_annual_names = list(itertools.chain(*annual_names))
        flat_annual_names.insert(0, construction_name)

        # waterfall chart
        water_fig = go.Figure(go.Waterfall(
            name="20", orientation="v",
            measure=flat_annual_measures,

            x=flat_annual_names,
            textposition="outside",

            y=flat_annual_cashflows,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))

        water_fig.update_layout(
            title="Profit and Loss statement", showlegend=True)

        graphs_column.write(water_fig)

        # Annual Opex-Capex + Period Opex-Capex - Pie chart in Subplots##############################################################################
        labels = ["CAPEX", "OPEX"]
        data_fig3_periodop = [total_construction_cost, annual_costs*periods]
        data_fig3_yearop = [total_construction_cost/periods, annual_costs]
        colors = ['rgb(220, 197, 164)', 'rgb(236, 230, 206)']
        # Create subplots: use 'domain' type for Pie subplot
        fig3 = make_subplots(rows=1, cols=2, specs=[
                             [{'type': 'domain'}, {'type': 'domain'}]])
        fig3.add_trace(go.Pie(labels=labels, values=data_fig3_periodop, name="Capex & Opex in the period", marker_colors=colors),
                       1, 1)
        fig3.add_trace(go.Pie(labels=labels, values=data_fig3_yearop, name="Capex & Opex per year in the period"),
                       1, 2)
        # Use `hole` to create a donut-like pie chart
        fig3.update_traces(hole=.5, hoverinfo="label+value+name")
        fig3.update_layout(
            title_text="Capex & Opex",
            # Add annotations in the center of the donut pies.
            annotations=[dict(text='Period', x=0.17, y=0.5, font_size=20, showarrow=False),
                         dict(text='Per Year', x=0.85, y=0.5, font_size=20, showarrow=False)])
        graphs_column.write(fig3)
