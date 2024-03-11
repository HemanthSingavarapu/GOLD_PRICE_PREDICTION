'''
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import math

col1, col2 = st.columns([2, 1])  # Adjust the ratio as per need

with col1:
    # Streamlit app
    st.title('Price Prediction App')
    # File upload
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, parse_dates=True, index_col='Date')
        st.write(df.head())

        # Identify non-date, integer, or float type columns for analysis
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_column = st.selectbox('Select a column for analysis:', numeric_cols)

        if selected_column:
            # Proceed with analysis and predictions using the selected column
            data = df[[selected_column]]
            st.subheader(f'Original {selected_column} Values')
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df.index, y=data[selected_column], mode='lines', name=selected_column))
            fig.update_layout(title=f'{selected_column} History', xaxis_title='Date', yaxis_title=f'{selected_column} Value')
            st.plotly_chart(fig, use_container_width=True)

            # Prepare data
            dataset = data.values
            training_data_len = math.ceil(len(dataset) * .8)

            # Scale the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(dataset)

            # Create training dataset
            train_data = scaled_data[0:training_data_len, :]
            X_train = []
            y_train = []
            for i in range(60, len(train_data)):
                X_train.append(train_data[i-60:i, 0])
                y_train.append(train_data[i, 0])

            # Convert to numpy and then reshape
            X_train, y_train = np.array(X_train), np.array(y_train)

            # Getting user input for model parameters
            epochs = st.slider('Number of epochs', 1,50, 3)
            batch_size = st.slider('Batch Size', 1, 128, 5)


models = {
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression': SVR(),
    'Gradient Boosting': GradientBoostingRegressor()
}

# Predict future price
st.markdown("<h2>Select number of days to predict into the future</h2>", unsafe_allow_html=True)
future_days = st.slider('Select number of days to predict into the future', min_value=1, max_value=30, value=7)

if st.button('Predict Future Prices'):
    for selected_algorithm, model in models.items():
        with st.spinner(f'Predicting future prices using {selected_algorithm}...'):
            last_60_days = data[-60:].values
            last_60_days_scaled = scaler.transform(last_60_days)
            current_batch = last_60_days_scaled.reshape((1, 60))

            # Initialize and fit the model
            model.fit(X_train, y_train)

            predictions = []

            for _ in range(int(future_days)):
                pred_price = model.predict(current_batch)
                current_batch = np.append(current_batch[:, 1:], [pred_price], axis=1)
                predictions.append(pred_price[0])

            predictions = np.array(predictions).reshape(-1, 1)
            predictions = scaler.inverse_transform(predictions)

            # Display future predictions for the selected model
            st.subheader(f'{selected_algorithm} Predictions')
            for i, price in enumerate(predictions):
                st.write(f"Day {i+1}: Predicted {selected_column} Value: {price[0]:.2f}")
'''
# Rest of the code remains unchanged
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import math

# Streamlit app
st.title('Price Prediction App')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file, parse_dates=True, index_col='Date')
    st.write(df.head())

    # Identify non-date, integer, or float type columns for analysis
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    selected_column = st.selectbox('Select a column for analysis:', numeric_cols)

    if selected_column:
        # Proceed with analysis and predictions using the selected column
        data = df[[selected_column]]
        st.subheader(f'Original {selected_column} Values')
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=data[selected_column], mode='lines', name=selected_column))
        fig.update_layout(title=f'{selected_column} History', xaxis_title='Date', yaxis_title=f'{selected_column} Value')
        st.plotly_chart(fig, use_container_width=True)

        # Prepare data
        dataset = data.values
        training_data_len = math.ceil(len(dataset) * .8)

        # Scale the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(dataset)

        # Create training dataset
        train_data = scaled_data[0:training_data_len, :]
        X_train = []
        y_train = []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i-60:i, 0])
            y_train.append(train_data[i, 0])

        # Convert to numpy and then reshape
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Define available ML models (placed before dropdown for correct usage)
        models = {
            'Random Forest': RandomForestRegressor(),
            'Support Vector Regression': SVR(),
            'Gradient Boosting': GradientBoostingRegressor()
        }

        # Dropdown for selecting ML algorithm
        selected_algorithm = st.selectbox('Select ML Algorithm:', models.keys())

        # Get user input for model parameters
        epochs = st.slider('Number of epochs', 1, 50, 3)
        batch_size = st.slider('Batch Size', 1, 128, 5)

        # Predict future price
        st.markdown("<h2>Select the number of days to predict into the future</h2>", unsafe_allow_html=True)
        future_days = st.slider('Select number of days to predict into the future', min_value=1, max_value=30, value=7)

        if st.button('Predict Future Prices'):
            with st.spinner(f'Predicting future prices using {selected_algorithm}...'):
                model = models[selected_algorithm]  # Retrieve the selected model

                last_60_days = data[-60:].values
                last_60_days_scaled = scaler.transform(last_60_days)
                current_batch = last_60_days_scaled.reshape((1, 60))

                # Initialize and fit the model
                model.fit(X_train, y_train)

                predictions = []

                for _ in range(int(future_days)):
                    pred_price = model.predict(current_batch)
                    current_batch = np.append(current_batch[:, 1:], [pred_price], axis=1)
                    predictions.append(pred_price[0])  # Append the predicted price

                predictions = np.array(predictions).reshape(-1, 1)
                predictions = scaler.inverse_transform(predictions)

                # Calculate accuracy
                actual_values = data[-len(predictions):].values
                mse = mean_squared_error(actual_values, predictions)
                rmse = math.sqrt(mse)
                r2 = r2_score(actual_values, predictions)

                # Display future predictions and accuracy for the selected model
                columns1, columns2 = st.columns(2)

                with columns1:
                    
                    st.subheader('Accuracy Metrics')
                    st.write(f'Mean Squared Error (MSE): {mse:.2f}')
                    st.write(f'Root Mean Squared Error (RMSE): {rmse:.2f}')
                    st.write(f'R-squared (R2): {r2:.2f}')
                    correlation_matrix = np.corrcoef(predictions.flatten(), actual_values.flatten())
                    fig_corr_matrix = ff.create_annotated_heatmap(
                        z=correlation_matrix,
                        x=['Predicted', 'Actual'],
                        y=['Predicted', 'Actual'],
                        colorscale='Viridis'
                    )
                    fig_corr_matrix.update_layout(title='Correlation Matrix for Accuracy Metrics')
                    st.plotly_chart(fig_corr_matrix, use_container_width=True)

                    
                    
                    

                with columns2:
                    ## Display predictions in a table
                    st.subheader(f'{selected_algorithm} Predictions')
                    predictions_table = pd.DataFrame({'Day': range(1, len(predictions) + 1),
                                  'Predicted Value': predictions.flatten(),
                                  'Actual Value': actual_values.flatten()})
                    st.table(predictions_table)
                    
                    # Visualize predictions vs actual values
            fig_predictions = go.Figure()
            fig_predictions.add_trace(go.Scatter(x=df.index[-len(predictions):], y=predictions.flatten(), mode='lines', name='Predicted'))
            fig_predictions.add_trace(go.Scatter(x=df.index[-len(predictions):], y=actual_values.flatten(), mode='lines', name='Actual'))
            fig_predictions.update_layout(title=f'{selected_algorithm} Predictions vs Actual', xaxis_title='Date', yaxis_title=f'{selected_column} Value')
            st.plotly_chart(fig_predictions, use_container_width=True)
