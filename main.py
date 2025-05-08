import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from keras import Input
import plotly.graph_objs as go
from plotly.offline import plot



def Convert_Volume(val):
    if isinstance(val, str):
        val = val.replace(",", "").strip()
        if val.endswith("B"):
            return float(val[:-1]) * 1e9
        elif val.endswith("M"):
            return float(val[:-1]) * 1e6
        else:
            return float(val)
    elif isinstance(val, (int, float)):
        return float(val)  # already numeric
    return None


#Populate any missing data with the data from previous trading day
def ForwardFillData():
    
    #Import and Parse the Date column and define the format of date
    df = pd.read_csv("FTSE100.csv", parse_dates=["Date"], dayfirst=True)

    #Modify Date series 
    df.set_index("Date", inplace=True)
    all_business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_business_days)
    df.index.name = 'Date'

    #Do Forward fill so the missing dates have the price,volume and percentage as same as the last business day 
    df.ffill(inplace=True)
    
    #Convert volume to a float instead of B and M denoting Billion and million respectively
    df["Vol."] = df["Vol."].apply(Convert_Volume).astype(float)
    
    
    #Sort and write output to a csv file to visualisation
    df = df.sort_index(ascending=False)
    df.to_csv('FTSE100_DataProcessed.csv', index=True)


def VisualiseData():
    # Load CSV file
    df = pd.read_csv("FTSE100_DataProcessed.csv", parse_dates=["Date"])

    # Clean and sort
    df["Price"] = df["Price"].str.replace(",", "").astype(float)
    df = df.sort_values("Date")

    
    price_line = go.Scatter(
        x=df["Date"],
        y=df["Price"],
        mode="lines",
        name="Price",
        line=dict(color="green", width=2),
        yaxis="y1",
        hovertemplate="Date: %{x}<br>Price: %{y:.2f}<extra></extra>"
    )

    volume_bar = go.Bar(
        x=df["Date"],
        y=df["Vol."],
        name="Volume",
        marker_color="skyblue",
        yaxis="y2",
        opacity=0.6,
        hovertemplate="Date: %{x}<br>Volume: %{y:.0f}<extra></extra>"
    )

    # --- Layout ---
    layout = go.Layout(
        title="Price & Volume Chart",
        xaxis=dict(title="Date", tickformat="%Y-%m-%d"),
        yaxis=dict(title="Price", side="right", showgrid=True),
        yaxis2=dict(title="Volume", overlaying="y", side="left", showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(color="black"),
        hovermode="x unified",
        width=1000,
        height=500,
        legend=dict(x=0.01, y=0.99)
    )
 
    fig = go.Figure(data=[volume_bar, price_line], layout=layout)
    plot(fig, filename="price_volume_chart.html")


def Build_Model():

    #Number of days to predict prices
    FORECAST_DAYS = 7 

    # Load and preprocess data
    df = pd.read_csv("FTSE100_DataProcessed.csv")
    df['Price'] = df['Price'].str.replace(',', '').astype(float)
    df = df.sort_values('Date') 

    # Scale prices
    scaler = MinMaxScaler()
    scaled_prices = scaler.fit_transform(df[['Price']])

    # Create sequences
    def create_sequences(data, window_size=60):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data[i:i+window_size])
            y.append(data[i+window_size])
        return np.array(X), np.array(y)

    window_size = 60
    X, y = create_sequences(scaled_prices, window_size)

    # Build LSTM model with Input() layer
    model = Sequential()
    model.add(Input(shape=(X.shape[1], 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=20, batch_size=16, verbose=0)

    # Predict next N days
    last_sequence = scaled_prices[-window_size:]
    predicted = []
    input_seq = last_sequence.reshape(1, window_size, 1)

    for _ in range(FORECAST_DAYS):
        next_price = model.predict(input_seq, verbose=0)
        predicted.append(next_price[0, 0])
        input_seq = np.append(input_seq[:, 1:, :], [[[next_price[0, 0]]]], axis=1)

    # Inverse transform to get actual price
    predicted_prices = scaler.inverse_transform(np.array(predicted).reshape(-1, 1))

    # Convert dates
    df['Date'] = pd.to_datetime(df['Date'])
    historical_dates = df['Date'].values
    original_prices = scaler.inverse_transform(scaled_prices)

    # Create future dates (as pandas datetime for plotting)
    last_date = df['Date'].iloc[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=FORECAST_DAYS, freq='D')

    # Print predicted results with dates
    print(f"\nPredicted Prices for next {FORECAST_DAYS} days:")
    for i, price in enumerate(predicted_prices, 1):
        print(f"{future_dates[i-1].date()}: {price[0]:.2f}")

    # Plot interactive chart
    fig = go.Figure()

    # Historical data
    fig.add_trace(go.Scatter(
        x=historical_dates,
        y=original_prices.flatten(),
        mode='lines',
        name='Historical Prices',
        line=dict(color='gray')
    ))

    # Predicted data
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=predicted_prices.flatten(),
        mode='lines+markers',
        name='Predicted Prices',
        line=dict(color='blue')
    ))

    # Layout
    fig.update_layout(
        title=f"FTSE 100 - Historical & {FORECAST_DAYS}-Day Forecast",
        xaxis_title="Date",
        yaxis_title="FTSE 100 Price",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(size=14),
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True
    )

    fig.show()

def main():
    import sys
    print(sys.version)
    print('Starting Programme')
    print('Starting processing data')
    ForwardFillData()
    print('Finished Processing Data and outputted the data to FTSE100_DataProcessed')
    print('Start Generating initial data visualisation')
    VisualiseData()
    print('Finished Initial  visuals')
    Build_Model()
    print('Finished building model')




if __name__ == "__main__":
    main()
