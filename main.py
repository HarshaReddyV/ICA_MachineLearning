import pandas as pd
import plotly.graph_objs as go
from plotly.offline import plot
from plotly.offline import plot
import plotly.graph_objs as go


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



def BackFillData():
    
    #Import and Parse the Date column and define the format of date
    df = pd.read_csv("FTSE100.csv", parse_dates=["Date"], dayfirst=True)

    #Modify Date series 
    df.set_index("Date", inplace=True)
    all_business_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='B')
    df = df.reindex(all_business_days)
    df.index.name = 'Date'

    #Do Backward fill so the missing dates have the price,volume and percentage as same as the last business day 
    df.ffill(inplace=True)
    
    #Convert volume to a float instead of B and M denoting Billion and million respectively
    df["Vol."] = df["Vol."].apply(Convert_Volume).astype(float)
    
    
    #Sort and write output to a csv file
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
    
def main():
    print('Starting Programme')
    print('Starting processing data')
    BackFillData()
    print('Finished Processing Data and outputted the data to FTSE100_DataProcessed')
    print('Start Generating initial data visualisation')
    VisualiseData()
    print('Finished Initial  visuals')





if __name__ == "__main__":
    main()
