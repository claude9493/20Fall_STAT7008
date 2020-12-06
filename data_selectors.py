import pandas as pd
import numpy as np

data_tweet = {
    'us corn': pd.read_csv("data/tweet/corn_clean_tweet.csv", index_col="date"),
    'us soybeans':pd.read_csv("data/tweet/soybeans_clean_tweet.csv", index_col="date"),
    'us wheat': pd.read_csv("data/tweet/wheat_clean_tweet.csv", index_col="date")
}
df_price = pd.read_csv("data/df_price.csv", index_col="Date")
df = pd.read_csv("data/investing_news_clean.csv")
df.date = pd.to_datetime(df.date)

def data_selector_news(grain, lag):
    """
    For studying the delay predicting affect of news, for the grain commodity price.
    Parameters:
        grain : type of wanted grain, "all" for all grains, grain should be in COMMODITY_LIST
        lag   : delay of wanted price data 
    Return:
        Dataframe includes date of news, news content, price data after lag days, price data, trend of price data.
    """
    import datetime
    grain = grain.lower()
    
    if grain == "all":
        df_price_lag = df_price.copy()
        df_news = df.copy()
    else:
        df_price_lag = df_price.copy()[[f"{grain}_close"]]
        df_price_lag[f"{grain}_close_trend"] = (df_price_lag[[f"{grain}_close"]].diff().apply(lambda x: np.sign(x)))
        df_price_lag[f"{grain}_close_trend"].replace(0, -1.0, inplace=True)
        df_news = df.copy()
        df_news = df_news[df_news.type == grain.lower()]
        
    df_price_lag[f"date_lag_{lag}"] = (pd.to_datetime(df_price_lag.index).to_series()
                                       .apply(lambda x: x - datetime.timedelta(days = lag)))
    
    data = pd.merge(df_news[["date","type", "words"]], df_price_lag, left_on="date", right_on=f"date_lag_{lag}", 
                    how="inner", suffixes=["_text", "_price"])
    data = data.drop(f"date_lag_{lag}", axis=1)
    print(f"{len(data)} rows of data are selected.")
    return data
    
def data_selector_tweet(grain, start, end, plot=False):
    """
    Select tweets in given date range and corresponding commodity price
    """
    grain = grain.lower()
    assert grain in data_tweet.keys(), f"{grain} is not in our database."
    df_price_copy = df_price.copy().loc[:, [f"{grain}_close"]]
    if plot:
        df_price_copy.loc[start:end, [f"{grain}_close"]].plot(figsize=(10,5))
        
    df_price_copy[f"{grain}_trend_next"] = (df_price_copy[[f"{grain}_close"]].diff()
                                             .apply(lambda x: np.sign(x)).shift(-1))
    df_price_copy.replace(0, -1.0, inplace=True)
#     df_price_coly[f"date_lag_{lag}"] = (pd.to_datetime(df_price_coly.index).to_series()
#                                        .apply(lambda x: x - datetime.timedelta(days = lag)))
    return pd.merge(data_tweet[grain].drop(["tweet"], axis=1), 
                    df_price_copy.loc[start:end, [f"{grain}_close", f"{grain}_trend_next"]], 
                    left_index=True, right_index=True)