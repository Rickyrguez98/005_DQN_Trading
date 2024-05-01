import ta
import pandas as pd

def clean_ds(df):
    df = df.copy()
    for i in range(1, 21):
        df[f'X_t-{i}'] = df['Close'].shift(i)

    # Shift Close Column up by 5 rows
    df['Pt_1'] = df['Close'].shift(-1)
    df['Pt_2'] = df['Close'].shift(-2)
    df['Pt_3'] = df['Close'].shift(-3)
    df['Pt_4'] = df['Close'].shift(-4)
    df['Pt_5'] = df['Close'].shift(-5)
    df['Pt_6'] = df['Close'].shift(-6)
    df['Pt_7'] = df['Close'].shift(-7)
    df['Pt_8'] = df['Close'].shift(-8)
    df['Pt_9'] = df['Close'].shift(-9)
    df['Pt_10'] = df['Close'].shift(-10)
    df['Pt_11'] = df['Close'].shift(-11)
    df['Pt_12'] = df['Close'].shift(-12)
    df['Pt_13'] = df['Close'].shift(-13)
    df['Pt_14'] = df['Close'].shift(-14)
    df['Pt_15'] = df['Close'].shift(-15)
    df['Pt_16'] = df['Close'].shift(-16)
    df['Pt_17'] = df['Close'].shift(-17)
    df['Pt_18'] = df['Close'].shift(-18)
    df['Pt_19'] = df['Close'].shift(-19)
    df['Pt_20'] = df['Close'].shift(-20)

    # Agregamos RSI
    rsi_data = ta.momentum.RSIIndicator(close=df['Close'], window=28)
    df['RSI'] = rsi_data.rsi()

    # La Y
    df['Y_BUY'] = df['Close'] < df['Pt_5']
    df['Y_SELL'] = df['Close'] > df['Pt_5']

    # df['Y_BUY'] = df['Y_BUY'].astype(int)
    # df['Y_SELL'] = df['Y_SELL'].astype(int)

    return df