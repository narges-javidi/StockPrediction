from pytse_client import Ticker

def get_tse_data(symbol="فولاد"):
    t = Ticker(symbol)
    return t.history
