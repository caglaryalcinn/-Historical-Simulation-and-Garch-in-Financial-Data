import yfinance as yf

class Dataf:

    def get_close_prices(ticker, start, end):
        data = yf.download(ticker, start=start, end=end)
        return data['Close']
