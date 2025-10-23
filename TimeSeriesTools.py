import warnings
import sys
import os
from math import sqrt
from itertools import product
from contextlib import contextmanager
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import shapiro, normaltest, kruskal
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.neural_network import MLPRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from prophet.plot import plot_cross_validation_metric
from arch import arch_model
from arch.univariate.base import ConvergenceWarning

warnings.filterwarnings("ignore")

class TimeSeriesTools:
    
    def __init__(self):
        self.y = None
        self.vol_model = None
        self.vol_result = None
        self.vol_forecast_df = None



    def prepare_returns(close_prices):
        close_prices = close_prices.dropna()
        log_returns = np.log(close_prices / close_prices.shift(1)).dropna()
    
        plt.figure(figsize=(10,4))
        plt.plot(log_returns, color="black")
        plt.title("Log Getiri Serisi")
        plt.grid(True, linestyle=":")
        plt.show()
    
        print(f"Ortalama getiri: {log_returns.mean():.6f}")
        print(f"Varyans: {log_returns.var():.6f}")
        return log_returns
            
    """Using the ADF test, the find_best_d function first determines the necessary d (difference) value to make the data stationary. 
    All potential (p,d,q) combinations within the specified limits for p and q are evaluated while maintaining this d value constant.
    For every combination, a model is constructed and an Akaike information criterion (AIC) score is determined. 
    The "best model" is chosen as the one that offers the best fit, or the one with the lowest AIC score. 
    The performance of this optimal model is further assessed using time_series_cv (moving window test) if desired (use_cv=True), 
    and an RMSE score is generated. The function shows a summary of the best model and its prediction graph, then returns the model object."""


    
    def find_best_d(self, y, max_d=2, alpha=0.05):
        for d in range(max_d + 1):
            series = np.diff(y, n=d) if d > 0 else y
            p_value = adfuller(series)[1]
            if p_value < alpha:
                return d
        return max_d

    def time_series_cv(self, y, order, n_splits=5, test_size=12):
        errors = []
        for i in range(n_splits):
            train_end = len(y) - (n_splits - i) * test_size
            train, test = y[:train_end], y[train_end:train_end + test_size]
            try:
                model = ARIMA(train, order=order).fit()
                forecast = model.forecast(steps=len(test))
                rmse = sqrt(mean_squared_error(test, forecast))
                errors.append(rmse)
            except:
                continue
        return np.mean(errors) if errors else np.inf

    def auto_arima(
        self,
        y,
        p_range=(0, 5),
        q_range=(0, 5),
        max_d=2,
        criterion="aic",
        use_cv=True,
        n_splits=5,
        test_size=12,
        forecast_steps=30,
        plot=True,
        show_summary=True
    ):

        d = self.find_best_d(y, max_d=max_d)
        print(f"The most suitable difference degree d = {d}")

        best_score, best_order, best_model = np.inf, None, None

       
        for p, q in product(range(p_range[0], p_range[1] + 1),
                            range(q_range[0], q_range[1] + 1)):
            try:
                model = ARIMA(y, order=(p, d, q)).fit()
                score = getattr(model, criterion)
                if score < best_score:
                    best_score, best_order, best_model = score, (p, d, q), model
            except:
                continue

        if best_model is None:
            raise ValueError("No suitable ARIMA model was found. The most suitable difference order is d. .")

        print(f" Best model: ARIMA{best_order}")
        print(f" {criterion.upper()}: {best_score:.4f}")

        cv_rmse = None
        if use_cv:
            cv_rmse = self.time_series_cv(y, best_order, n_splits=n_splits, test_size=test_size)
            print(f" Cross-Validation RMSE: {cv_rmse:.4f}")

        if show_summary:
            print(" Model Sumamary:\n")
            print(best_model.summary())

        forecast = best_model.forecast(steps=forecast_steps)

        if plot:
            plt.figure(figsize=(10, 5))
            plt.plot(y.index, y, label="Real", color="black")
            plt.plot(pd.date_range(start=y.index[-1], periods=forecast_steps+1, freq='D')[1:], 
                     forecast, color="red", linestyle="--", label=f"ARIMA{best_order} Forecast")
            plt.title(f"ARIMA{best_order} Forecast (AIC={best_score:.2f})")
            plt.legend()
            plt.grid(True, linestyle=":")
            plt.show()

        model_info = pd.DataFrame({
            "Order": [best_order],
            "Criterion": [criterion.upper()],
            "Best Score": [best_score],
            "CV RMSE": [cv_rmse]
        })
        print(" Model Information:")
        display(model_info)

        info = {
            "order": best_order,
            "criterion": criterion,
            "best_score": best_score,
            "cv_rmse": cv_rmse
        }

        return forecast, best_model, info





    def plot_trend_decomposition(self, y, model="additive", period=None, show_components=True):
       
            decomposition = seasonal_decompose(y, model=model, period=period)
    
            plt.figure(figsize=(10,5))
            plt.plot(y, label="Real", color="black")
            plt.plot(decomposition.trend, label="Trend", color="red", linewidth=2)
            plt.title("Real Data and Trend Component")
            plt.xlabel("Time")
            plt.ylabel("Value")
            plt.legend()
            plt.grid(True, linestyle=":")
            plt.tight_layout()
            plt.show()
    
            if show_components:
                decomposition.plot()
                plt.suptitle("Time Series Decomposition (Trend + Seasonal + Residual)", y=1.02)
                plt.tight_layout()
                plt.show()




    def plot_time_series(self, y):
        plt.figure(figsize=(10,5))
        plt.plot(y, color='black', linewidth=1.5)
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()

    def test_stationarity(self, y, alpha=0.05):
        adf_p = adfuller(y.dropna())[1]
        try:
            kpss_p = kpss(y.dropna(), nlags="auto")[1]
        except:
            kpss_p = np.nan

        print("ADF p-value:", round(adf_p,4))
        print("KPSS p-value:", round(kpss_p,4))
        if adf_p < alpha and kpss_p > alpha:
            print("The series is  stationary.")
        else:
            print("The series is not stationary. It may be necessary to take the difference or extract the trend..")

    def plot_acf_pacf(self, y, lags=30):
        fig, axes = plt.subplots(1,2, figsize=(12,4))
        plot_acf(y.dropna(), lags=lags, ax=axes[0])
        plot_pacf(y.dropna(), lags=lags, ax=axes[1])
        axes[0].set_title("ACF")
        axes[1].set_title("PACF")
        plt.tight_layout()
        plt.show()


    def fit_arima_model(self, y, order=(1,1,1)):
        model = ARIMA(y, order=order)
        fitted = model.fit()
        print(fitted.summary())
        return fitted

    def forecast_arima(self, y, model, steps=12):
     
        if not isinstance(y.index, pd.DatetimeIndex):
            print("[WARNING] Index is being created in a non-date format...")
            y.index = pd.date_range(start="2000-01-01", periods=len(y), freq="D")

        freq = pd.infer_freq(y.index) or "D"
        future_index = pd.date_range(start=y.index[-1] + pd.Timedelta(1, freq),
                                     periods=steps, freq=freq)

        
        forecast = model.forecast(steps=steps)

        plt.figure(figsize=(10,5))
        plt.plot(y.index, y, label="Real", color="black")
        plt.plot(future_index, forecast, color="blue", linestyle="--", label="Forecast")
        plt.title("ARIMA Forecast")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()

        return forecast

   
    def manual_decompose(self, y, model="additive", period=None):
        decomposition = seasonal_decompose(y, model=model, period=period)
        decomposition.plot()
        plt.suptitle("Trend + Seasonality + Noise", y=1.02)
        plt.tight_layout()
        plt.show()
        return decomposition




    def test_seasonality(self, y, period):
        df = pd.DataFrame({'y': y})
        df['season'] = [i % period for i in range(len(df))]
        groups = [group["y"].values for _, group in df.groupby("season")]
        stat, p = kruskal(*groups)
        print(f"Kruskal–Wallis H-statistics: {stat:.4f}")
        print(f"p-değeri: {p:.4f}")
        if p < 0.05:
            print("seasonality is present.")
        else:
            print("No significant seasonality was observed..")

    def test_normality(self, residuals):
        shapiro_p = shapiro(residuals)[1]
        dagostino_p = normaltest(residuals)[1]
        print(f"Shapiro–Wilk p-value: {shapiro_p:.4f}")
        print(f"D’Agostino p-value: {dagostino_p:.4f}")
        if shapiro_p > 0.05 and dagostino_p > 0.05:
            print("normal residuals")
        else:
            print("non-normal residuals")

    
    def save_to_excel(self, df, filename="forecast_results.xlsx"):
        path = os.path.abspath(filename)
        df.to_excel(path, index=False)
        print(f"💾 Saved {path}")
        return path

    """The backtesting logic in the forecast_nn function is designed to reliably measure model performance. 
    First, the time series is prepared as X (input features) and Y (targets) according to the lags parameter. 
    An initial dataset is defined using initial_train_size (initial training size), 
    and the remainder of the data is split into n_splits parts of test_size (test size). 
    A loop is started to proceed through these parts. If the method is selected as “expanding,” 
    the MLPRegressor model is retrained in each loop using all data from the beginning up to that point. 
    If the method is selected as “rolling,” the model is trained only on the train_size size of rolling window data this time. 
    This model, trained in each loop, predicts the next test_size number of steps. 
    These predictions, actual values, corresponding dates (dates_all), and the RMSE error for that iteration are recorded. 
    When the iteration is complete, the average of all RMSE errors (avg_rmse) is calculated, 
    and all collected predictions (preds_all) are combined into a single graph against the actual values (actual_all), using the date axis (dates_all)."""


    def forecast_nn(self, y, lags=5, steps=12, hidden_layer_sizes=(64, 32),
                          activation="relu", solver="adam", random_state=42,
                          backtest=True, method="expanding", test_size=12, train_size=None,
                          max_iter=500): 
            
            from sklearn.preprocessing import MinMaxScaler
            from sklearn.neural_network import MLPRegressor
            from sklearn.metrics import mean_squared_error
            import matplotlib.pyplot as plt
        
            if not isinstance(y.index, pd.DatetimeIndex):
                y.index = pd.date_range(start="2000-01-01", periods=len(y), freq="D")
        
            scaler = MinMaxScaler()
            y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))
            X, Y = [], []
            for i in range(len(y_scaled) - lags):
                X.append(y_scaled[i:i+lags].flatten())
                Y.append(y_scaled[i+lags])
            X, Y = np.array(X), np.array(Y)
            
            
            model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                                 activation=activation, solver=solver,
                                 random_state=random_state, 
                                 max_iter=max_iter) 
       
            if backtest:
                errors, preds_all, actual_all = [], [], []
                dates_all = [] 
                
                total_points = len(X) 
                
                if train_size is None or train_size <= 0:
                    initial_train_size = test_size 
                    if method == "rolling":
                        print(f"The ‘rolling’ method requires ‘train_size’.. "
                              f"Temporarily train_size={total_points // 2} is being used.")
                        initial_train_size = total_points // 2 
                else:
                    initial_train_size = train_size
    
                if initial_train_size >= total_points - test_size:
                     print(f" train_size ({initial_train_size}) covers most of the data set. "
                           "Backtest için yeterli veri kalmayabilir.")
                     n_splits = 0
                else:
                     n_splits = (total_points - initial_train_size) // test_size
    
                if n_splits <= 0:
                    print(f"There is insufficient data for backtesting. . "
                          f"Veri: {total_points}, initial_train_size: {initial_train_size}, test_size: {test_size}")
                
                
                for i in range(n_splits):
                    test_start_idx = initial_train_size + (i * test_size)
                    test_end_idx = test_start_idx + test_size
                    
                    if method == "expanding":
                        train_start_idx = 0
                        train_end_idx = test_start_idx
                    elif method == "rolling":
                        train_end_idx = test_start_idx
                        train_start_idx = train_end_idx - initial_train_size
                    else:
                        raise ValueError("Method must be 'expanding' or 'rolling'.")
    
                    if test_end_idx > total_points:
                        test_end_idx = total_points
                        if test_start_idx >= test_end_idx: continue
    
                    X_train, Y_train = X[train_start_idx:train_end_idx], Y[train_start_idx:train_end_idx]
                    X_test, Y_test = X[test_start_idx:test_end_idx], Y[test_start_idx:test_end_idx]
    
                    test_dates = y.index[test_start_idx + lags : test_end_idx + lags]
    
                    if len(X_test) == 0 or len(X_train) == 0:
                        print(f"Split {i} skipped . Train: {len(X_train)}, Test: {len(X_test)}")
                        continue
                        
                    model.fit(X_train, Y_train.ravel())
                    preds = model.predict(X_test)
                    
                    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
                    Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1)).flatten()
                    preds_all.extend(preds_inv)
                    actual_all.extend(Y_test_inv)
                    rmse = sqrt(mean_squared_error(Y_test_inv, preds_inv))
                    errors.append(rmse)
                    dates_all.extend(test_dates) 
                
                avg_rmse = np.mean(errors)
                print(f"[Backtest - {method}] avarage RMSE: {avg_rmse:.4f}")
                
                plt.figure(figsize=(10, 5))
                plt.plot(dates_all, actual_all, label="Real", color="black")
                plt.plot(dates_all, preds_all, label="Forecast", color="green", linestyle="--")
                plt.title(f"MLP Backtest ({method}) - RMSE: {avg_rmse:.2f}")
                plt.legend(); plt.grid(True, linestyle=":")
                plt.show()
    
         
            if len(X) == 0:
                 print("The model could not be trained for future prediction.")
                 return None, None
                 
            model.fit(X, Y.ravel()) 
            
            input_seq = y_scaled[-lags:].flatten()
            preds = []
            for _ in range(steps):
                pred = model.predict(input_seq.reshape(1, -1))
                preds.append(pred[0])
                input_seq = np.append(input_seq[1:], pred)
            preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
            
            freq = pd.infer_freq(y.index) or "D"
            future_index = pd.date_range(start=y.index[-1], periods=steps + 1, freq=freq)[1:]
            
            forecast_df = pd.DataFrame({"Date": future_index, "Forecast": preds_inv})
            path = self.save_to_excel(forecast_df, "forecast_nn_results.xlsx")
        
            plt.figure(figsize=(10, 5))
            plt.plot(y.index, y, label="Real", color="black")
            plt.plot(future_index, preds_inv, "--", color="green", label="NN Forecast")
            plt.axvline(y.index[-1], color="gray", linestyle=":")
            plt.title(f"MLP Forecast({method})")
            plt.legend(); plt.grid(True, linestyle=":")
            plt.show()
            
            return forecast_df, path


    """The backtesting structure in the forecast_lstm function is similar to the forecast_nn (MLP) approach, 
    but it is organized to meet the LSTM's 3D input requirement (sample, time step, feature). 
    After the data is prepared as X and Y, an initial data set of initial_train_size is determined, 
    and the remaining data is divided into n_splits parts of test_size. 
    The critical point in this process is that in each iteration (for i in range(n_splits)), 
    a completely new LSTM model is created from scratch and trained (fit) using the build_lstm() function. 
    This approach prevents data leakage from the previous step. 
    The model is trained according to either an expanding window (with data growing from the beginning) 
    or a rolling window (with fixed-size sliding data), depending on the selected method. The trained model predicts the next test_size step. 
    The predictions obtained throughout all iterations, the actual values, the dates of these values,
    and the RMSE scores are collected in separate lists. Finally, the average RMSE of all tests (avg_rmse) is reported, 
    and all collected dates (dates_all) are used as the X-axis to display all past predictions of the model (preds_all) against actual values (actual_all) in a single graph."""

    
    def forecast_lstm(self, y, lags=10, steps=12, epochs=50, batch_size=8,
                      lstm_units=64, activation="tanh", recurrent_activation="sigmoid",
                      backtest=True, method="expanding", test_size=12, train_size=None):
        
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.metrics import mean_squared_error
        import matplotlib.pyplot as plt
        
        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.date_range(start="2000-01-01", periods=len(y), freq="D")
    
        scaler = MinMaxScaler()
        y_scaled = scaler.fit_transform(np.array(y).reshape(-1, 1))
        X, Y = [], []
        for i in range(len(y_scaled) - lags):
            X.append(y_scaled[i:i+lags]) 
            Y.append(y_scaled[i+lags])
        X, Y = np.array(X), np.array(Y)
    
        def build_lstm():
            model = Sequential([
                LSTM(lstm_units, activation=activation,
                     recurrent_activation=recurrent_activation, input_shape=(lags, 1)),
                Dense(1)
            ])
            model.compile(optimizer="adam", loss="mse")
            return model
    
        
        if backtest:
            preds_all, actual_all, rmses = [], [], []
            dates_all = [] 
            
            total_points = len(X) 
            
            if train_size is None or train_size <= 0:
                initial_train_size = test_size 
                if method == "rolling":
                    print(f"The ‘rolling’ method requires ‘train_size. "
                          f"Temporarily train_size={total_points // 2} is being used.")
                    initial_train_size = total_points // 2 
            else:
                initial_train_size = train_size

            if initial_train_size >= total_points - test_size:
                 print(f"train_size ({initial_train_size}) covers most of the data set. "
                       "Backtest için yeterli veri kalmayabilir.")
                 n_splits = 0
            else:
                 n_splits = (total_points - initial_train_size) // test_size

            if n_splits <= 0:
                print(f"Insufficient data for backtesting "
                      f"Data: {total_points}, initial_train_size: {initial_train_size}, test_size: {test_size}")
            
            
            for i in range(n_splits):
                test_start_idx = initial_train_size + (i * test_size)
                test_end_idx = test_start_idx + test_size
                
                if method == "expanding":
                    train_start_idx = 0
                    train_end_idx = test_start_idx
                elif method == "rolling":
                    train_end_idx = test_start_idx
                    train_start_idx = train_end_idx - initial_train_size
                else:
                    raise ValueError("Method must be 'expanding' or 'rolling'.")

                if test_end_idx > total_points:
                    test_end_idx = total_points
                    if test_start_idx >= test_end_idx: continue

                X_train, Y_train = X[train_start_idx:train_end_idx], Y[train_start_idx:train_end_idx]
                X_test, Y_test = X[test_start_idx:test_end_idx], Y[test_start_idx:test_end_idx]

               
                test_dates = y.index[test_start_idx + lags : test_end_idx + lags]

                if len(X_test) == 0 or len(X_train) == 0:
                    print(f"Split {i} atlandı (boş set). Train: {len(X_train)}, Test: {len(X_test)}")
                    continue
                    
                model = build_lstm()
                model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=0)
                
                preds = model.predict(X_test, verbose=0)
                preds_inv = scaler.inverse_transform(preds)
                Y_test_inv = scaler.inverse_transform(Y_test.reshape(-1, 1))
                
                preds_all.extend(preds_inv.flatten())
                actual_all.extend(Y_test_inv.flatten())
                rmses.append(sqrt(mean_squared_error(Y_test_inv, preds_inv)))
                dates_all.extend(test_dates) # <-- X-EKSENİ DÜZELTMESİ (3/4)
    
            avg_rmse = np.mean(rmses) 
            print(f"[Backtest - {method}] avarage RMSE: {avg_rmse:.4f}")
            
            plt.figure(figsize=(10, 5))
            plt.plot(dates_all, actual_all, label="Real", color="black")
            plt.plot(dates_all, preds_all, label="Forecast", color="blue", linestyle="--")
            plt.title(f"LSTM Backtest ({method}) - RMSE: {avg_rmse:.2f}")
            plt.legend(); plt.grid(True, linestyle=":")
            plt.show()

 
        
        if len(X) == 0:
             print("The model could not be trained for future prediction..")
             return None, None
             
        model = build_lstm()
        model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=0)
        
        input_seq = y_scaled[-lags:]
        preds = []
        for _ in range(steps):
            pred = model.predict(input_seq.reshape(1, lags, 1), verbose=0)
            preds.append(pred[0, 0])
            input_seq = np.vstack([input_seq[1:], pred])
        
        preds_inv = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    
        freq = pd.infer_freq(y.index) or "D"
        future_index = pd.date_range(start=y.index[-1], periods=steps + 1, freq=freq)[1:]
        
        forecast_df = pd.DataFrame({"Date": future_index, "Forecast": preds_inv})
        path = self.save_to_excel(forecast_df, "forecast_lstm_results.xlsx")
    
        plt.figure(figsize=(10, 5))
        plt.plot(y.index, y, color="black", label="Real")
        plt.plot(future_index, preds_inv, "--", color="blue", label="LSTM Forecast")
        plt.axvline(y.index[-1], color="gray", linestyle=":")
        plt.title(f"LSTM Forecast + Backtest ({method})")
        plt.legend(); plt.grid(True, linestyle=":")
        plt.show()
        return forecast_df, path


    """The prophet_forecast function performs an automatic forecast using the Facebook Prophet library. 
    It converts the data into ds-y format and trains a Prophet model with annual/weekly seasonality. If the initial and horizon parameters are not provided, 
    they are automatically set to test 10% of the data. To measure the model's performance, 
    the built-in cross_validation and performance_metrics tools are run, and an RMSE graph is displayed. 
    Then, the model is retrained with all the data to make a final forecast for the future (for periods) and the graph is plotted. 
    All results (CV, metrics, prediction) are saved to Excel, and the model objects are returned."""

    
    def prophet_forecast(self, y, periods=30, freq="D", initial=None, horizon=None, period="30 days"):
        from prophet import Prophet
        from prophet.diagnostics import cross_validation, performance_metrics
        from prophet.plot import plot_cross_validation_metric

        if not isinstance(y.index, pd.DatetimeIndex):
            y.index = pd.to_datetime(y.index, errors="coerce")
            y = y.dropna()
        df = pd.DataFrame({"ds": y.index, "y": y.values})

        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        m.fit(df)

        total_days = (df["ds"].max() - df["ds"].min()).days
        if initial is None or horizon is None:
            horizon_days = max(int(total_days * 0.1), 14)
            initial_days = total_days - horizon_days
            initial = f"{initial_days} days"
            horizon = f"{horizon_days} days"
            print(f"Length of data {total_days} Day -> initial={initial}, horizon={horizon}")

        print("Prophet cross-validation start...")
        df_cv = cross_validation(m, initial=initial, period=period, horizon=horizon, parallel="processes")

        df_p = performance_metrics(df_cv)
        print("\nProphet Cross-Validation metrics:")
        print(df_p[["horizon", "rmse", "mae", "mape", "coverage"]].head())

        plot_cross_validation_metric(df_cv, metric="rmse")
        plt.title("Prophet Cross-Validation RMSE")
        plt.grid(True, linestyle=":"); plt.show()

        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast = m.predict(future)
        m.plot(forecast); plt.title("Prophet Forecast (Full Model)"); plt.grid(True, linestyle=":"); plt.show()
        m.plot_components(forecast); plt.show()

        df_cv.to_excel("prophet_cv_results.xlsx", index=False)
        df_p.to_excel("prophet_cv_metrics.xlsx", index=False)
        forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].to_excel("prophet_forecast_results.xlsx", index=False)
        print("Prophet Results saved.")
        return m, forecast, df_cv, df_p


    def get_residuals(self, model):
       
        try:
            residuals = model.resid
            print(f"The residual series was successfully acquired. (Length: {len(residuals)})")
            print(f" Mean error: {residuals.mean():.4f} | Std: {residuals.std():.4f}")
            return residuals
        except Exception as e:
            print(f"The residual series was not successfully acquired: {e}")
            return None


    """The suppress_output function is a utility designed to hide unnecessary console output generated during automatic model search operations. 
    A function such as fit_best_vol_model_debug tries dozens of models to find the best one and can produce numerous warnings 
    or progress outputs during this process, such as ConvergenceWarning. When used within a with block,
    the suppress_output function temporarily “silences” all these warnings (warnings.catch_warnings) and standard output/error messages (sys.stdout/stderr) by redirecting them to os.devnull. 
    This ensures that your console does not become cluttered with unnecessary “noise” while the analysis is running, 
    and only the final results table you want to see is displayed cleanly."""
    
    @contextmanager
    def suppress_output(self):
        with open(os.devnull, "w") as devnull:
            old_stdout, old_stderr = sys.stdout, sys.stderr
            sys.stdout, sys.stderr = devnull, devnull
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=ConvergenceWarning)
                yield
            sys.stdout, sys.stderr = old_stdout, old_stderr


    def prepare_returns(self, y):
        y = y.dropna()
        log_returns = np.log(y / y.shift(1)).dropna()
        plt.figure(figsize=(10, 4))
        plt.plot(log_returns, color="black")
        plt.title("Log Getiri Serisi")
        plt.grid(True, linestyle=":")
        plt.show()
        return log_returns

    def fit_arch(self, y, lags=1):
        model = arch_model(y, vol="ARCH", p=lags)
        result = model.fit(disp="off")
        print(result.summary())

        plt.figure(figsize=(10, 4))
        plt.plot(result.conditional_volatility, color="blue", label="ARCH Volatility")
        plt.legend(); plt.grid(True, linestyle=":"); plt.show()
        return result

    def fit_garch(self, y, p=1, q=1):
        model = arch_model(y, vol="GARCH", p=p, q=q)
        result = model.fit(disp="off")
        print(result.summary())

        plt.figure(figsize=(10, 4))
        plt.plot(result.conditional_volatility, color="green", label="GARCH Volatility")
        plt.legend(); plt.grid(True, linestyle=":"); plt.show()
        return result

    def fit_egarch(self, y, p=1, q=1):
        model = arch_model(y, vol="EGARCH", p=p, q=q)
        result = model.fit(disp="off")
        print(result.summary())

        plt.figure(figsize=(10, 4))
        plt.plot(result.conditional_volatility, color="red", label="EGARCH Volatility")
        plt.legend(); plt.grid(True, linestyle=":"); plt.show()
        return result

    def fit_gjr_garch(self, y, p=1, q=1, o=1):
        model = arch_model(y, vol="GARCH", p=p, o=o, q=q)
        result = model.fit(disp="off")
        print(result.summary())

        plt.figure(figsize=(10, 4))
        plt.plot(result.conditional_volatility, color="purple", label="GJR-GARCH Volatility")
        plt.legend(); plt.grid(True, linestyle=":"); plt.show()
        return result

    """This function forecasts future volatility using a trained GARCH model (result). 
    It calls the result.forecast() method. If the model (such as EGARCH) does not support analytical forecasting, 
    it automatically switches to the method="simulation" (simulation) method. Volatility is calculated by taking the square root of the forecasted variance. 
    Volatility estimates for the future are displayed on the same graph as the volatility of the past 200 periods. 
    The estimates are saved to Excel and returned as a DataFrame."""


    
    def forecast_garch_volatility(self, result, y, steps=30, title="GARCH Volatility Forecast", save_path="volatility_forecast.xlsx"):
    
        try:
            forecasts = result.forecast(horizon=steps)
        except ValueError:
            print("Analytical estimation is not supported; switching to simulation method...")
            forecasts = result.forecast(horizon=steps, method="simulation", simulations=500)

        var_forecast = forecasts.variance.values[-1, :]
        vol_forecast = np.sqrt(var_forecast)

        if not isinstance(y.index, pd.DatetimeIndex):
            print("The time series index is not a DatetimeIndex, it is being converted...")
            y.index = pd.to_datetime(y.index, errors="coerce")

        freq = pd.infer_freq(y.index)
        if freq is None:
            print("[Index frequency could not be detected automatically, ‘B’ (business day) is assumed.")
            freq = "B"

        future_index = pd.date_range(start=y.index[-1], periods=steps + 1, freq=freq)[1:]


        plt.figure(figsize=(10, 5))
        plt.plot(y.index[-200:], result.conditional_volatility[-200:], label="Past Volatility", color="black")
        plt.plot(future_index, vol_forecast, label="Estimated Volatility", color="blue", linestyle="--")
        plt.title(title)
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.show()

        forecast_df = pd.DataFrame({
            "Estimated Volatility": vol_forecast
        }, index=future_index)

        forecast_df.to_excel(save_path)
        print(f" Volatility forecasts for the next {steps} period have been recorded. → {save_path}")
        print(forecast_df.round(6))

        return forecast_df

    def color_status(self,val):
        return f"color: {'green' if val == 'Stationary' else 'red'}"

    def plot_volatility_fit_and_forecast(self,result, y, steps=30, title="Volatility Adjustment and Forecast"):
        cond_vol = result.conditional_volatility
    
        try:
            forecasts = result.forecast(horizon=steps)
            var_forecast = forecasts.variance.values[-1, :]
            vol_forecast = np.sqrt(var_forecast)
        except Exception as e:
            print(f"Forecast could not be calculated: {e}")
            vol_forecast = None
    
        if isinstance(y.index, pd.DatetimeIndex):
            last_date = y.index[-1]
            future_dates = pd.date_range(start=last_date, periods=steps + 1, freq="D")[1:]
        else:
            future_dates = np.arange(len(y), len(y) + steps)
    
        plt.figure(figsize=(12, 6))
        plt.plot(y.index, y, color="black", linewidth=1, alpha=0.7, label="log retuns")
        plt.plot(y.index, cond_vol, color="red", linewidth=1.5, label="Model Volality")
        plt.fill_between(y.index, -cond_vol, cond_vol, color="red", alpha=0.15)
    
        if vol_forecast is not None:
            plt.plot(future_dates, vol_forecast, color="blue", linestyle="--", linewidth=1.8, label=f"{steps} Forecast")
            plt.fill_between(future_dates, -vol_forecast, vol_forecast, color="blue", alpha=0.1)
          
    
        plt.title(title, fontsize=14, fontweight="bold")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True, linestyle=":")
        plt.tight_layout()
        plt.show()



    """This function automatically selects the best volatility model. 
    It tests all combinations within the specified model_types (GARCH, EGARCH, etc.) and p and q ranges.
    Each model is trained silently, and the persistence value (the sum of $\alpha + \beta$) is checked. 
    Among the stable models that fall below the specified threshold, the model with the lowest AIC score is selected as the “best model.
    ” The results of all trials are displayed in a table color-coded according to stability status. 
    The fit/prediction graph of the best model is plotted, and the model object (best_res) is returned."""

    
    def fit_best_vol_model(
        self,
        y,
        p_range=(1, 3),
        q_range=(1, 3),
        model_types=["GARCH", "EGARCH", "GJR-GARCH", "ARCH"],
        dist="normal",
        show_table=True,
        threshold=1.05,  
        forecast_steps=60
    ):
        best_aic = np.inf
        best_info = None
        best_res = None
        results = []
    
        for vol_type in model_types:
            for p in range(p_range[0], p_range[1] + 1):
                for q in range(q_range[0], q_range[1] + 1):
                    try:
                        model = arch_model(y, vol=vol_type, p=p, q=q, dist=dist)
                        with self.suppress_output():
                            res = model.fit(disp="off")
    
                        if hasattr(res, "params"):
                            alpha = res.params.get("alpha[1]", 0)
                            beta = res.params.get("beta[1]", 0)
                            persistence = alpha + beta
                            status = "Stationary" if persistence < threshold else "Non-Stationary"
    
                            results.append((vol_type, p, q, res.aic, persistence, status))
    
                            if persistence < threshold and res.aic < best_aic:
                                best_aic = res.aic
                                best_info = (vol_type, p, q, persistence, res.aic)
                                best_res = res
                    except Exception:
                        continue
    
        if not results:
            print("No model that was run")
            return None, None
    
        df = pd.DataFrame(results, columns=["Model", "p", "q", "AIC", "α+β", "Status"]).sort_values("AIC")
    
        if show_table:
            print("Stationary = green, Non-Stationary = red:")
            display(df.style.applymap(self.color_status, subset=["Status"]))
    
        if best_info is None:
            print("No Stationary model was found..")
        else:
            vol_type, p, q, persistence, aic = best_info
            print(f"\n✅ Best Model: {vol_type}({p},{q}) | AIC={aic:.3f} | α+β={persistence:.3f}")
    
            self.plot_volatility_fit_and_forecast(
                best_res,
                y,
                steps=forecast_steps,
                title=f"{vol_type}({p},{q}) | AIC={aic:.1f} | α+β={persistence:.3f}"
            )
    
        return best_res, best_info