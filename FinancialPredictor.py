import os
import json
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class CNN_LSTM(nn.Module):
    def __init__(self, window_size):
        super(CNN_LSTM, self).__init__()
        # CNN layer with 1D convolution
        # input channels = 1, output channels = 64, kernel size = 2
        self.conv1d = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=2)
        self.dropout = nn.Dropout(0.2)
        # LSTM layers
        # Note new sequence length = window_size - kernel_size + 1
        self.lstm = nn.LSTM(input_size=64, hidden_size=50, batch_first=True)
        self.fc = nn.Linear(50, 1)
    
    def forward(self, x):
        # Note as we are reviewing each data point wiht a window 
        # x: (batch, 1, window_size)
        x = self.conv1d(x) 
        x = self.dropout(x)
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]
        out = self.fc(out) 
        return out.squeeze(1)

class FinancialPredictor:
    def __init__(self, store_dir="./store/", models_dir="./models/", window_size=10):
        """
        On initialization, the class loads (or creates) two dictionaries:
          - self.data_dict: {asset_name: file_path} for asset CSV files.
          - self.models_dict: {model_name: {"model_path": ..., "scaler_path": ..., "end_date": ...}}
        These dictionaries are stored in JSON files in the store_dir.
        """
        self.store_dir = store_dir
        self.models_dir = models_dir
        self.window_size = window_size
        os.makedirs(self.store_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        self.data_dict_path = os.path.join(self.store_dir, "data_dict.json")
        self.models_dict_path = os.path.join(self.store_dir, "models_dict.json")
        
        if os.path.exists(self.data_dict_path):
            with open(self.data_dict_path, "r") as f:
                self.data_dict = json.load(f)
        else:
            self.data_dict = {}
        
        if os.path.exists(self.models_dict_path):
            with open(self.models_dict_path, "r") as f:
                self.models_dict = json.load(f)
        else:
            self.models_dict = {}
    
    def _save_dicts(self):
        """
        Save the two dictionaries to JSON files. 
        This must be called every time either dictionary is modified.
        """
        with open(self.data_dict_path, "w") as f:
            json.dump(self.data_dict, f, indent=4)
        with open(self.models_dict_path, "w") as f:
            json.dump(self.models_dict, f, indent=4)
    
    def add_data(self, name, file_path):
        """
        Add or update the file path for a asset.
          name: key (e.g., "SPY")
          file_path: path to the CSV file
        """
        self.data_dict[name] = file_path
        self._save_dicts()
        print(f"Data for asset '{name}' has been updated/added with file: {file_path}")

    def predict_model(self, model_name, assetname, end_date, file_path=None, num_epochs=50, batch_size=16):
        """
        Train and store a model using data from the specified asset up to (and including) end_date.
          model_name: key to store the model metadata in models_dict.
          assetname: key for the asset (used in the data dictionary)
          end_date: training cutoff date (string in "yyyy-mm-dd" format)
          file_path: if provided, update the data dictionary for this asset.
        Returns the in-sample error (MSE on returns).
        """
        # Update or retrieve the file path for the asset.
        if file_path is not None:
            self.data_dict[assetname] = file_path
        else:
            if assetname not in self.data_dict:
                print("Error: File path not specified and not found in the data dictionary.")
                return
            file_path = self.data_dict[assetname]
        self._save_dicts()

        # Load, sort, and filter the CSV data.
        df = pd.read_csv(file_path, parse_dates=['Date'])
        df.sort_values("Date", inplace=True)
        # Filter data on or prior to end_date.
        df_train = df[df["Date"] <= end_date].copy()
        if df_train.empty:
            print("No training data available on or prior to the specified end_date.")
            return
        closes = df_train["Close"].values.reshape(-1, 1)
        log_prices = np.log(closes)
        log_returns = np.diff(log_prices, axis=0)  # (n-1, 1)
        if len(log_returns) < self.window_size:
            print("Not enough data for the specified window size.")
            return

        # We train with log returns to prevent parameter explosion
        scaler = StandardScaler()
        scaled_returns = scaler.fit_transform(log_returns)

        # Use sliding windows
        X_list, y_list = [], []
        for i in range(self.window_size, len(scaled_returns)):
            X_list.append(scaled_returns[i - self.window_size:i, 0])
            y_list.append(scaled_returns[i, 0])
        X_np = np.array(X_list)  # (num_samples, window_size)
        y_np = np.array(y_list)  # (num_samples,)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)  # (samples, 1, window_size)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Init the model, loss function, and optimizer
        model = CNN_LSTM(self.window_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")

        # In-sample error
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            insample_error = criterion(preds, y_tensor).item()
        print(f"In-sample MSE on returns: {insample_error:.6f}")

        # Save the model and scaler
        model_path = os.path.join(self.models_dir, f"{model_name}.pt")
        scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

        # Update the models dictionary with metadata
        self.models_dict[assetname] = {"model_path": model_path,
                                          "scaler_path": scaler_path,
                                          "end_date": str(end_date)}
        self._save_dicts()

        print(f"Model '{assetname}' saved to {model_path}.")
        return insample_error

    def test_model(self, assetname, start_date, model_name = None):
        """
        Test the model specified by model_name (which should exist in the models dictionary)
        on test data for the given asset (from the data dictionary) starting from start_date.
        Produces two side-by-side plots:
          1) Time vs. log returns (true vs. predicted)
          2) Time vs. closing prices (true vs. predicted)
        The y-axis of the plots is the MSE errors
        """
        # Check that the asset exists.
        if assetname not in self.data_dict:
            print("asset data not found in the data dictionary.")
            return
        file_path = self.data_dict[assetname]

        if model_name == None:
            if assetname not in self.models_dict:
                print("Model not found in the models dictionary.")
                return
            model_info = self.models_dict[assetname]
            model_path = model_info["model_path"]
            print(model_path)
            scaler_path = model_info["scaler_path"]
        else:
            if model_name not in self.models_dict:
                print("Model not found in the models dictionary.")
                return
            model_info = self.models_dict[model_name]
            model_path = model_info["model_path"]
            print(model_path)
            scaler_path = model_info["scaler_path"]

        # Load the model & scalar
        model = CNN_LSTM(self.window_size)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        # Load test data and filter by start_date.
        df_test = pd.read_csv(file_path, parse_dates=['Date'])
        df_test.sort_values("Date", inplace=True)
        df_test = df_test[df_test["Date"] >= start_date].copy()
        if df_test.empty:
            print("No test data available from the specified start_date.")
            return

        # Compute log returns on the test data
        closes = df_test["Close"].values.reshape(-1, 1)
        log_prices = np.log(closes)
        log_returns = np.diff(log_prices, axis=0)  # shape: (n-1, 1)
        if len(log_returns) < self.window_size:
            print("Not enough test data for the specified window size.")
            return

        # Scale test log returns using the loaded scaler
        scaled_returns = scaler.transform(log_returns)

        # Record true target sliding windows
        X_list, y_list = [], []
        true_returns, true_closings, time_stamps = [], [], []
        test_dates = df_test["Date"].values  # use test file timestamps
        for i in range(self.window_size, len(scaled_returns)):
            X_list.append(scaled_returns[i - self.window_size:i, 0])
            y_list.append(scaled_returns[i, 0])
            # True log return and closing price:
            true_returns.append(log_returns[i, 0])
            # True closing: use the closing price corresponding to index i+1 (since diff shifts index by one)
            if i + 1 < len(closes):
                true_closings.append(closes[i + 1, 0])
            if i + 1 < len(test_dates):
                time_stamps.append(pd.to_datetime(test_dates[i + 1]))
        X_test = np.array(X_list)
        y_test = np.array(y_list)

        # Convert to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
        with torch.no_grad():
            pred_scaled_tensor = model(X_test_tensor)
        pred_scaled = pred_scaled_tensor.numpy().flatten()
        # Get predicted log returns
        pred_log_returns = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

        # Compute predicted closing prices
        predicted_closings = []
        for idx, i in enumerate(range(self.window_size, len(scaled_returns))):
            pred_close = closes[i][0] * np.exp(pred_log_returns[idx])
            predicted_closings.append(pred_close)
        predicted_closings = np.array(predicted_closings)
        true_closings = np.array(true_closings)
        true_returns = np.array(true_returns)

        # Compute mse's
        returns_mse = np.mean((pred_log_returns - np.array(true_returns)) ** 2)
        closings_mse = np.mean((predicted_closings - true_closings) ** 2)

        # Create side-by-side plots
        fig, axs = plt.subplots(1, 2, figsize=(16, 6))
        axs[0].plot(time_stamps, true_returns, label="True Returns", linestyle='-', color='blue')
        axs[0].plot(time_stamps, pred_log_returns, label="Predicted Returns", linestyle='-', color='red')
        axs[0].set_xlabel("Time")
        axs[0].set_ylabel("Log Return")
        axs[0].set_title(f"Log Returns\nMSE: {returns_mse:.6f}")
        axs[0].legend()
        
        axs[1].plot(time_stamps, true_closings, label="True Closings", linestyle='-', color='blue')
        axs[1].plot(time_stamps, predicted_closings, label="Predicted Closings", linestyle='-', color='red')
        axs[1].set_xlabel("Time")
        axs[1].set_ylabel("Closing Price")
        axs[1].set_title(f"Closing Prices\nMSE: {closings_mse:.6f}")
        axs[1].legend()
        
        plt.tight_layout()
        os.makedirs('./results/CNNLSTM', exist_ok=True)
        if model_name:
            plt.savefig(f'./results/CNNLSTM/{assetname}_{model_name}_results.png')
        else:
            plt.savefig(f'./results/CNNLSTM/{assetname}_results.png')


        print(f"Test completed. Returns MSE: {returns_mse:.6f}, Closings MSE: {closings_mse:.6f}")
        return returns_mse, closings_mse
    
    def train_with_all(self, end_date, num_epochs=50, batch_size=16):
        """
        Train a model using data from all assets (from self.data_dict) on or prior to end_date.
        Data from each asset is processed separately to create sliding windows from log returns.
        All windows are then combined to train a single model.
        The trained model and its scaler are stored in self.models_dict under the key "KS_Model".
        Returns the in-sample MSE error (on returns).
        """
        # collect all log returns (per asset) for scaling
        all_returns_list = []
        for asset, file_path in self.data_dict.items():
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.sort_values("Date", inplace=True)
            df_train = df[df["Date"] <= end_date].copy()
            if len(df_train) < 2:
                print(f"Not enough data for asset '{asset}' before {end_date}. Skipping.")
                continue
            closes = df_train["Close"].values.reshape(-1, 1)
            log_prices = np.log(closes)
            log_returns = np.diff(log_prices, axis=0)  # (n-1, 1)
            all_returns_list.append(log_returns)
        
        if not all_returns_list:
            print("No sufficient training data available from any assets.")
            return
        
        combined_returns = np.vstack(all_returns_list)
        scaler = StandardScaler()
        scaler.fit(combined_returns)
        
        # create sliding windows for each asset and aggregate
        X_total = []
        y_total = []
        for asset, file_path in self.data_dict.items():
            df = pd.read_csv(file_path, parse_dates=['Date'])
            df.sort_values("Date", inplace=True)
            df_train = df[df["Date"] <= end_date].copy()
            if len(df_train) < self.window_size + 1:
                print(f"Not enough data for asset '{asset}' for the given window size. Skipping.")
                continue
            closes = df_train["Close"].values.reshape(-1, 1)
            log_prices = np.log(closes)
            log_returns = np.diff(log_prices, axis=0)
            scaled_returns = scaler.transform(log_returns)
            for i in range(self.window_size, len(scaled_returns)):
                X_total.append(scaled_returns[i - self.window_size:i, 0])
                y_total.append(scaled_returns[i, 0])
        
        if len(X_total) == 0:
            print("No training windows were created. Check your data and window_size.")
            return
        
        X_np = np.array(X_total)  # (total_samples, window_size)
        y_np = np.array(y_total)  # (total_samples,)
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X_np, dtype=torch.float32).unsqueeze(1)  # (samples, 1, window_size)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Initialize model, loss, and optimizer
        model = CNN_LSTM(self.window_size)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # Training loop
        model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                output = model(batch_X)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
            epoch_loss /= len(dataset)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.6f}")
        
        # In-sample error
        model.eval()
        with torch.no_grad():
            preds = model(X_tensor)
            insample_error = criterion(preds, y_tensor).item()
        print(f"In-sample MSE on returns (all assets): {insample_error:.6f}")
        
        # Save as "KS_Model" as in Kitchen Sink - different but concept is similar
        model_path = os.path.join(self.models_dir, "KS_Model.pt")
        scaler_path = os.path.join(self.models_dir, "KS_Model_scaler.pkl")
        torch.save(model.state_dict(), model_path)
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)
        
        self.models_dict["KS_Model"] = {"model_path": model_path,
                                        "scaler_path": scaler_path,
                                        "end_date": str(end_date)}
        self._save_dicts()
        print("Unified model 'KS_Model' has been saved.")
        return insample_error

if __name__ == "__main__":
    predictor = FinancialPredictor(window_size=7)
    mse_list = []
    
    def construct(name, file_path):
        predictor.add_data(name, file_path)
        print("Training model ...")
        insample_error = predictor.predict_model(f"{name}_model", name, "2020-06-01")
        print(f"In-sample error (MSE on returns): {insample_error:.6f}")
        print("Testing model ...")
        mse_list.append(predictor.test_model(name, "2000-01-01"))

    all_tests = [('SPY','./data/SPY_all.csv'),('MSFT','./data/MSFT_all.csv'),('AMZN','./data/AMZN_all.csv'),('META','./data/META_all.csv'),('AAPL','./data/AAPL_all.csv'),('GOOG','./data/GOOG_all.csv'),('NVDA','./data/NVDA_all.csv'),('TSLA','./data/TSLA_all.csv'),("NVDA_daily","./data/NVDA_daily.csv")]
    for item in all_tests:
        construct(item[0], item[1])
    for i in range(len(mse_list)):
        print(f'{all_tests[i][0]}: (Returns MSE = {float(mse_list[i][0])}, Closings MSE = {float(mse_list[i][1])})')

    mse_cross_val_return = [[" " for _ in range(len(all_tests)+1)] for _ in range(len(all_tests)+2)]
    mse_cross_val_closing = [[" " for _ in range(len(all_tests)+1)] for _ in range(len(all_tests)+2)]

    for i in range(len(all_tests)):
        mse_cross_val_return[i+1][0] = all_tests[i][0]
        mse_cross_val_return[0][i+1] = all_tests[i][0]
        mse_cross_val_closing[i+1][0] = all_tests[i][0]
        mse_cross_val_closing[0][i+1] = all_tests[i][0]

    mse_cross_val_return[-1][0] = "Kitchen Sink"
    mse_cross_val_closing[-1][0] = "Kitchen Sink"


    print("Cross Validation Check...")
    for i in range(len(all_tests)):
        for j in range(len(all_tests)):
            dummy = predictor.test_model(all_tests[i][0], "2000-01-01", model_name=all_tests[j][0])
            mse_cross_val_return[j+1][i+1] = float(dummy[0])
            mse_cross_val_closing[j+1][i+1] = float(dummy[1])

    predictor.train_with_all("2020-06-01", num_epochs=50, batch_size=16)

    for i in range(len(all_tests)):
        dummy = predictor.test_model(all_tests[i][0], "2000-01-01", "KS_Model")
        mse_cross_val_return[-1][i+1] = float(dummy[0])
        mse_cross_val_closing[-1][i+1] = float(dummy[1])


    mse_cross_val_return = pd.DataFrame(mse_cross_val_return)
    mse_cross_val_return.to_csv('./results/CNNLSTM/cross_val_mse_return.csv', index=False)  
    print("Cross Validation return MSE's saved to ./results/CNNLSTM/cross_val_mse_return.csv")
    mse_cross_val_closing = pd.DataFrame(mse_cross_val_closing)
    mse_cross_val_closing.to_csv('./results/CNNLSTM/cross_val_mse_closing.csv', index=False)  
    print("Cross Validation return MSE's saved to ./results/CNNLSTM/cross_val_mse_closing.csv")