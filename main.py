import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from itertools import product
import os

matplotlib.use('Agg')  # Use non-interactive backend for plotting


def load_data(file_path):
    """
    Load BTC data from a CSV file and calculate EMAs.
    Args:
    - file_path (str): Path to the CSV file.
    Returns:
    - pd.DataFrame: Data with calculated EMAs.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['ShortEMA'] = data['close'].ewm(span=21, adjust=False).mean()
    data['LongEMA'] = data['close'].ewm(span=200, adjust=False).mean()
    return data


def simulate_trading(data, initial_capital, stop_loss_pct, take_profit_pct, commission_pct):
    """
    Simulate the EMA Crossover trading strategy.
    Args:
    - data (pd.DataFrame): BTC data with EMAs.
    - initial_capital (float): Starting capital for trading.
    - stop_loss_pct (float): Stop loss percentage.
    - take_profit_pct (float): Take profit percentage.
    - commission_pct (float): Commission percentage per trade.
    Returns:
    - float: Total PnL.
    - float: PnL percentage.
    - list: Trade details including entry/exit prices.
    - list: Buy signals for plotting.
    - list: Sell signals for plotting.
    """
    capital = initial_capital
    position = 0
    entry_price = 0
    buy_signals = []
    sell_signals = []
    trades = []

    for i in range(1, len(data)):
        if position == 0:  # No current position
            if data['ShortEMA'].iloc[i] > data['LongEMA'].iloc[i] and \
                    data['ShortEMA'].iloc[i - 1] <= data['LongEMA'].iloc[i - 1]:
                position = 1
                entry_price = data['close'].iloc[i]
                buy_signals.append((data.index[i], entry_price))
                trades.append({
                    'type': 'long',
                    'entry_date': data.index[i],
                    'entry_price': entry_price,
                    'exit_date': None,
                    'exit_price': None,
                    'pnl': None
                })

        elif position == 1:  # In a long position
            stop_loss_price = entry_price * (1 - stop_loss_pct)
            take_profit_price = entry_price * (1 + take_profit_pct)

            if data['close'].iloc[i] <= stop_loss_price or data['close'].iloc[i] >= take_profit_price:
                position = 0
                exit_price = data['close'].iloc[i]
                sell_signals.append((data.index[i], exit_price))
                trade_profit = (exit_price - entry_price) * (1 if exit_price >= take_profit_price else -1)
                trade_net_profit = trade_profit - (entry_price * commission_pct) - (exit_price * commission_pct)
                capital += trade_net_profit
                trades[-1].update({
                    'exit_date': data.index[i],
                    'exit_price': exit_price,
                    'pnl': trade_net_profit
                })

    total_pnl = capital - initial_capital
    pnl_percentage = (total_pnl / initial_capital) * 100
    return total_pnl, pnl_percentage, trades, buy_signals, sell_signals


def vary_parameters(data, initial_capital, commission_pct, stop_loss_range, take_profit_range, asset_name, strategy):
    """
    Vary stop loss and take profit parameters and save results for each combination.
    Args:
    - data (pd.DataFrame): BTC data with EMAs.
    - initial_capital (float): Starting capital for trading.
    - commission_pct (float): Commission percentage per trade.
    - stop_loss_range (list): List of stop loss percentages to test.
    - take_profit_range (list): List of take profit percentages to test.
    - asset_name (str): Name of the asset for use in filenames.
    - strategy (str): Strategy name for use in filenames.
    """
    for stop_loss_pct, take_profit_pct in product(stop_loss_range, take_profit_range):
        total_pnl, pnl_percentage, trades, buy_signals, sell_signals = simulate_trading(
            data, initial_capital, stop_loss_pct, take_profit_pct, commission_pct
        )

        # Define file names based on current SL, TP, asset name, and strategy
        file_suffix = f"{asset_name}_{strategy}_SL_{int(stop_loss_pct * 100)}_TP_{int(take_profit_pct * 100)}"
        csv_filename = f"output/trades_{file_suffix}.csv"
        image_filename = f"output/plot_{file_suffix}.png"

        # Save trades to CSV
        trades_df = pd.DataFrame(trades)
        trades_df.to_csv(csv_filename, index=False)

        # Plot signals and save the plot
        plot_signals(data, buy_signals, sell_signals, output_file=image_filename)

        print(
            f"SL: {stop_loss_pct * 100:.1f}%, TP: {take_profit_pct * 100:.1f}% -> Net PnL: Rs {total_pnl:.2f}, Saved to {csv_filename} and {image_filename}")


def plot_signals(data, buy_signals, sell_signals, output_file):
    """
    Plot BTC prices, EMAs, and buy/sell signals.
    Args:
    - data (pd.DataFrame): BTC data with EMAs.
    - buy_signals (list): Buy signals.
    - sell_signals (list): Sell signals.
    - output_file (str): Filename to save the plot.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['close'], label='Close Price', alpha=0.5)
    plt.plot(data['ShortEMA'], label='21 EMA', linestyle='--', alpha=0.7)
    plt.plot(data['LongEMA'], label='200 EMA', linestyle='--', alpha=0.7)
    if buy_signals:
        plt.scatter(*zip(*buy_signals), marker='^', color='g', label='Buy Signal')
    if sell_signals:
        plt.scatter(*zip(*sell_signals), marker='v', color='r', label='Sell Signal')
    plt.title('BTC Hourly - EMA Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs)')
    plt.legend()
    plt.savefig(output_file)
    plt.close()


def main(varySLTP, initial_capital, commission_pct, stop_loss_pct, take_profit_pct, asset_name, strategy):
    # Define input and output directories
    input_dir = 'input/'
    output_dir = 'output/'

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    file_path = input_dir + 'BTC-Hourly.csv'  # Use input directory for file path

    # Load data
    data = load_data(file_path)

    if varySLTP:
        # Define parameter ranges for varying SL and TP
        stop_loss_range = [x / 100 for x in range(1, 11)]  # 1% to 10%
        take_profit_range = [x / 100 for x in range(1, 11)]  # 1% to 10%

        # Vary SL and TP parameters and save results
        vary_parameters(data, initial_capital, commission_pct, stop_loss_range, take_profit_range, asset_name, strategy)
    else:
        # Run simulation with initial parameters
        total_pnl, pnl_percentage, trades, buy_signals, sell_signals = simulate_trading(
            data, initial_capital, stop_loss_pct, take_profit_pct, commission_pct
        )

        # Output the PnL results
        print(f"Net PnL: Rs {total_pnl:.2f}")
        print(f"Net PnL Percentage: {pnl_percentage:.2f}%")

        # Display trade details
        trades_df = pd.DataFrame(trades)
        print(trades_df)

        # Define output filenames based on asset and strategy
        csv_filename = f"{output_dir}trades_{asset_name}_{strategy}.csv"
        image_filename = f"{output_dir}plot_{asset_name}_{strategy}.png"

        # Save trades to a CSV file in the output directory
        trades_df.to_csv(csv_filename, index=False)

        # Plot results
        plot_signals(data, buy_signals, sell_signals, output_file=image_filename)


if __name__ == "__main__":
    # Set parameters to be supplied to main()
    initial_capital = 100000  # Rs 1,00,000
    commission_pct = 0.01  # 1% commission per trade
    stop_loss_pct = 0.04  # 4% stop loss
    take_profit_pct = 0.05  # 5% take profit
    asset_name = 'BTC'  # Asset name for file naming
    strategy = 'EMA_Crossover'  # Strategy name for file naming

    # Run main function with the defined parameters
    main(
        varySLTP=False,  # Change to False to run with initial parameters only
        initial_capital=initial_capital,
        commission_pct = commission_pct,
        stop_loss_pct = stop_loss_pct,
        take_profit_pct = take_profit_pct,
        asset_name = asset_name,
        strategy = strategy
    )
