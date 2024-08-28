import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend


def load_data(file_path):
    """
    Load BTC data from CSV file and calculate EMAs.

    Args:
    - file_path (str): Path to the CSV file containing BTC data.

    Returns:
    - pd.DataFrame: Data with calculated EMAs.
    """
    data = pd.read_csv(file_path, parse_dates=['date'], index_col='date')
    data['ShortEMA'] = data['close'].ewm(span=21, adjust=False).mean()
    data['LongEMA'] = data['close'].ewm(span=200, adjust=False).mean()
    return data


def simulate_trading(data, initial_capital, stop_loss_pct, take_profit_pct, commission_pct):
    """
    Simulate trading based on EMA Crossover strategy with stop loss and take profit.

    Args:
    - data (pd.DataFrame): BTC data with EMAs calculated.
    - initial_capital (float): Initial capital for trading.
    - stop_loss_pct (float): Stop loss percentage.
    - take_profit_pct (float): Take profit percentage.
    - commission_pct (float): Commission percentage per trade.

    Returns:
    - float: Total PnL.
    - float: PnL percentage.
    - list: List of trades with details.
    - list: Buy signals for plotting.
    - list: Sell signals for plotting.
    """
    capital = initial_capital
    position = 0  # 0 means no position, 1 means long position
    entry_price = 0
    buy_signals = []
    sell_signals = []
    trades = []

    for i in range(1, len(data)):
        if position == 0:  # No current position
            if data['ShortEMA'].iloc[i] > data['LongEMA'].iloc[i] and data['ShortEMA'].iloc[i - 1] <= \
                    data['LongEMA'].iloc[i - 1]:
                position = 1
                entry_price = data['close'].iloc[i]
                buy_signals.append((data.index[i], entry_price))
                trades.append(
                    {'type': 'long', 'entry_date': data.index[i], 'entry_price': entry_price, 'exit_date': None,
                     'exit_price': None, 'pnl': None})

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
                trades[-1].update({'exit_date': data.index[i], 'exit_price': exit_price, 'pnl': trade_net_profit})

    total_pnl = capital - initial_capital
    pnl_percentage = (total_pnl / initial_capital) * 100
    return total_pnl, pnl_percentage, trades, buy_signals, sell_signals


def plot_signals(data, buy_signals, sell_signals, output_file='output/BTC_Hourly_EMA_Strategy.png'):
    """
    Plot the BTC close prices, EMAs, and buy/sell signals.

    Args:
    - data (pd.DataFrame): BTC data with EMAs calculated.
    - buy_signals (list): List of buy signals.
    - sell_signals (list): List of sell signals.
    - output_file (str): Path to save the plot image.
    """
    plt.figure(figsize=(14, 7))
    plt.plot(data['close'], label='Close Price', alpha=0.5)
    plt.plot(data['ShortEMA'], label='21 EMA', linestyle='--', alpha=0.7)
    plt.plot(data['LongEMA'], label='200 EMA', linestyle='--', alpha=0.7)
    if buy_signals:
        plt.scatter(*zip(*buy_signals), marker='^', color='g', label='Buy Signal')
    if sell_signals:
        plt.scatter(*zip(*sell_signals), marker='v', color='r', label='Sell Signal')
    plt.title('BTC EMA Crossover Strategy')
    plt.xlabel('Date')
    plt.ylabel('Price (Rs)')
    plt.legend()
    plt.savefig(output_file)
    plt.show()


def main():
    # File path to your data
    file_path = 'input/BTC-Hourly.csv'

    # Strategy parameters
    initial_capital = 100000  # Rs 1,00,000
    stop_loss_pct = 0.04  # 4% stop loss
    take_profit_pct = 0.05  # 5% take profit
    commission_pct = 0.01  # 1% commission per trade

    # Load data
    data = load_data(file_path)

    # Simulate trading
    total_pnl, pnl_percentage, trades, buy_signals, sell_signals = simulate_trading(
        data, initial_capital, stop_loss_pct, take_profit_pct, commission_pct
    )

    # Output the PnL results
    print(f"Net PnL: Rs {total_pnl:.2f}")
    print(f"Net PnL Percentage: {pnl_percentage:.2f}%")

    # Display trade details
    trades_df = pd.DataFrame(trades)
    # print(trades_df)

    # Save trades to a CSV file
    trades_df.to_csv('output/trades_summary.csv', index=False)

    # Plot results
    plot_signals(data, buy_signals, sell_signals)


if __name__ == "__main__":
    main()
