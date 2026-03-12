import yfinance as yf 
import pandas as pd
import pandas_datareader.data as pdr
import numpy as np

from scipy.optimize import curve_fit
from scipy.stats import norm

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors

from IPython.display import display

import datetime
import itertools 
import os
import seaborn as sns

# Visualize historical risk-free rate (transitions from LIBOR --> SOFR) 
def plot_risk_free_rates(df_rates):
    plt.figure(figsize=(10, 4.5))
    
    # Using your Navy #003366 for consistency
    plt.plot(df_rates.index, df_rates['3M LIBOR/SOFR'] * 100, color='#003366', linewidth=1.5)

    # Formatting
    plt.title('3-Month LIBOR / SOFR History', fontsize=12, fontweight='bold', loc='left', color='#1C2C54')
    plt.ylabel('Annualized Interest Rate (%)', fontsize=10)
    plt.xlabel('Year', fontsize=10)

    # Grid and Style
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.axhline(0, color='black', linewidth=0.8)

    # X-Axis formatting
    plt.gca().xaxis.set_major_locator(mdates.YearLocator(5))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    
    plt.tight_layout()
    plt.show()

def plot_vrp(df, start_date=None, end_date=None):
    """
    Plots the aggregate Volatility Risk Premium across a specific timeframe.
    Excludes VRP Inversion from the legend for a cleaner presentation.
    """
    # 1. Filter the DataFrame by the requested timeframe
    plot_df = df.copy()
    if start_date:
        plot_df = plot_df[plot_df.index >= pd.to_datetime(start_date)]
    if end_date:
        plot_df = plot_df[plot_df.index <= pd.to_datetime(end_date)]

    # 2. Identify and Average VRP columns
    vrp_cols = [col for col in plot_df.columns if col.startswith('VRP_')]
    avg_vrp = plot_df[vrp_cols].mean(axis=1)
    
    # 3. 63-day rolling mean for structural trend
    rolling_vrp = avg_vrp.rolling(window=63).mean() 
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 4. Plotting
    ax.plot(avg_vrp.index, avg_vrp, color='#1C2C54', alpha=0.25, lw=0.8, 
            label='Daily Universe Average (Unfiltered)')
    
    ax.plot(rolling_vrp.index, rolling_vrp, color='#1C2C54', lw=2.5, 
            label='Macro VRP Index (63-Day Moving Average)')
    
    # 5. Shading Logic
    # Positive VRP (The Rent)
    ax.fill_between(rolling_vrp.index, rolling_vrp, 0, where=(rolling_vrp >= 0), 
                    color='#d4af37', alpha=0.25, label='Positive Risk Premium (IV > RV)')
    
    # Crimson shading for Inversions (No Label/Legend Entry)
    ax.fill_between(rolling_vrp.index, rolling_vrp, 0, where=(rolling_vrp < 0), 
                    color='crimson', alpha=0.25)
    
    # 6. Zero Line
    ax.axhline(0, color='black', lw=1.2, alpha=0.8)
    
    # 7. Institutional Formatting
    ax.set_title("The Volatility Risk Premium (Cross-Asset Average)", 
                 fontsize=15, fontweight='bold', loc='left', pad=25, color='#1C2C54')
    
    plt.annotate('Methodology: Equal-weighted average of VRP (IV - RV) across SPY, GLD, USO, FXE, and HYG.', 
                 xy=(0.01, -0.12), xycoords='axes fraction', fontsize=9, style='italic', color='#555555')
    
    ax.set_ylabel("Average VRP %", fontsize=10, fontweight='bold')
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=0))
    ax.grid(True, which='major', linestyle=':', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.show()


def run_backtest(master_df, df_rates, initial_capital=10000000, leverage=2.0, friction_bps=0.0005, short_borrow_rate=0.005, z_threshold=0.0, start_date=None, end_date=None, sizing_method='equal'):
    
    capital = initial_capital
    ledger = []
    
    # 1. Align dates across the Master Data and the Rates Data
    common_dates = master_df.index.intersection(df_rates.index).sort_values()
    
    # Filter the dates based on user input
    if start_date is not None:
        common_dates = common_dates[common_dates >= pd.to_datetime(start_date)]
    if end_date is not None:
        common_dates = common_dates[common_dates <= pd.to_datetime(end_date)]
        
    # Safety check
    if len(common_dates) < 2:
        print("Warning: Not enough overlapping dates in the specified range to run the backtest.")
        return pd.DataFrame()
        
    ttm = 30.0 / 365 # Constant 30-day proxy
    
    # Pre-identify Z-score columns to avoid searching for them inside the loop
    z_cols = [col for col in master_df.columns if col.startswith('Z_')]
    
    for i in range(1, len(common_dates)):
        today = common_dates[i]
        yesterday = common_dates[i-1]
        
        # PRO-TIP: Extract the full row once per day to drastically speed up the loop
        yest_data = master_df.loc[yesterday]
        today_data = master_df.loc[today]
        
        # 2. Signals & Rates
        yesterday_z = yest_data[z_cols]
        
        max_z = yesterday_z.max()
        short_asset = yesterday_z.idxmax().replace('Z_', '')
        
        min_z = yesterday_z.min()
        long_asset = yesterday_z.idxmin().replace('Z_', '')
        
        z_spread = max_z - min_z
        daily_rate = df_rates.loc[yesterday, '3M LIBOR/SOFR']
        
        # 3. Capital Allocation & Sizing
        gross_exposure = capital * leverage
        
        if z_spread >= z_threshold:
            if sizing_method == 'risk_parity':
                # Pull directly from the extracted row
                rv_short = yest_data[f'RV_{short_asset}']
                rv_long = yest_data[f'RV_{long_asset}']
                
                inv_rv_short = 1.0 / rv_short
                inv_rv_long = 1.0 / rv_long
                total_inv_rv = inv_rv_short + inv_rv_long
                
                weight_short = inv_rv_short / total_inv_rv
                weight_long = inv_rv_long / total_inv_rv
                
                notional_s = gross_exposure * weight_short
                notional_l = gross_exposure * weight_long
                
            else:
                notional_s = gross_exposure / 2.0
                notional_l = gross_exposure / 2.0
        else:
            notional_s = 0
            notional_l = 0
            
        # Initialize daily metrics to zero
        gamma_pnl_s = vega_pnl_s = theta_pnl_s = gross_pnl_short = hedging_cost_s = borrow_cost_s = net_pnl_short = 0
        gamma_pnl_l = vega_pnl_l = theta_pnl_l = gross_pnl_long = hedging_cost_l = net_pnl_long = 0
        
        # ==========================================
        # 4. The Spread Gate (Trade BOTH or NEITHER)
        # ==========================================
        if z_spread >= z_threshold:
            
            # --- SHORT LEG ---
            spot_yest_s = yest_data[f'Spot_{short_asset}']
            spot_today_s = today_data[f'Spot_{short_asset}']
            iv_yest_s = yest_data[f'IV_{short_asset}']
            iv_today_s = today_data[f'IV_{short_asset}']
            
            shares_s = notional_s / spot_yest_s  
            gamma_s, vega_s, theta_s = calculate_greeks(spot_yest_s, spot_yest_s, ttm, iv_yest_s, daily_rate)
            
            dS_s = spot_today_s - spot_yest_s
            dIV_s = iv_today_s - iv_yest_s
            
            gamma_pnl_s = -1 * (0.5 * gamma_s * (dS_s**2)) * shares_s
            vega_pnl_s = -1 * (vega_s * dIV_s) * shares_s
            theta_pnl_s = -1 * theta_s * shares_s
            gross_pnl_short = gamma_pnl_s + vega_pnl_s + theta_pnl_s
            
            shares_traded_s = abs(gamma_s * dS_s * shares_s)
            hedging_cost_s = shares_traded_s * spot_today_s * friction_bps
            
            borrow_cost_s = notional_s * (short_borrow_rate / 252.0) 
            net_pnl_short = gross_pnl_short - hedging_cost_s - borrow_cost_s
            
            # --- LONG LEG ---
            spot_yest_l = yest_data[f'Spot_{long_asset}']
            spot_today_l = today_data[f'Spot_{long_asset}']
            iv_yest_l = yest_data[f'IV_{long_asset}']
            iv_today_l = today_data[f'IV_{long_asset}']
            
            shares_l = notional_l / spot_yest_l  
            gamma_l, vega_l, theta_l = calculate_greeks(spot_yest_l, spot_yest_l, ttm, iv_yest_l, daily_rate)
            
            dS_l = spot_today_l - spot_yest_l
            dIV_l = iv_today_l - iv_yest_l
            
            gamma_pnl_l = (0.5 * gamma_l * (dS_l**2)) * shares_l
            vega_pnl_l = (vega_l * dIV_l) * shares_l
            theta_pnl_l = theta_l * shares_l
            gross_pnl_long = gamma_pnl_l + vega_pnl_l + theta_pnl_l
            
            shares_traded_l = abs(gamma_l * dS_l * shares_l)
            hedging_cost_l = shares_traded_l * spot_today_l * friction_bps
            net_pnl_long = gross_pnl_long - hedging_cost_l
            
        else:
            short_asset = 'None' 
            long_asset = 'None' 
            
        # ==========================================
        # 5. Aggregation & Ledger
        # ==========================================
        cash_yield = capital * (daily_rate / 252.0)
        daily_net_pnl = net_pnl_short + net_pnl_long + cash_yield
        capital += daily_net_pnl
        
        ledger.append({
            'Date': today,
            'Short_Asset': short_asset,
            'Long_Asset': long_asset,
            'Total_Gamma_PnL': gamma_pnl_s + gamma_pnl_l,
            'Total_Vega_PnL': vega_pnl_s + vega_pnl_l,
            'Total_Theta_PnL': theta_pnl_s + theta_pnl_l,
            'Gross_PnL_Short': gross_pnl_short,
            'Gross_PnL_Long': gross_pnl_long,
            'Hedging_Costs': hedging_cost_s + hedging_cost_l,
            'Borrow_Costs': borrow_cost_s,
            'Cash_Yield': cash_yield,
            'Daily_Net_PnL': daily_net_pnl,
            'Capital': capital
        })
        
    return pd.DataFrame(ledger).set_index('Date')

def calculate_greeks(spot, strike, time_to_maturity, iv, rate):
    # Protect against division by zero as expiration approaches
    T = max(time_to_maturity, 1e-5) 
    S = spot
    K = strike
    sigma = iv / 100.0  # Convert IV from percentage to decimal
    r = rate            # Risk-free rate (already decimal)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    pdf_d1 = norm.pdf(d1)
    cdf_d2 = norm.cdf(d2)
    cdf_neg_d2 = norm.cdf(-d2)
    
    # Gamma (doubled for straddle)
    gamma_single = pdf_d1 / (S * sigma * np.sqrt(T))
    straddle_gamma = 2 * gamma_single
    
    # Vega (doubled, divided by 100 for a 1-point IV move)
    vega_single = S * np.sqrt(T) * pdf_d1
    straddle_vega = (2 * vega_single) / 100.0
    
    # Theta (daily decay, divided by 252)
    theta_common = -(S * sigma * pdf_d1) / (2 * np.sqrt(T))
    call_theta = theta_common - r * K * np.exp(-r * T) * cdf_d2
    put_theta = theta_common + r * K * np.exp(-r * T) * cdf_neg_d2
    straddle_theta = (call_theta + put_theta) / 252.0
    
    return straddle_gamma, straddle_vega, straddle_theta

def get_performance(results_df, strategy_name="2.0x VRP Macro", initial_capital=10000000):
    res = results_df.copy()
    res['Daily_Return'] = res['Capital'].pct_change()
    
    # 1. Performance & Risk Math
    total_return = (res['Capital'].iloc[-1] / initial_capital) - 1
    ann_return = (1 + total_return) ** (252 / len(res)) - 1
    ann_vol = res['Daily_Return'].std() * np.sqrt(252)
    sharpe = (res['Daily_Return'].mean() * 252) / ann_vol if ann_vol != 0 else 0
    max_dd = ((res['Capital'] - res['Capital'].cummax()) / res['Capital'].cummax()).min()
    
    # 2. Activity Metrics (Simpler, more transparent)
    active_days = res[res['Short_Asset'] != 'None'].shape[0]
    pct_active = active_days / len(res)
    
    # 3. The PnL Bridge Components
    gross_opt_pnl = res['Total_Gamma_PnL'].sum() + res['Total_Vega_PnL'].sum() + res['Total_Theta_PnL'].sum()
    cash_yield = res['Cash_Yield'].sum()
    hedge_costs = res['Hedging_Costs'].sum()
    borrow_costs = res['Borrow_Costs'].sum()
    net_pnl = res['Daily_Net_PnL'].sum()
    
    # 4. Create Structured DataFrame
    stats = {
        'Ann Return': f"{ann_return:.2%}",
        'Ann Vol': f"{ann_vol:.2%}",
        'Sharpe': f"{sharpe:.2f}",
        'Max DD': f"{max_dd:.2%}",
        'Active Days %': f"{pct_active:.1%}",
        'Trading Days': f"{active_days:,}", # Updated from Total Trades
        
        'Gross Opt PnL': f"${gross_opt_pnl:,.0f}",
        'Cash Yield': f"${cash_yield:,.0f}",
        'Hedge Costs': f"-${hedge_costs:,.0f}",
        'Borrow Costs': f"-${borrow_costs:,.0f}",
        
        'Net PnL': f"<b>${net_pnl:,.0f}</b>",
        'Final Equity': f"<b>${res['Capital'].iloc[-1]:,.0f}</b>"
    }
    
    summary = pd.DataFrame(stats, index=[strategy_name])
    
    # 5. Visual "Gold Standard" Styling
    styled_df = summary.style.set_properties(**{
        'text-align': 'center', 
        'padding': '14px',
        'border': '1px solid #eeeeee',
        'color': '#000000' # Global Black Text
    }).set_table_styles([
        
        # Header Styling: Professional Navy & Gold
        {'selector': 'thead th', 'props': [
            ('background-color', '#1C2C54 !important'), 
            ('color', 'white !important'), 
            ('font-weight', 'bold'), 
            ('text-transform', 'uppercase'),
            ('letter-spacing', '1px'),
            ('border-bottom', '3px solid #d4af37 !important'), # GOLD ACCENT
            ('text-align', 'center'),
            ('border', '1px solid #1C2C54')
        ]},
        
        # Row Labels: Modern subtle gray
        {'selector': 'tbody th', 'props': [
            ('text-align', 'left'), 
            ('font-weight', 'bold'),
            ('background-color', '#ffffff !important'),
            ('color', '#333333'),
            ('padding-right', '20px'),
            ('border-right', '1px solid #dee2e6')
        ]},
        
        # Highlight: PnL Bridge (Gross to Costs) - Light Yellow/Gold
        {'selector': 'td.col6, td.col7, td.col8, td.col9', 'props': [
            ('background-color', '#fffdf0')
        ]},
        
        # Highlight: Final Results (Net PnL & Equity) - Light Blue Focus
        {'selector': 'td.col10, td.col11', 'props': [
            ('background-color', '#f0f7ff'),
            ('font-weight', 'bold'),
            ('color', '#1C2C54')
        ]},
        
        # Section Dividers (Vertical)
        {'selector': 'td.col5, thead th.col5', 'props': [('border-right', '2px solid #aaa !important')]},
        {'selector': 'td.col9, thead th.col9', 'props': [('border-right', '2px solid #aaa !important')]}
        
    ], overwrite=True) # Forces removal of any default Jupyter CSS
    
    return styled_df

def get_asset_attribution(results_df):
    """
    Calculates and formats the Gross PnL attribution by individual asset class,
    broken down by Short Volatility and Long Volatility trades, with a master total.
    """
    # 1. Sum up the money made/lost when SHORTING each asset's volatility
    short_pnl = results_df.groupby('Short_Asset')['Gross_PnL_Short'].sum()
    
    # 2. Sum up the money made/lost when LONGING each asset's volatility
    long_pnl = results_df.groupby('Long_Asset')['Gross_PnL_Long'].sum()
    
    # 3. Combine into a master attribution table
    attribution = pd.DataFrame({
        'Short_Vol_PnL': short_pnl,
        'Long_Vol_PnL': long_pnl
    }).fillna(0)
    
    # 4. Drop the 'None' category (days where the machine sat in cash)
    if 'None' in attribution.index:
        attribution = attribution.drop('None')
        
    # 5. Calculate Net PnL per asset and sort to find the heavy hitters
    attribution['Total_Asset_PnL'] = attribution['Short_Vol_PnL'] + attribution['Long_Vol_PnL']
    attribution = attribution.sort_values(by='Total_Asset_PnL', ascending=False)
    
    # ==========================================
    # NEW: 6. Append the Master Total Row
    # ==========================================
    attribution.loc['Total Option Gross PnL'] = attribution.sum()
    
    # 7. Apply presentation formatting
    styled_attribution = attribution.style.format("${:,.0f}") \
        .set_properties(**{'text-align': 'right', 'padding': '10px'}) \
        .set_table_styles([
            # Transparent Level 0 Titles
            {'selector': 'th.col_heading', 'props': [('text-align', 'right'), ('background-color', '#f8f9fa')]},
            {'selector': 'th.row_heading', 'props': [('text-align', 'left')]},
            
            # Add a vertical divider before the total column
            {'selector': 'th.col2, td.col2', 'props': [('border-left', '2px solid black'), ('font-weight', 'bold')]},
            
            # NEW: Make the bottom Total Row pop (Heavy top border, bold, light blue)
            {'selector': 'tr:last-child td, tr:last-child th', 
             'props': [('border-top', '2px solid black'), ('font-weight', 'bold'), ('background-color', '#e6f4ff')]}
        ])
        
    return styled_attribution

def get_greek_attribution(results_df):
    """
    Isolates the portfolio's options Greeks into a clean, presentation-ready 
    vertical table with descriptors left-aligned next to the Greek names.
    """
    # 1. Calculate the core Greeks
    gamma = results_df['Total_Gamma_PnL'].sum()
    vega = results_df['Total_Vega_PnL'].sum()
    theta = results_df['Total_Theta_PnL'].sum()
    
    # 2. Build the DataFrame with a MultiIndex for a "left-shifted" descriptor look
    # We use a list of tuples to create the hierarchy
    index = pd.MultiIndex.from_tuples([
        ('Gamma', 'Price Curvature'),
        ('Vega', 'Implied Volatility'),
        ('Theta', 'Time Decay'),
        ('Total Option Gross PnL', '')
    ])
    
    greeks = pd.DataFrame({
        'PnL Contribution': [gamma, vega, theta, (gamma + vega + theta)]
    }, index=index)
    
    # 3. Apply presentation formatting
    styled_greeks = greeks.style.format("${:,.0f}") \
        .set_properties(**{'text-align': 'right', 'padding': '10px'}) \
        .set_table_styles([
            # Hide the empty index names header row for maximum cleanliness
            {'selector': 'th.index_name', 'props': [('display', 'none')]},
            
            # Headers and Index Alignment
            {'selector': 'th.col_heading', 'props': [('text-align', 'right'), ('background-color', '#f8f9fa')]},
            {'selector': 'th.row_heading', 'props': [('text-align', 'left'), ('font-weight', 'bold')]},
            
            # Formatting the second level of the index (the descriptors)
            {'selector': 'th.level1', 'props': [('font-weight', 'normal'), ('padding-left', '5px')]},
            
            # The "Total" row visual break (Top border and light red background)
            {'selector': 'tr:last-child td, tr:last-child th', 
             'props': [('border-top', '2px solid black'), 
                       ('font-weight', 'bold'), 
                       ('background-color', '#fff0f0'),
                       ('color', 'black')]}
        ])
        
    return styled_greeks

# Plot growth of the portfolio over time, highlighting pure profit above the initial watermark.
def plot_equity(results_df, strategy_name="VRP Macro Strategy", initial_capital=10000000):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    
    # 1. Main Equity Line (Vibrant Green)
    ax.plot(results_df.index, results_df['Capital'], color='#27ae60', lw=2, label=strategy_name)
    
    # 2. Mountain Shading
    # Green fill for profit
    ax.fill_between(results_df.index, results_df['Capital'], initial_capital, 
                    where=(results_df['Capital'] >= initial_capital), 
                    color='#27ae60', alpha=0.15, interpolate=True)
    
    # Red fill for drawdowns below initial principal
    ax.fill_between(results_df.index, results_df['Capital'], initial_capital, 
                    where=(results_df['Capital'] < initial_capital), 
                    color='#e74c3c', alpha=0.15, interpolate=True)
    
    # 3. Baseline Reference
    ax.axhline(initial_capital, color='#333333', lw=1.2, linestyle='--', alpha=0.6, label='Initial Capital ($10M)')
    
    # 4. Y-Axis Formatter (Millions)
    formatter = mtick.FuncFormatter(lambda x, pos: f'${x*1e-6:,.1f}M')
    ax.yaxis.set_major_formatter(formatter)
    
    # 5. Clean Institutional Labels
    ax.set_title(f"Portfolio Value: {strategy_name}", fontsize=14, fontweight='bold', loc='left', pad=15, color='#1C2C54')
    ax.set_ylabel("Total Net Equity (USD)", fontsize=10, fontweight='bold', color='#333333')
    ax.set_xlabel("") 
    
    # 6. Grid and Spines
    ax.grid(True, which='major', linestyle=':', alpha=0.5) # Using the subtle dot grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 7. Legend
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    
    plt.tight_layout()
    plt.show()

# Plots peak-to-trough drawdowns over time
def plot_drawdown(results_df):
    rolling_max = results_df['Capital'].cummax()
    drawdown = (results_df['Capital'] - rolling_max) / rolling_max
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # 1. Plot the underwater curve and add the "blood" shading
    ax.plot(drawdown.index, drawdown, color='#e74c3c', lw=1.5)
    ax.fill_between(drawdown.index, drawdown, 0, color='#c0392b', alpha=0.3)
    
    # NEW: Explicitly draw the "Water Surface" (0% Drawdown Line)
    ax.axhline(0, color='black', lw=1.5)
    
    # 2. Format the Y-Axis into clean Percentages (e.g., -15%)
    formatter = mtick.FuncFormatter(lambda x, pos: f'{x*100:.0f}%')
    ax.yaxis.set_major_formatter(formatter)
    
    # Force the top of the chart to rigidly stop at 0%
    ax.set_ylim(bottom=drawdown.min() * 1.1, top=0) 
    
    # 3. Clean Titling and Labels
    ax.set_title("Historical Drawdown (% Below Peak)", fontsize=14, fontweight='bold', loc='left', pad=15, color='#1C2C54')
    ax.set_ylabel("Drawdown Depth", fontsize=12, fontweight='bold')
    ax.set_xlabel("") 
    
    # 4. Grid and Spines 
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # (Removed the code that hid the bottom spine so the X-axis returns)
    
    plt.tight_layout()
    plt.show()

def get_full_alpha_validation_table(results_df, market_data, start_date='2020-01-01', end_date=None):
    """
    Dynamic 5-Column Alpha Exhibit: Strategy vs. Equity, Short Vol, Long Vol, and Trend (DBMF).
    """
    # 1. Prepare Daily Returns
    strat_ret = (results_df['Daily_Net_PnL'] / results_df['Capital'].shift(1))
    equity_ret = market_data['Spot_Equity'].pct_change(fill_method=None)
    short_vol = market_data['SVXY'].pct_change(fill_method=None)
    long_vol = market_data['VXX'].pct_change(fill_method=None)
    trend_ret = market_data['DBMF'].pct_change(fill_method=None)
    
    # 2. Combine and Slice
    combined = pd.concat([
        strat_ret.to_frame('Strategy'), 
        equity_ret.to_frame('S&P 500'), 
        short_vol.to_frame('Short Vol (SVXY)'),
        long_vol.to_frame('Long Vol (VXX)'),
        trend_ret.to_frame('Trend (DBMF)')
    ], axis=1)
    
    if start_date:
        combined = combined.loc[start_date:]
    if end_date:
        combined = combined.loc[:end_date]
        
    combined = combined.dropna()

    # 3. Dynamic Date Range for Header
    actual_start = combined.index.min().strftime('%Y')
    actual_end = combined.index.max().strftime('%Y')
    date_range_str = f"{actual_start} – {actual_end}"

    # 4. Performance Stats Helper
    def get_metrics(series):
        ann_ret = (1 + series.mean())**252 - 1
        ann_vol = series.std() * np.sqrt(252)
        sharpe = (series.mean() * 252) / ann_vol if ann_vol != 0 else 0
        wealth = (1 + series).cumprod()
        dd = (wealth - wealth.cummax()) / wealth.cummax()
        return [f"{ann_ret:.2%}", f"{ann_vol:.2%}", f"{sharpe:.2f}", f"{dd.min():.2%}"]

    # 5. Build Comparison Matrix
    stats_data = {
        "Strategy": [date_range_str] + get_metrics(combined['Strategy']),
        "Equity (SPY)": [date_range_str] + get_metrics(combined['S&P 500']),
        "Short Vol (SVXY)": [date_range_str] + get_metrics(combined['Short Vol (SVXY)']),
        "Long Vol (VXX)": [date_range_str] + get_metrics(combined['Long Vol (VXX)']),
        "Trend (DBMF)": [date_range_str] + get_metrics(combined['Trend (DBMF)'])
    }
    
    metrics_labels = ["Analysis Period", "Annualized Return", "Annualized Vol", "Sharpe Ratio", "Max Drawdown"]
    df_alpha = pd.DataFrame(stats_data, index=metrics_labels)

    # 6. Styling: Locked-in Gold Standard
    styled = df_alpha.style.set_properties(**{
        'text-align': 'center', 'padding': '14px', 'border': '1px solid #eeeeee', 'color': '#000000'
    }).set_table_styles([
        {'selector': 'th', 'props': [
            ('background-color', '#1C2C54'), ('color', 'white'), 
            ('font-weight', 'bold'), ('text-transform', 'uppercase'),
            ('letter-spacing', '1px'), ('border-bottom', '3px solid #d4af37'), 
            ('text-align', 'center'), ('border', '1px solid #1C2C54')
        ]},
        {'selector': 'th.row_heading', 'props': [
            ('text-align', 'left'), ('font-weight', 'bold'), 
            ('background-color', '#ffffff'), ('color', '#333333'),
            ('padding-right', '20px'), ('border', '1px solid #eeeeee')
        ]},
        {'selector': 'tr:nth-child(1) td', 'props': [
            ('font-weight', 'bold'), ('color', '#555555'), 
            ('background-color', '#ffffff'), ('border-bottom', '2px solid #f0f0f0')
        ]},
        {'selector': 'td.col0', 'props': [
            ('background-color', '#f0f7ff'), ('font-weight', 'bold'), ('color', '#1C2C54')
        ]},
        {'selector': 'tr:last-child th, tr:last-child td', 'props': [
            ('border-bottom', '2.5px solid #333333'), ('font-weight', '900')
        ]}
    ])
    
    return styled, combined.corr()

def plot_equity_with_benchmark(results_df, market_data, initial_capital=10000000):
    """
    Platinum Standard Wealth Index: Strategy vs. S&P 500.
    Highlights alpha generation with high-water mark shading.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Scaled S&P 500 Benchmark Line
    # Scale the S&P 500 Spot_Equity to start at $10M
    spy_prices = market_data['Spot_Equity'].loc[results_df.index]
    spy_wealth = (spy_prices / spy_prices.iloc[0]) * initial_capital
    
    # Main Strategy Equity Line
    # Using #1C2C54 (Navy) for a high-end institutional feel
    ax.plot(results_df.index, results_df['Capital'], color='#1C2C54', 
            lw=2.8, label='VRP Macro Strategy', zorder=5)
    
    ax.plot(results_df.index, spy_wealth, color='#7f8c8d', lw=1.5, 
            linestyle='-', alpha=0.9, label='S&P 500 Benchmark (Scaled)')

    # Initial Capital Baseline
    ax.axhline(initial_capital, color='black', lw=1.2, linestyle='--', 
               alpha=0.8, label='Initial Capital ($10M)', zorder=2)
    
    # Professional Formatting
    # Format Y-Axis into Millions
    formatter = mtick.FuncFormatter(lambda x, pos: f'${x*1e-6:,.1f}M')
    ax.yaxis.set_major_formatter(formatter)
    
    # Labels & Title
    ax.set_title("Strategy Performance vs. S&P 500 Benchmark", 
                 fontsize=18, fontweight='bold', loc='left', pad=20, color='#1C2C54')
    ax.set_ylabel("Portfolio Value (USD)", fontsize=11, fontweight='bold', color='#444')
    ax.set_xlabel("") 
    
    # Grid & Spine Polish
    ax.grid(True, which='major', linestyle=':', alpha=0.5, color='#aaa')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#ddd')
    ax.spines['bottom'].set_color('#ddd')
    
    # Legend Placement (Lower Right often better for rising equity curves)
    ax.legend(loc='upper left', frameon=False, fontsize=11, ncol=1)
    
    plt.tight_layout()
    plt.show()

def get_benchmark_comparison_table(results_df, market_data, strategy_name="2.0x Risk Parity"):
    """
    Platinum Standard Exhibit: Matches the Alpha Validation Table styling.
    """
    # 1. Align Returns and Extract Date Range
    strat_returns = (results_df['Daily_Net_PnL'] / results_df['Capital'].shift(1)).dropna()
    bench_returns = market_data['Spot_Equity'].loc[results_df.index].pct_change(fill_method=None).dropna()
    
    combined = pd.concat([strat_returns, bench_returns], axis=1).dropna()
    combined.columns = ['Strategy', 'S&P 500']
    
    start_year = combined.index.min().year
    end_year = combined.index.max().year
    date_range_str = f"{start_year} – {end_year}"

    # 2. Performance Logic (FIXED: Calculates wealth directly from aligned series)
    def get_stats(series):
        total_return = (1 + series).prod() - 1
        ann_ret = (1 + total_return) ** (252 / len(series)) - 1
        ann_vol = series.std() * np.sqrt(252)
        sharpe = (series.mean() * 252) / ann_vol if ann_vol != 0 else 0
        
        wealth = (1 + series).cumprod()
        dd = (wealth - wealth.cummax()) / wealth.cummax()
        return ann_ret, ann_vol, sharpe, dd.min()

    s_ret, s_vol, s_sh, s_dd = get_stats(combined['Strategy'])
    b_ret, b_vol, b_sh, b_dd = get_stats(combined['S&P 500'])

    # 3. Relationship Metrics
    corr = combined.corr().iloc[0, 1]
    beta = combined.cov().iloc[0, 1] / combined['S&P 500'].var()
    
    # 4. Create Comparison DataFrame
    comparison_data = {
        strategy_name: [
            date_range_str, f"{s_ret:.2%}", f"{s_vol:.2%}", f"{s_sh:.2f}", f"{s_dd:.2%}", f"{corr:.3f}", f"{beta:.3f}"
        ],
        "S&P 500 (SPY)": [
            date_range_str, f"{b_ret:.2%}", f"{b_vol:.2%}", f"{b_sh:.2f}", f"{b_dd:.2%}", "1.000", "1.000"
        ]
    }
    
    metrics = ["Analysis Period", "Annualized Return", "Annualized Vol", "Sharpe Ratio", "Max Drawdown", "Correlation to S&P", "Beta vs S&P"]
    df_compare = pd.DataFrame(comparison_data, index=metrics)
    df_compare.index.name = None 
    
    # 5. Styling: Hard-locking formatting to match Alpha Validation Table
    styled_compare = df_compare.style.set_properties(**{
        'text-align': 'center', 
        'padding': '14px',
        'border': '1px solid #eeeeee',
        'color': '#000000' # Global Black Text
    }) \
    .set_table_styles([
        # Header Styling: Navy with Gold Base
        {'selector': 'th', 'props': [
            ('background-color', '#1C2C54'), 
            ('color', 'white'), 
            ('font-weight', 'bold'), 
            ('text-transform', 'uppercase'),
            ('letter-spacing', '1px'),
            ('border-bottom', '3px solid #d4af37'), # GOLD ACCENT
            ('text-align', 'center'),
            ('border', '1px solid #1C2C54')
        ]},
        # Row Labels Styling
        {'selector': 'th.row_heading', 'props': [
            ('text-align', 'left'), 
            ('font-weight', 'bold'),
            ('background-color', '#ffffff'),
            ('color', '#333333'),
            ('padding-right', '20px'),
            ('border', '1px solid #eeeeee')
        ]},
        # Analysis Period Row (Row 0)
        {'selector': 'tr:nth-child(1) td', 'props': [
            ('font-weight', 'bold'),
            ('color', '#555555'),
            ('background-color', '#ffffff'),
            ('border-bottom', '2px solid #f0f0f0')
        ]},
        # Strategy Column Focus (Light Blue)
        {'selector': 'td.col0', 'props': [
            ('background-color', '#f0f7ff'), 
            ('font-weight', 'bold'), 
            ('color', '#1C2C54')
        ]},
        # S&P Column (Clean White)
        {'selector': 'td.col1', 'props': [
            ('background-color', '#ffffff'), 
            ('color', '#444444'),
            ('font-weight', '400')
        ]},
        # Bottom Bold Anchor (Edge-to-Edge)
        {'selector': 'tr:last-child th, tr:last-child td', 'props': [
            ('border-bottom', '2.5px solid #333333'),
            ('font-weight', '900')
        ]}
    ])
    
    return styled_compare

def get_annual_returns(results_df, market_data, strategy_name="2.0x Risk Parity"):
    # 1. Align and Calculate Daily Returns
    strat_daily = (results_df['Daily_Net_PnL'] / results_df['Capital'].shift(1)).dropna()
    bench_daily = market_data['Spot_Equity'].loc[results_df.index].pct_change().dropna()
    
    combined = pd.concat([strat_daily, bench_daily], axis=1).dropna()
    combined.columns = ['Strategy', 'S&P 500']
    
    # 2. Group by Year and Calculate Geometric Annual Returns
    annual_stats = combined.groupby(combined.index.year).apply(lambda x: (1 + x).prod() - 1)
    
    # --- PARTIAL YEAR LABELING ---
    last_year = annual_stats.index[-1]
    last_date_str = combined.index.max().strftime('%B').upper()
    annual_stats = annual_stats.rename(index={last_year: f"{last_year} (TO {last_date_str})"})
    
    # 3. Calculate "Since Inception" and CAGR (FIXED: Locked to aligned geometric returns)
    cum_strat = (1 + combined['Strategy']).prod() - 1
    cum_bench = (1 + combined['S&P 500']).prod() - 1
    
    cagr_strat = (1 + cum_strat) ** (252 / len(combined)) - 1
    cagr_bench = (1 + cum_bench) ** (252 / len(combined)) - 1
    
    # 4. Assemble DataFrame
    summary_rows = pd.DataFrame(
        [[cum_strat, cum_bench], [cagr_strat, cagr_bench]], 
        columns=['Strategy', 'S&P 500'], 
        index=['Since Inception', 'Compound Annual Return (CAGR)']
    )
    final_df = pd.concat([annual_stats, summary_rows])
    final_df.columns = [strategy_name, "S&P 500 (SPY)"]
    
    # 5. Apply the "Gold Standard" Styling
    styled = final_df.style.format("{:.2%}") \
        .set_properties(**{
            'text-align': 'center',
            'padding': '14px',
            'border': '1px solid #eeeeee', # Subtle internal grid
            'color': '#000000'
        }) \
        .set_table_styles([
            # Navy Header with Gold Accent Line
            {'selector': 'th', 'props': [
                ('background-color', '#1C2C54'), 
                ('color', 'white'), 
                ('font-weight', 'bold'), 
                ('text-transform', 'uppercase'),
                ('letter-spacing', '1px'),
                ('border-bottom', '3px solid #d4af37'), # GOLD ACCENT
                ('text-align', 'center'),
                ('border', '1px solid #1C2C54')
            ]},
            # Row Labels
            {'selector': 'th.row_heading', 'props': [
                ('text-align', 'left'), 
                ('font-weight', 'bold'),
                ('background-color', '#ffffff'),
                ('color', '#333'),
                ('padding-right', '20px'),
                ('border', '1px solid #eeeeee')
            ]},
            # Strategy Column (Light Blue Focus)
            {'selector': 'td.col0', 'props': [
                ('background-color', '#f0f7ff'), 
                ('font-weight', 'bold'),
                ('color', '#1C2C54')
            ]},
            # S&P Column (Clean White)
            {'selector': 'td.col1', 'props': [
                ('background-color', '#ffffff'), 
                ('color', '#333'),
                ('font-weight', '400')
            ]},
            # --- FULL WIDTH SUMMARY DIVIDERS (Edge-to-Edge) ---
            # Divider ABOVE "Since Inception"
            {'selector': 'tr:nth-last-child(2) th, tr:nth-last-child(2) td', 'props': [
                ('border-top', '2.5px solid #333333'),
                ('font-weight', '900')
            ]},
            # Font size bump for summary rows
            {'selector': 'tr:nth-last-child(2) td, tr:last-child td', 'props': [
                ('font-size', '1.05em')
            ]},
            # Divider BELOW "CAGR"
            {'selector': 'tr:last-child th, tr:last-child td', 'props': [
                ('border-bottom', '2.5px solid #333333'),
                ('font-weight', '900')
            ]}
        ])

    return styled

def plot_equity_comparison(res_eq, res_rp, initial_capital=10000000):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    
    # 1. Plot Equal Weight (Blue)
    ax.plot(res_eq.index, res_eq['Capital'], 
            color='#1f77b4', lw=2.5, label='Equal Weight Sizing')
    
    # 2. Plot Risk Parity (Red)
    ax.plot(res_rp.index, res_rp['Capital'], 
            color='#d62728', lw=2.5, label='Risk Parity Sizing')
    
    # 3. Add the Initial Capital Baseline
    ax.axhline(initial_capital, color='black', lw=1.5, linestyle='--', alpha=0.8, label='Initial Capital ($10M)')
    
    # 4. Format the Y-Axis into Millions
    formatter = mtick.FuncFormatter(lambda x, pos: f'${x*1e-6:,.1f}M')
    ax.yaxis.set_major_formatter(formatter)
    
    # 5. Clean Titling and Labels
    ax.set_title("Strategy Growth Comparison: Equal Weight vs. Risk Parity", 
                 fontsize=16, fontweight='bold', loc='left')
    ax.set_ylabel("Total Equity", fontsize=12, fontweight='bold')
    
    # 6. Grid and Spines
    ax.grid(True, which='major', linestyle='--', alpha=0.4)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # 7. Legend Placement
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.show()

def get_strategy_comparison(res_eq, res_rp):
    """
    Takes pre-calculated result DataFrames and generates a styled 
    comparison table. 
    """
    # 1. Map results to their respective names
    strategies = [
        {'name': 'Equal Weight', 'df': res_eq},
        {'name': 'Inverse Vol.', 'df': res_rp}
    ]

    comparison_rows = []

    for strat in strategies:
        # 2. Generate metrics using your existing performance function
        perf_summary = get_performance(strat['df'], strategy_name=strat['name'])
        
        # Extract the raw data if it's currently a Styler object
        raw_df = perf_summary.data if hasattr(perf_summary, 'data') else perf_summary
        comparison_rows.append(raw_df)

    # 3. Concatenate into a single Master Comparison table
    master_comparison = pd.concat(comparison_rows)

    # 4. Apply the EXACT original styling logic
    styled_comparison = master_comparison.style.set_properties(**{
        'text-align': 'center', 
        'padding': '14px',
        'border': '1px solid #eeeeee',
        'color': '#000000' # Force Global Black Text
    }) \
    .set_table_styles([
        # Header Styling: Professional Navy & Gold
        {'selector': 'th', 'props': [
            ('background-color', '#1C2C54'), 
            ('color', 'white'), 
            ('font-weight', 'bold'), 
            ('text-transform', 'uppercase'),
            ('letter-spacing', '1px'),
            ('border-bottom', '3px solid #d4af37'), # GOLD ACCENT
            ('text-align', 'center'),
            ('border', '1px solid #1C2C54')
        ]},
        # Row Labels: Modern subtle gray
        {'selector': 'th.row_heading', 'props': [
            ('text-align', 'left'), 
            ('font-weight', 'bold'),
            ('background-color', '#fdfdfd'),
            ('color', '#333333'),
            ('padding-right', '20px'),
            ('border-right', '1px solid #dee2e6')
        ]},
        # Highlight: PnL Bridge (Gross to Costs) - Light Yellow/Gold
        {'selector': 'td.col6, td.col7, td.col8, td.col9', 'props': [
            ('background-color', '#fffdf0')
        ]},
        # Highlight: Final Results (Net PnL & Equity) - Light Blue Focus
        {'selector': 'td.col10, td.col11', 'props': [
            ('background-color', '#f0f7ff'),
            ('font-weight', 'bold'),
            ('color', '#1C2C54')
        ]},
        # Section Dividers (Vertical)
        {'selector': 'td.col5, th.col5', 'props': [('border-right', '2px solid #aaa')]},
        # The column index for the next divider depends on your get_performance columns.
        # Assuming col9 remains the boundary for the cost section:
        {'selector': 'td.col9, th.col9', 'props': [('border-right', '2px solid #aaa')]}
    ])

    return styled_comparison

def plot_comparison_window(master_df, df_rates, 
                           start_date='2019-01-01', end_date='2021-12-31', 
                           z_thresh=0.0, initial_capital=10000000):
    """
    Generates a comparative equity curve for Equal Weight vs. Risk Parity sizing.
    Uses the unified master_df and institutional Navy/Gold branding.
    """
    # 1. Run Backtests internally for the specified window
    res_eq = run_backtest(
        master_df, 
        df_rates,
        start_date=start_date, 
        end_date=end_date,
        z_threshold=z_thresh, 
        sizing_method='equal',
        initial_capital=initial_capital
    )
    
    res_rp = run_backtest(
        master_df, 
        df_rates,
        start_date=start_date, 
        end_date=end_date,
        z_threshold=z_thresh, 
        sizing_method='risk_parity',
        initial_capital=initial_capital
    )

    # 2. Setup Figure
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # 3. Plot Lines (Aligned with Navy & Gold Standard)
    # Equal Weight as the secondary comparison line
    ax.plot(res_eq.index, res_eq['Capital'], 
            color='#95a5a6', alpha=0.8, lw=2.0, label='Equal Weight Sizing')
    
    # Risk Parity as the primary focus
    ax.plot(res_rp.index, res_rp['Capital'], 
            color='#1C2C54', alpha=0.9, lw=2.5, label='Risk Parity Sizing')
    
    # 4. Add the Initial Capital Baseline (Gold Accent)
    ax.axhline(initial_capital, color='#d4af37', lw=1.5, linestyle='--', alpha=0.9, label='Initial Capital')
    
    # 5. Format the Y-Axis into Millions
    formatter = mtick.FuncFormatter(lambda x, pos: f'${x*1e-6:,.1f}M')
    ax.yaxis.set_major_formatter(formatter)
    
    # 6. Professional Titling
    date_range_str = f"({start_date} to {end_date})"
    ax.set_title(f"Strategy Growth Comparison", 
                 fontsize=14, fontweight='bold', loc='left', color='#1C2C54')
    ax.set_ylabel("Total Equity", fontsize=12, fontweight='bold', color='#333333')
    
    # 7. Grid and Spines Cleanup
    ax.grid(True, which='major', linestyle='--', alpha=0.4, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')
    
    # 8. Legend Placement
    ax.legend(loc='upper left', frameon=False, fontsize=11)
    
    plt.tight_layout()
    plt.show()

def plot_drawdown_comparison(master_df, df_rates, 
                             start_date='2019-01-01', end_date='2021-12-31', 
                             z_thresh=0.0):
    """
    Plots the underwater (drawdown) curves for Equal Weight vs. Risk Parity.
    Uses the unified master_df and institutional Navy/Gold/Crimson branding.
    """
    # 1. Run Backtests internally via unified Master DataFrame
    res_eq = run_backtest(
        master_df, 
        df_rates,
        start_date=start_date, 
        end_date=end_date,
        z_threshold=z_thresh, 
        sizing_method='equal'
    )
    
    res_rp = run_backtest(
        master_df, 
        df_rates,
        start_date=start_date, 
        end_date=end_date,
        z_threshold=z_thresh, 
        sizing_method='risk_parity'
    )

    # 2. Calculate Drawdowns
    dd_eq = (res_eq['Capital'] - res_eq['Capital'].cummax()) / res_eq['Capital'].cummax()
    dd_rp = (res_rp['Capital'] - res_rp['Capital'].cummax()) / res_rp['Capital'].cummax()

    # 3. Setup Figure
    fig, ax = plt.subplots(figsize=(10, 4.5))
    
    # 4. Plot Drawdowns (Navy & Crimson Scheme)
    
    # Equal Weight: The "Sea of Crimson" (Filled area to show pain)
    ax.fill_between(res_eq.index, dd_eq, 0, color='#d62728', alpha=0.25, label='Equal Weight Drawdown')
    
    # Risk Parity: The "Safe Passage" (Solid Navy Line)
    ax.plot(res_rp.index, dd_rp, color='#1C2C54', alpha=0.9, lw=2.5, label='Risk Parity Drawdown')

    # 5. Add Institutional Guardrail (-20% line) using Gold Accent
    ax.axhline(-0.20, color='#d4af37', lw=1.5, linestyle='--', alpha=0.9, label='-20% Drawdown')
    
    # 6. Format Y-Axis as Percentages
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    
    # 7. Professional Titling
    date_range_str = f"({start_date} to {end_date})"
    ax.set_title(f"Historical Drawdown Comparison", 
                 fontsize=14, fontweight='bold', loc='left', color='#1C2C54')
    ax.set_ylabel("Decline from Peak (%)", fontsize=11, fontweight='bold', color='#333333')
    
    # 8. Clean Grid and Spines
    ax.grid(True, which='major', linestyle='--', alpha=0.3, color='#cccccc')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#aaaaaa')
    ax.spines['bottom'].set_color('#aaaaaa')
    
    # 9. Legend Placement
    ax.legend(loc='lower left', frameon=False, fontsize=10, ncol=3)
    
    plt.tight_layout()
    plt.show()

def get_sensitivity_table(master_df, df_rates, sizing_method='equal', thresholds=None):
    if thresholds is None:
        thresholds = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        
    sensitivity_rows = []

    for threshold in thresholds:
        df_results = run_backtest(
            master_df, 
            df_rates, 
            z_threshold=threshold, 
            sizing_method=sizing_method
        )
        
        # Generate metrics for each threshold
        perf_summary = get_performance(df_results, strategy_name=f"Z-Spread ≥ {threshold:.1f}")
        raw_df = perf_summary.data if hasattr(perf_summary, 'data') else perf_summary
        sensitivity_rows.append(raw_df)

    master_sensitivity = pd.concat(sensitivity_rows)

    # Apply the Institutional Navy & Gold Styling
    styled_sensitivity = master_sensitivity.style.set_properties(**{
        'text-align': 'center', 
        'padding': '14px',
        'border': '1px solid #eeeeee',
        'color': '#000000' # Global Black Text for body
    }) \
    .set_table_styles([
        # 1. TOP HEADER: Institutional Navy & Gold
        {'selector': 'th.col_heading', 'props': [
            ('background-color', '#1C2C54'), 
            ('color', 'white'), 
            ('font-weight', 'bold'), 
            ('text-transform', 'uppercase'),
            ('letter-spacing', '1px'),
            ('border-bottom', '3px solid #d4af37'), # Gold Accent
            ('text-align', 'center'),
            ('border', '1px solid #1C2C54')
        ]},
        
        # 2. LEFT INDEX: Clean white with bold labels
        {'selector': 'th.row_heading', 'props': [
            ('text-align', 'left'), 
            ('font-weight', 'bold'),
            ('background-color', '#ffffff'),
            ('color', '#333333'),
            ('padding-right', '20px'),
            ('border-right', '1px solid #dee2e6')
        ]},
        
        # 3. HIGHLIGHT: PnL Bridge (Light Yellow)
        {'selector': 'td.col6, td.col7, td.col8, td.col9', 'props': [
            ('background-color', '#fffdf0')
        ]},
        
        # 4. HIGHLIGHT: Final Results (Light Blue Focus)
        {'selector': 'td.col10, td.col11', 'props': [
            ('background-color', '#f0f7ff'),
            ('font-weight', 'bold'),
            ('color', '#1C2C54')
        ]},
        
        # 5. VERTICAL DIVIDERS (Maintaining Waterfall Structure)
        {'selector': 'td.col5, th.col5', 'props': [('border-right', '2px solid #aaa')]},
        {'selector': 'td.col9, th.col9', 'props': [('border-right', '2px solid #aaa')]}
    ])

    return styled_sensitivity

