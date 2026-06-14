import pandas as pd
import tushare as ts

# Initialize Tushare
ts.set_token('db04790b0214c9122022fbd224d720e9cfa1fdcccb74edd4216f6bca')
pro = ts.pro_api()

def check_if_adjusted(csv_path, ts_code):
    # 1. Load your local data
    df = pd.read_csv(csv_path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    
    # 2. Get the first date and last date of your data
    first_date = df['date'].iloc[0].strftime('%Y%m%d')
    last_date = df['date'].iloc[-1].strftime('%Y%m%d')
    
    # 3. Fetch the adjustment factors from Tushare
    adj_df = pro.adj_factor(ts_code=ts_code, start_date=first_date, end_date=last_date)
    adj_df['trade_date'] = pd.to_datetime(adj_df['trade_date'])
    adj_df = adj_df.sort_values('trade_date')
    
    # 4. Check if the adjustment factor changed
    first_factor = adj_df['adj_factor'].iloc[0]
    last_factor = adj_df['adj_factor'].iloc[-1]
    
    print(f"--- Verification for {ts_code} ---")
    if first_factor == last_factor:
        print("Stock had no splits/dividends in this period. Cannot determine just from price.")
    else:
        print(f"Dividends/Splits occurred! Adj Factor changed from {first_factor} to {last_factor}.")
        
        # Calculate what the Forward Adjusted (QFQ) Open price SHOULD be on day 1
        raw_open_day1 = df['open'].iloc[0]
        expected_qfq_open = raw_open_day1 * (first_factor / last_factor)
        
        print(f"Your CSV Day 1 Open: {raw_open_day1}")
        print(f"Expected QFQ Open  : {expected_qfq_open:.2f}")
        
        if abs(raw_open_day1 - expected_qfq_open) < 0.05:
            print("✅ Conclusion: Your data is ADJUSTED (前复权).")
        else:
            print("❌ Conclusion: Your data is UNADJUSTED (未复权).")

# Run it on your file
check_if_adjusted(r'.\stock_data_all\688600.csv', '688600.SH')