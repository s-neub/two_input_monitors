import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta
from faker import Faker # <-- Import Faker

def generate_synthetic_portfolio_data(weeks=52, num_holdings=15, output_filename='synthetic_data/synthetic_mlops_financial_data.csv'):
    """
    Generates a synthetic financial dataset for MLOps monitoring.
    
    Uses Faker to create more realistic dimensional data (names, tickers).
    
    The dataset simulates weekly performance of personal financial holdings
    and includes ground truth and predictions from a hypothetical ML model.
    """
    
    print(f"Generating {weeks} weeks of data for {num_holdings} synthetic holdings...")
    
    # --- 1. Initialize Faker ---
    fake = Faker()

    # --- 2. Define Dimensions (Categorical Data) ---
    
    # Define our static, logical categories
    sectors_by_asset = {
        'Equity': ['Technology', 'Healthcare', 'Financials', 'Consumer Staples', 'Real Estate', 'Industrials', 'Energy'],
        'Fixed Income': ['N/A'],
        'Alternative': ['Technology', 'Real Estate', 'N/A'], # e.g., Crypto, REITs
        'Cash': ['N/A']
    }
    
    assets_by_account = {
        'Retirement': ['Equity', 'Fixed Income'],
        'Taxable': ['Equity', 'Fixed Income', 'Alternative', 'Cash'],
        'Crypto Wallet': ['Alternative']
    }

    # Use Faker to create our base accounts
    accounts = [
        {'Account_ID': f'IRA-{fake.numerify(text="####")}', 'Account_Type': 'Retirement'},
        {'Account_ID': f'BRK-{fake.numerify(text="####")}', 'Account_Type': 'Taxable'},
        {'Account_ID': f'CRY-{fake.numerify(text="####")}', 'Account_Type': 'Crypto Wallet'}
    ]

    # --- 3. Generate Master Holdings List ---
    master_holdings = []
    generated_tickers = set() # Ensure unique tickers

    while len(master_holdings) < num_holdings:
        # Generate a unique ticker
        ticker = fake.lexify(text='????', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ')
        if ticker in generated_tickers:
            continue
        generated_tickers.add(ticker)

        # Pick a random account
        account = random.choice(accounts)
        
        # Pick a valid asset class for that account
        asset_class = random.choice(assets_by_account[account['Account_Type']])
        
        # Pick a valid sector for that asset class
        sector = random.choice(sectors_by_asset[asset_class])

        # Generate a fake company name
        holding_name = fake.company()
        
        # For crypto, let's make the name/ticker more thematic
        if account['Account_Type'] == 'Crypto Wallet':
            holding_name = f"{fake.word().capitalize()}Coin"
            ticker = fake.lexify(text='???', letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ') # Crypto uses 3-letter tickers
        
        holding = {
            'Account_ID': account['Account_ID'],
            'Account_Type': account['Account_Type'],
            'Asset_Class': asset_class,
            'Industry_Sector': sector,
            'Ticker': ticker,
            'Holding_Name': holding_name
        }
        master_holdings.append(holding)

    print(f"Created {len(master_holdings)} unique holdings.")

    # Generate weekly timestamps (e.g., 52 Fridays)
    end_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    date_range = pd.to_datetime([end_date - timedelta(weeks=x) for x in range(weeks)][::-1])
    
    # --- 4. Initialize Data Generation ---
    
    generated_data = []
    
    # Store the "previous week's value" to simulate a random walk
    last_market_value = {h['Ticker']: random.uniform(5000, 25000) for h in master_holdings}

    for date in date_range:
        for holding in master_holdings:
            
            # --- 5. Simulate Market Data (Quantitative) ---
            
            # Define volatility and drift by asset class
            if holding['Asset_Class'] == 'Equity':
                drift = 0.0015
                volatility = 0.025
            elif holding['Asset_Class'] == 'Fixed Income':
                drift = 0.0005
                volatility = 0.005
            elif holding['Asset_Class'] == 'Alternative':
                drift = 0.002
                volatility = 0.05 # Higher volatility
            else: # Cash
                drift = 0.0001
                volatility = 0.0005

            # Simulate return
            weekly_return_pct = np.random.normal(loc=drift, scale=volatility)
            
            prev_value = last_market_value[holding['Ticker']]
            market_value = prev_value * (1 + weekly_return_pct)
            weekly_return_usd = market_value - prev_value
            
            last_market_value[holding['Ticker']] = market_value

            # --- 6. Simulate Ground Truth (For ML Model) ---
            actual_performance = 1 if weekly_return_pct > 0 else 0

            # --- 7. Simulate ML Model Predictions ---
            prob_positive = 0.5 
            
            if actual_performance == 1:
                prob_positive = np.random.uniform(0.6, 0.95) 
            else:
                prob_positive = np.random.uniform(0.05, 0.4)

            predicted_probability = np.clip(prob_positive + np.random.normal(0, 0.1), 0.0, 1.0)
            predicted_performance = 1 if predicted_probability > 0.5 else 0

            # --- 8. Append Row ---
            row = {
                'Week_Ending': date,
                'Account_ID': holding['Account_ID'],
                'Account_Type': holding['Account_Type'],
                'Asset_Class': holding['Asset_Class'],
                'Industry_Sector': holding['Industry_Sector'],
                'Ticker': holding['Ticker'],
                'Holding_Name': holding['Holding_Name'], # <-- Added new column
                'Market_Value': round(market_value, 2),
                'Weekly_Return_USD': round(weekly_return_usd, 2),
                'Weekly_Return_Pct': round(weekly_return_pct, 4),
                'Actual_Performance': actual_performance,
                'Predicted_Performance': predicted_performance,
                'Predicted_Probability': round(predicted_probability, 4)
            }
            generated_data.append(row)

    print(f"Generated {len(generated_data)} total rows of data.")

    # Create output directory if it doesn't exist
    import os
    os.makedirs('synthetic_data', exist_ok=True)

    # Save to CSV
    pd.DataFrame(generated_data).to_csv(output_filename, index=False)
   
    print(f"\nSuccessfully saved data to {output_filename}")

    return pd.DataFrame(generated_data)

if __name__ == "__main__":
    # Generate the data
    # You can change these parameters
    portfolio_data = generate_synthetic_portfolio_data(weeks=52, num_holdings=20)
    
    print(f"\n--- Data Head (with new 'Holding_Name' column) ---")
    print(portfolio_data.head())
    
    print(f"\n--- Data Info ---")
    portfolio_data.info()
    
    print(f"\n--- Data Description (Quantitative) ---")
    print(portfolio_data.describe())
    
    print(f"\n--- Data Description (Dimensional) ---")
    # We now have more unique values thanks to Faker
    print(portfolio_data.describe(include=['object']))
 
    
    # --- Example of how to calculate MLOps metrics from this file ---
    print("\n--- Example MLOps Metric Calculation (requires scikit-learn & faker) ---")
    try:
        from sklearn.metrics import confusion_matrix, fbeta_score, roc_auc_score
        
        y_true = portfolio_data['Actual_Performance']
        y_pred = portfolio_data['Predicted_Performance']
        y_score = portfolio_data['Predicted_Probability']
        
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
        
        f2 = fbeta_score(y_true, y_pred, beta=2)
        print(f"F2 Score: {f2:.4f}")
        
        auc = roc_auc_score(y_true, y_score)
        print(f"AUC ROC: {auc:.4f}")
        
    except ImportError:
        print("Install scikit-learn and faker (`pip install scikit-learn faker`) to run this script.")
