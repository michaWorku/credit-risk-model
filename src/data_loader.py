from pathlib import Path
import pandas as pd

def load_data(file_path: Path, file_type: str = 'csv', delimiter: str = ',') -> pd.DataFrame:
    """
    Loads data from a specified file path into a pandas DataFrame.

    Parameters:
    file_path (Path): The path to the data file.
    file_type (str): The type of the file ('csv', 'txt', 'excel'). Defaults to 'csv'.
    delimiter (str): The delimiter for CSV or TXT files. Defaults to ','.

    Returns:
    pd.DataFrame: The loaded DataFrame, or an empty DataFrame if an error occurs.
    """
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return pd.DataFrame()

    try:
        if file_type == 'csv':
            df = pd.read_csv(file_path, delimiter=delimiter)
        elif file_type == 'txt':
            df = pd.read_csv(file_path, delimiter=delimiter)
        elif file_type == 'excel':
            df = pd.read_excel(file_path)
        else:
            print(f"Error: Unsupported file type '{file_type}'. Supported types are 'csv', 'txt', 'excel'.")
            return pd.DataFrame()
        print(f"Successfully loaded data from {file_path}")
        return df
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

if __name__ == "__main__":
    # Example usage:
    # Assuming data.csv is in credit-risk-model/data/raw/
    raw_data_dir = Path(__file__).parent.parent / "data" / "raw"
    data_file_path = raw_data_dir / "data.csv"

#     # Create a dummy data.csv for testing if it doesn't exist
#     if not data_file_path.is_file():
#         print(f"Creating dummy data.csv at {data_file_path} for demonstration.")
#         data_file_path.parent.mkdir(parents=True, exist_ok=True)
#         dummy_data = """TransactionId,BatchId,AccountId,SubscriptionId,CustomerId,Currency Code,CountryCode,ProviderId,ProductId,ProductCategory,ChannelId,Amount,Value,TransactionStartTime,PricingStrategy,FraudResult
# T1,B1,A1,S1,C1,KES,254,P1,P_A,Electronics,Web,100.0,100.0,2023-01-01 10:00:00,Strategy A,0
# T2,B1,A1,S1,C1,KES,254,P2,P_B,Apparel,Android,150.5,150.5,2023-01-05 11:30:00,Strategy B,0
# T3,B2,A2,S2,C2,UGX,256,P1,P_C,Books,IOS,50.0,50.0,2023-01-10 14:00:00,Strategy A,1
# T4,B2,A3,S3,C3,KES,254,P3,P_D,Electronics,Web,200.0,200.0,2023-02-01 09:15:00,Strategy C,0
# T5,B3,A1,S1,C1,KES,254,P1,P_E,Food,Android,75.2,75.2,2023-02-15 16:45:00,Strategy A,0
# T6,B3,A4,S4,C4,UGX,256,P2,P_F,Apparel,Web,120.0,120.0,2023-03-01 12:00:00,Strategy B,0
# T7,B4,A5,S5,C5,KES,254,P3,P_G,Books,IOS,30.0,30.0,2023-03-05 08:00:00,Strategy C,1
# T8,B4,A1,S1,C1,KES,254,P1,P_H,Electronics,Web,250.0,250.0,2023-03-10 10:00:00,Strategy A,0
# T9,B5,A6,S6,C6,UGX,256,P2,P_I,Food,Android,90.0,90.0,2023-04-01 13:00:00,Strategy B,0
# T10,B5,A7,S7,C7,KES,254,P1,P_J,Apparel,IOS,60.0,60.0,2023-04-05 15:00:00,Strategy A,0
# """
#         with open(data_file_path, 'w') as f:
#             f.write(dummy_data)

    df = load_data(data_file_path)
    if not df.empty:
        print(df.head())
        print(df.info())

    # Example for Xente_Variable_Definitions.csv
    xente_vars_path = raw_data_dir / "Xente_Variable_Definitions.csv"
    # Assuming Xente_Variable_Definitions.csv is also in data/raw/
    # You might want to load this separately if it's used for metadata, not direct analysis.
    # For now, we'll just acknowledge its presence.
    if xente_vars_path.is_file():
        print(f"Found Xente_Variable_Definitions.csv at: {xente_vars_path}")
