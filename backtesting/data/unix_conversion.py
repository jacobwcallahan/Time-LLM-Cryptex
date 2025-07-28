import pandas as pd

filename = "iar_LLAMA3.1_L12_daily_S_seq168_pred7_p6_s4_v1000.csv"  # Replace with your actual filename
df = pd.read_csv(filename)

# Convert the 'timestamp' column to datetime, then to UNIX time (seconds)
df['timestamp'] = pd.to_datetime(df['timestamp']).astype('int64') // 10**9

# Save the updated DataFrame to a new CSV file (or overwrite the original)
df.to_csv(filename + "_UNIX.csv", index=False)

print("Timestamp column replaced with UNIX time. Saved as " + filename + "_UNIX.csv")
