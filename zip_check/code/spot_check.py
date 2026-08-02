import pandas as pd

df = pd.read_csv('dataset/output.csv')
msgs = pd.read_csv('dataset/messages.csv')
merged = df.merge(msgs[['message_id', 'message_text', 'media_type']], on='message_id', how='left')

sample = merged.sample(15, random_state=42)
print("\n==========================================================================")
print("                   RANDOM 15-ROW PREDICTION SPOT CHECK                    ")
print("==========================================================================\n")
for _, r in sample.iterrows():
    print(f"ID: {r.message_id:<8} | Action: {r.action:<6} | Type: {r.message_type:<15} | Conf: {r.confidence}")
    print(f"Snippet: \"{str(r.message_text)[:90]}...\"")
    print(f"Reason:  {r.reason}")
    print("-" * 74)
