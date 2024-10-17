import shioaji as sj
import pandas as pd
api = sj.Shioaji()
api.login(
    api_key="CztRiojJmJKjBBk1Q9wN6X4R8c1jLYABK2KzJExGMKeJ", 
    secret_key="8SEUC86Jo66UFADDTGF7KXUqZ19fiaJeCYgkBUmTFyeP",
    contracts_cb=lambda security_type: print(f"{repr(security_type)} fetch done.")

)


ticks = api.ticks(
    contract=api.Contracts.Stocks["2330"], 
    date="2024-10-08",
    query_type=sj.constant.TicksQueryType.RangeTime,
    time_start="09:00:00",
    time_end="09:30:01"
    
    
)
ticks

df = pd.DataFrame({**ticks})
df.ts = pd.to_datetime(df.ts)


df.to_csv('tsmc_ticks_data.csv', index=False)