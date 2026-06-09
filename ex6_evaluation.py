import pandas as pd

gs = pd.read_csv("grid_search_results.csv")
rs = pd.read_csv("random_search_results.csv")

print("\n=== TOP GRID SEARCH ===")
print(gs.head(3))

print("\n=== TOP RANDOM SEARCH ===")
print(rs.head(3))

summary = pd.DataFrame({
    "Modele/Config":[
        "Best Grid Search",
        "Best Random Search",
        "2nd Random Search",
        "3rd Random Search"
    ],
    "Val MSE":[
        gs.iloc[0]["val_mse"],
        rs.iloc[0]["val_mse"],
        rs.iloc[1]["val_mse"],
        rs.iloc[2]["val_mse"]
    ]
})

summary.to_csv("final_summary.csv",index=False)

print(summary)