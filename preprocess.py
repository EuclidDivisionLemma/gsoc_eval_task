import pandas

real = pandas.read_csv("./datasets/real_events.csv")
virtual = pandas.read_csv("./datasets/virtual_events.csv")

real = pandas.DataFrame(
    {
        "id": real.loc[:, "id"],
        "pt": real.loc[:, ["pt_real", "z_gluon"]].sum(axis=1),
        "y": real.loc[:, "y_real"],
        "weight": real.loc[:, "weight"],
    }
)

combined = pandas.concat((real, virtual), axis=0)
combined.to_csv("./datasets/combined.csv", index=False)
