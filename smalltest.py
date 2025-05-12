from metasyn import MetaFrame, VarSpec, demo_dataframe

df = demo_dataframe("hospital")
vs = [VarSpec(name="patient_id", unique=True)]
mf = MetaFrame.fit_dataframe(df, var_specs=vs)
mf.synthesize(30000)
