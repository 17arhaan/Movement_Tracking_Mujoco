import pandas as pd

def load_demonstrations(csv_path):
    """
    Expects columns:
      - qpos_0 ... qpos_{nq-1}
      - ctrl_0 ... ctrl_{nu-1}
    Returns (states: np.ndarray [T×nq], actions: np.ndarray [T×nu]).
    """
    df = pd.read_csv(csv_path)
    state_cols = sorted([c for c in df.columns if c.startswith("qpos_")])
    act_cols   = sorted([c for c in df.columns if c.startswith("ctrl_")])
    states = df[state_cols].values
    actions = df[act_cols].values
    return states, actions
