import numpy as np
from sksurv.metrics import concordance_index_censored


class C_index_metric():
    def __init__(self):
        super().__init__()
        self.risk_scores = []
        self.censorships =[]
        self.event_times = []
    
    def update(self, risk, censor, event_time):
        self.risk_scores.append(risk.numpy())
        self.censorships.append(censor.numpy())
        self.event_times.append(event_time.numpy())
    
    def compute(self):
        all_risk_scores = np.concatenate([arr.ravel() for arr in self.risk_scores])
        all_censorships = np.concatenate([arr.ravel() for arr in self.censorships])
        all_event_times = np.concatenate([arr.ravel() for arr in self.event_times])
        c_index = concordance_index_censored((1 - all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
        return c_index
    
    def reset(self):
        self.risk_scores = []
        self.censorships =[]
        self.event_times = []