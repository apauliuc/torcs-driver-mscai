import numpy as np
from ESN import ESN

data = np.load('mackey_glass_t17.npy')
esn = ESN(n_inputs=1,
          n_outputs=1,
          n_reservoir=500,
          spectral_radius=1.5,
          random_state=42)

trainlen = 2000
future = 2000
pred_training = esn.fit(np.ones(trainlen), data[:trainlen])
