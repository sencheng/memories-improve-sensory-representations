import numpy as np

TYPE = ["forming", "testing"][1]
FOLDER = "grid5"

ran = np.load("../results/{}/{}.npz".format(FOLDER, TYPE))["{}_ranges".format(TYPE)]
print(np.mean([len(r) for r in ran]))