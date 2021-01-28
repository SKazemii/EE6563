import numpy as np
import h5py

with h5py.File(
    "/Users/saeedkazemi/Documents/reproducible-research/data/colourvideo_sil.h5",
    "r",
) as hdf:
    items = list(hdf.items())
    print(items)

    surveydata = hdf.get("/surveydata")
    # items = list(barefoot.items())
    # print(items)

    # w2 = barefoot.get("slow")
    # items = list(w2.items())
    # print(items)

    # data = np.array(surveydata)
    print(surveydata.shape)
    # metadata = np.array(barefoot.get("metadata"))
    print(surveydata)
