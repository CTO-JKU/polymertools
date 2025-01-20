import numpy as np

from polymertools.deconvolution import MWDDeconv
from pandas import read_excel
import matplotlib.pyplot as plt

if __name__ == '__main__':

    data = read_excel('data/experimental_test_data.xlsx', sheet_name='Data MMD')

    deconv = MWDDeconv(active_sites=6, log_m_range=(2.8, 7))

    log_m = data.iloc[:,0].to_numpy()
    mmd = data.iloc[:,1].to_numpy()

    deconv.fit(log_m, mmd)

    deconv.plot_deconvolution()



