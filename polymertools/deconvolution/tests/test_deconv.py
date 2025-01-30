from polymertools.deconvolution import MWDDeconv
from pandas import read_excel
import numpy as np


if __name__ == '__main__':

    data = read_excel(r'C:\Users\Admin\Downloads\pratikum1.xls', sheet_name='Data MMD')

    deconv = MWDDeconv(active_sites=6, log_m_range=(2.8, 7))

    log_m, mmd = data.iloc[:,4].to_numpy(), data.iloc[:,5].to_numpy()


    # Remove nan from numpy array
    log_m = log_m[~np.isnan(log_m)]
    mmd = mmd[~np.isnan(mmd)]

    deconv.fit(log_m, mmd)

    deconv.plot_deconvolution()

    #deconv.export_deconvolution("Deconvolution_Group_1_Sample_1.xlsx")

