from polymertools.deconvolution import MWDDeconv
from pandas import read_excel

if __name__ == '__main__':
    deconv = MWDDeconv(active_sites=6)

    data = read_excel('data/experimental_test_data.xlsx', sheet_name='Data MMD')

    log_m = data.iloc[:,0].to_numpy()
    mmd = data.iloc[:,1].to_numpy()

    norm_data = deconv.fit(log_m, mmd)