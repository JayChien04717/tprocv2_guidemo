import matplotlib.pyplot as plt
from addict import Dict

def utility(keyword):
    if keyword=='spectrum':
        return 
    elif keyword=='amprabi':
        pass
    elif keyword=='lenrabi':
        pass
    elif keyword=='T1':
        pass
    elif keyword=='T2r':
        pass
    elif keyword=='T2e':
        pass



def plotall(data, fitdata):
    plt.subplot(311)
    plt.plot(data.x, data.mag, label = 'mag', marker='o', markersize=3)
    plt.plot(fitdata.x, fitdata.mag, label = 'mag', marker='o', markersize=3)
    plt.legend()

    plt.subplot(312)
    plt.plot(data.x, data.idata, label = 'idata', marker='o', markersize=3)
    plt.plot(fitdata.x, fitdata.idata, label = 'idata', marker='o', markersize=3)
    plt.legend()

    plt.subplot(313)
    plt.plot(data.x, data.qdata, label = 'qdata', marker='o', markersize=3)
    plt.plot(fitdata.x, fitdata.qdata, label = 'qdata', marker='o', markersize=3)
    plt.xlabel('Frequency(GHz)',fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotmeas(data):
    # plt.figure(figsize=(8,6))
    plt.subplot(311)
    plt.title(f'mag.',fontsize=15)
    plt.plot(data.x, data.mag, label = 'mag', marker='o', markersize=3)
    plt.legend()

    plt.subplot(312)
    plt.title(f'idata',fontsize=15)
    plt.plot(data.x, data.idata, label = 'idata', marker='o', markersize=3)
    plt.legend()

    plt.subplot(313)
    plt.plot(data.x, data.qdata, label = 'qdata', marker='o', markersize=3)
    plt.title(f'qdata',fontsize=15)
    plt.xlabel('Frequency(GHz)',fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()




def plotfit(fitdata):
    plt.figure(figsize=(8,6))
    plt.subplot(311)
    plt.title(f'mag.',fontsize=15)
    plt.plot(fitdata.x, fitdata.mag, label = 'fit')
    plt.legend()

    plt.subplot(312)
    plt.title(f'idata',fontsize=15)
    plt.plot(fitdata.x, fitdata.idata, label = 'fit')
    plt.legend()

    plt.subplot(313)
    plt.plot(fitdata.x, fitdata.qdata, label = 'fit')
    plt.title(f'qdata',fontsize=15)
    plt.xlabel('Frequency(GHz)',fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()