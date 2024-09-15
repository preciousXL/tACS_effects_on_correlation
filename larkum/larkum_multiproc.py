import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import neuron
from neuron import h
import numba
import time
from scipy.stats import pearsonr
from scipy.signal import hilbert
from scipy.signal import find_peaks

@numba.njit
def generateOUNoise_Liu(ts):
    # 全部以毫秒 ms为单位， 输入参数t不用乘以1e-3
    ''' 示例
    dt = 0.025
    duration = 200e3
    t = np.arange(0, duration, dt)
    y = σ*generateOUNoise_Liu(t) + μ
    '''
    tau_OU = 5  # 5ms
    OUNoise = np.zeros_like(ts)
    dt1 = ts[1] - ts[0]  # ms
    for i in range(len(OUNoise) - 1):
        OUNoise[i + 1] = OUNoise[i] - (OUNoise[i] / tau_OU) * dt1 + np.sqrt(dt1) * np.sqrt(
            2 / tau_OU) * np.random.randn()
    return OUNoise

@numba.njit
def calc_BS_spikerate(VBSin, timein):
    temp1 = []
    for i, item in enumerate(VBSin):
        if item > 10.:
            temp1.append(i)
    numspike = len(temp1)
    for j in range(1, len(temp1)):
        if temp1[j-1] + 1 == temp1[j]:
            numspike = numspike - 1
    return numspike/(timein/1e3)

@numba.njit
def calc_BS_spikenum(VBSin):
    temp1 = []
    for i, item in enumerate(VBSin):
        if item > 10.:
            temp1.append(i)
    numspike = len(temp1)
    for j in range(1, len(temp1)):
        if temp1[j - 1] + 1 == temp1[j]:
            numspike = numspike - 1
    return numspike

def calc_BS_Spikeindex(vsoma):
    temp1 = []
    for i, item in enumerate(vsoma):
        if item > 10.:
            temp1.append(i)
    numspike = len(temp1)
    tempindex = np.full(numspike, True, dtype=bool)
    for j in range(1, len(temp1)):
        if temp1[j-1] + 1 == temp1[j]:
            numspike = numspike - 1
            tempindex[j] = False
    spikeindex = np.array(temp1)
    spikeindex = spikeindex[tempindex]
    return spikeindex


def calcTacsSpikePLV(vsoma, Evar):
    '''采用希尔伯特变换, 0°对应了电场波形的峰值, 90°对应了电场下降边缘'''
    analyticSignal = hilbert(Evar)
    tacsInstantaneousPhase = np.angle(analyticSignal)  # 计算结果为弧度值,-π~π
    spikeIndex = calc_BS_Spikeindex(vsoma)
    spikePhaseRadian = tacsInstantaneousPhase[spikeIndex]
    spikePhaseRadian[spikePhaseRadian < 0] += 2 * np.pi
    pluralPLV = np.mean(np.exp(1j * spikePhaseRadian))
    spikePhaseDegree = np.rad2deg(spikePhaseRadian)
    fieldSpikePLV = np.abs(pluralPLV)

    return fieldSpikePLV, spikePhaseDegree, pluralPLV

# @numba.njit
def calc_spike_corr(Vin1, Vin2, duration_in, T_in, dt_in):
    # usually T_in = 10 ms
    index1 = [i * int(1 / dt_in) for i in range(int(duration_in) + 1)]  # +1为了包含最后一个索引，即包含最后 1ms 的数据
    list_y1 = []
    list_y2 = []
    for j in range(len(index1) - 1):
        if calc_BS_spikenum(Vin1[index1[j]:index1[j + 1]]) > 0:
            list_y1.append(1)
        else:
            list_y1.append(0)
        if calc_BS_spikenum(Vin2[index1[j]:index1[j + 1]]) > 0:
            list_y2.append(1)
        else:
            list_y2.append(0)

    index2 = [ii * int(T_in) for ii in range(int(duration_in / T_in) + 1)]
    list_n1 = []
    list_n2 = []
    for jj in range(len(index2) - 1):
        list_n1.append(np.sum(list_y1[index2[jj]:index2[jj + 1]]))
        list_n2.append(np.sum(list_y2[index2[jj]:index2[jj + 1]]))
    ans_in = pearsonr(np.array(list_n1), np.array(list_n2))[0]
    return ans_in


class PYModel():
    def __init__(self):
        self.init_para()
        self.cell_morphology()

    def init_para(self):
        self.V0 = 0.
        self.VT = 10.
        self.T_ref = 1.5

    def recordVector(self):
        self.tVec = h.Vector().record(h._ref_t)
        self.v0Vec = h.Vector().record(h.soma(0.5)._ref_v)
        self.IdstimVec = h.Vector().record(self.Idstim._ref_i)

    def stimulusadd(self):
        self.Idstim = h.IClamp(h.apic[63](0.5))
        # self.Idstim = h.IClamp(h.dend[25](0.5))
        self.Idstim.delay = 0.
        self.Idstim.dur = 1e9
        self.Idstim.amp = 0.

    def cell_morphology(self):
        h.load_file("nrngui.hoc")
        h.load_file("import3d.hoc")
        h.load_file("apical_simulation.hoc")
        EfieldFilename = "getes.hoc"
        h.load_file(EfieldFilename)
        self.stimulusadd()
        self.recordVector()
        h.finitialize(self.V0)
        h.fcurrent()
        for s in h.allsec():
            for seg in s:
                currents = 0
                # print(seg.pas.e, seg.v, seg.pas.g)
                seg.pas.e = seg.v / seg.pas.g
        '''introduce spiking mechanism (IF firing)'''
        self.spikeout = h.SpikeOut(h.soma(0.5))
        self.spikeout.vrefrac = self.V0
        self.spikeout.refrac = self.T_ref
        self.spikeout.thresh = self.VT


cell = PYModel()

dt = 0.025
duration = 1e3
tvar = np.arange(0, duration, dt)
Idvar = np.zeros_like(tvar)
Evar = np.zeros_like(tvar)


def runBSmodel_py(duration=duration, tvar=tvar, Idvar=Idvar, Evar=Evar, dt=0.025, theta=90, phi=90):
    # Initialize the IdStim
    h.calcesE(theta, phi)
    h.dt = dt
    h.tstop = duration
    h.setstim_snowp()
    h.stim_amp.from_python(Evar)
    h.stim_time.from_python(tvar)
    h.attach_stim()

    tvar_hoc = h.Vector().from_python(tvar)
    Idvar_hoc = h.Vector().from_python(Idvar)
    Idvar_hoc.play(cell.Idstim._ref_amp, tvar_hoc, True)

    for sec in h.allsec():
        for seg in sec.allseg():
            seg.v = cell.V0
    neuron.init()
    neuron.run(duration)

    t = cell.tVec.as_numpy().copy()
    V_BS = cell.v0Vec.as_numpy().copy()
    Id = cell.IdstimVec.as_numpy().copy()
    return t, V_BS, Id

def func1(paratimes, paraAE, parafE):
    temp = paratimes
    dt = 0.025
    duration = 21e3
    tvar = np.arange(0, duration, dt)
    AE, fE, phiE = paraAE, parafE, 0
    mu, sigma, c = 0.3, 0.2, 0.3

    Idc = generateOUNoise_Liu(tvar) * sigma * np.sqrt(c) + mu
    Id1 = generateOUNoise_Liu(tvar) * sigma * np.sqrt(1 - c) + Idc
    Id2 = generateOUNoise_Liu(tvar) * sigma * np.sqrt(1 - c) + Idc
    Evar = AE * np.sin(2 * np.pi * fE * tvar / 1e3 + phiE)  # V/m

    Idvar = Id1
    t, Vsoma1, _ = runBSmodel_py(duration=duration, tvar=tvar, Idvar=Idvar, Evar=Evar, dt=dt, theta=90, phi=90)
    Idvar = Id2
    t, Vsoma2, _ = runBSmodel_py(duration=duration, tvar=tvar, Idvar=Idvar, Evar=Evar, dt=dt, theta=90, phi=90)

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(calc_BS_spikerate(Vsoma1, duration) * calc_BS_spikerate(Vsoma2, duration))
    ans_rou = calc_spike_corr(Vsoma1, Vsoma2, duration, 10, dt)

    if AE == 0:
        Evar = np.sin(2 * np.pi * fE * tvar / 1e3 + phiE)  # 用于计算plv
    plv1, _, pluralPlv1 = calcTacsSpikePLV(Vsoma1, Evar)
    preferPhase1 = (np.angle(pluralPlv1)*180/np.pi + 360) % 360
    plv2, _, pluralPlv2 = calcTacsSpikePLV(Vsoma2, Evar)
    preferPhase2 = (np.angle(pluralPlv2) * 180 / np.pi + 360) % 360
    ans_plv = np.sqrt(plv1 * plv2)
    ans_phase = np.sqrt(preferPhase1 * preferPhase2)

    return ans_c, ans_fr, ans_rou, ans_plv, ans_phase


def calc_somatic_polarization_amplitude(paraAE, parafE):
    if paraAE == 0.:
        somaAp = 0.
    else:
        dt = 0.025
        if parafE <= 15.0:
            duration = int(15 * 1000/parafE)
        else:
            duration = 1e3
        tvar = np.arange(0, duration, dt)
        AE, fE, phiE = paraAE, parafE, 0
        Idvar = np.zeros_like(tvar)
        Evar = AE * np.sin(2 * np.pi * fE * tvar / 1e3 + phiE)
        t, vsoma, _ = runBSmodel_py(duration=duration, tvar=tvar, Idvar=Idvar, Evar=Evar, dt=dt, theta=90, phi=90)
        lastPeakIndex = find_peaks(vsoma)[0][-1]
        somaAp = vsoma[lastPeakIndex]

    return somaAp



if __name__ == "__main__":
    '''
    多间室模型下不同突触位置输入对相关性的影响：
    (1) position,  upper,   red, apic[63](0.5)
    (1) position,  lower, green, dend[25](0.5)
    '''
    start_time = time.time()
    casemark = 100
    if casemark == 1:
        # position,  upper,  red, apic[63](0.5)
        # simulation time is 17.49 hours
        list_times = np.arange(0, 20, 1)
        list_AE = [0, 2.5, 5, 7.5, 10]
        list_fE = [10]
        paras = np.array([[i, j, k] for i in list_times for j in list_AE for k in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(func1, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_times), len(list_AE), len(list_fE))).tolist()
        for i in range(len(list_times)):
            for j in range(len(list_AE)):
                for k in range(len(list_fE)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times20_c03_mu300_sigma200_AE0-10_fE10_apic63.npy', results)

    if casemark == 2:
        # position,  upper,  red, apic[63](0.5)
        # simulation time is 8.83 hours: list_fE = [1, 5, 10, 15, 20, 25]
        # simulation time is 6.14 hours: list_fE = [30, 40, 50, 60]
        # simulation time is 6.14 hours: list_fE = [70, 80, 90, 100]
        # simulation time is 10.36 hours: list_fE = [23, 28, 34, 37, 44, 47, 55]
        list_times = np.arange(0, 10, 1)
        list_AE = [10]
        list_fE = [23, 28, 34, 37, 44, 47, 55]
        paras = np.array([[i, j, k] for i in list_times for j in list_AE for k in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(func1, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_times), len(list_AE), len(list_fE))).tolist()
        for i in range(len(list_times)):
            for j in range(len(list_AE)):
                for k in range(len(list_fE)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times10_c03_mu300_sigma200_AE10_fE23-28-34-37-44-47-55_apic63.npy', results)

    if casemark == 3:
        # position,  lower,  blue, dend[25](0.5)
        # simulation time is ** hours: list_fE = [1, 5, 10, 15, 20, 25, 30, 40, 50]
        list_times = np.arange(0, 10, 1)
        list_AE = [10]
        list_fE = [1, 5, 10, 15, 20, 25, 30, 40, 50]
        paras = np.array([[i, j, k] for i in list_times for j in list_AE for k in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(func1, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_times), len(list_AE), len(list_fE))).tolist()
        for i in range(len(list_times)):
            for j in range(len(list_AE)):
                for k in range(len(list_fE)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times10_c03_mu300_sigma200_AE10_fE1-50_dend25.npy', results)

    casemark = 4
    if casemark == 4:
        '''胞体极化幅值随电场强度和电场频率的变化'''
        list_AE = np.arange(0, 10.1, 0.5)
        list_fE = [10]
        paras = np.array([[i, j] for i in list_AE for j in list_fE])
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_somatic_polarization_amplitude, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_AE), len(list_fE))).tolist()
        for i in range(len(list_AE)):
            for j in range(len(list_fE)):
                results[i][j] = res[num]
                num = num + 1
        results = np.array(results)
        np.save('data/somaAp_mu300_sigma200_AE0-10_fE10_dend63.npy', results)

        list_AE = [10]
        list_fE = np.arange(1, 100, 2)
        paras = np.array([[i, j] for i in list_AE for j in list_fE])
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(calc_somatic_polarization_amplitude, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(list_AE), len(list_fE))).tolist()
        for i in range(len(list_AE)):
            for j in range(len(list_fE)):
                results[i][j] = res[num]
                num = num + 1
        results = np.array(results)
        np.save('data/somaAp_mu300_sigma200_AE10_fE1-100_dend63.npy', results)



    end_time = time.time()
    print(end_time - start_time)