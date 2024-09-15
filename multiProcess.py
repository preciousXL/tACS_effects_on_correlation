import numpy as np
import os, sys, time
import matplotlib.pyplot as plt
import numba
import scipy.optimize
import scipy.signal
from scipy.optimize import leastsq, curve_fit
from scipy.stats import pearsonr
import multiprocessing
from scipy.signal import hilbert

@numba.njit
def generateOUNoise_Liu(tvar):
    tau_OU = 5  # 5ms
    OUNoise = np.zeros_like(tvar)
    dt1 = tvar[1] - tvar[0]  # ms
    for i in range(len(OUNoise) - 1):
        OUNoise[i + 1] = OUNoise[i] - (OUNoise[i] / tau_OU) * dt1 + np.sqrt(dt1) * np.sqrt(
            2 / tau_OU) * np.random.randn()
    return OUNoise

def calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma, tvar, Vth=19.0):
    '''峰值时刻作为放电时刻'''
    risingBefore = np.hstack((0, vsoma[1:] - vsoma[:-1])) > 0  # v(t)-v(t-1)>0
    fallingAfter = np.hstack((vsoma[1:] - vsoma[:-1], 0)) < 0  # v(t)-v(t+1)<0
    localMaximum = np.logical_and(fallingAfter, risingBefore)  # 逻辑与，上述两者逻辑与为真代表为局部最大值，是放电峰值可能存在的时刻
    largerThanThresh = vsoma > Vth  # 定义一个远大于放电阈值的电压值
    binarySpikeVector = np.logical_and(localMaximum, largerThanThresh)  # 放电峰值时刻二进制序列
    spikeInds = np.nonzero(binarySpikeVector)
    spikeNumber = np.sum(binarySpikeVector)
    outputSpikeTimes = tvar[spikeInds]
    firingRate = 1e3 * spikeNumber / (tvar[-1] - tvar[0])
    return spikeNumber, firingRate, spikeInds, outputSpikeTimes

def calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0):
    dt = tvar[1] - tvar[0]
    nDtPer1ms = int(1 / dt)
    _, _, spikeIndex, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=Vth)
    SpikeTimeSequence = np.zeros_like(tvar)
    SpikeTimeSequence[spikeIndex] = 1.0
    binarySpikeSequence = SpikeTimeSequence.reshape(-1, nDtPer1ms)  # 按照行reshape
    binarySpikeSequence = np.sum(binarySpikeSequence, axis=1)
    spikeCountSequence = binarySpikeSequence.reshape(-1, int(Tin))
    spikeCountSequence1 = np.sum(spikeCountSequence, axis=1)

    _, _, spikeIndex, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=Vth)
    SpikeTimeSequence = np.zeros_like(tvar)
    SpikeTimeSequence[spikeIndex] = 1.0
    binarySpikeSequence = SpikeTimeSequence.reshape(-1, nDtPer1ms)  # 按照行reshape
    binarySpikeSequence = np.sum(binarySpikeSequence, axis=1)
    spikeCountSequence = binarySpikeSequence.reshape(-1, int(Tin))
    spikeCountSequence2 = np.sum(spikeCountSequence, axis=1)

    outputCorrelation = pearsonr(spikeCountSequence1, spikeCountSequence2)[0]
    return outputCorrelation


def calcTacsSpikePLV(vsoma, Evar, tvar, Vth=19.0):
    '''采用希尔伯特变换, 0°对应了电场波形的峰值, 90°对应了电场下降边缘'''
    analyticSignal = hilbert(Evar)
    tacsInstantaneousPhase = np.angle(analyticSignal)  # 计算结果为弧度值,-π~π
    _, _, spikeIndex, spikeTime = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma, tvar, Vth=Vth)
    spikePhaseRadian = tacsInstantaneousPhase[spikeIndex]
    spikePhaseRadian[spikePhaseRadian < 0] += 2 * np.pi
    pluralPLV = np.mean(np.exp(1j * spikePhaseRadian))
    spikePhaseDegree = np.rad2deg(spikePhaseRadian)
    fieldSpikePLV = np.abs(pluralPLV)

    return fieldSpikePLV, spikePhaseDegree, pluralPLV

@numba.njit
def twoCompModel(params2C, vs, vd, Is=0.0, Id=0.0, E=0.0, dt=0.1):
    # Cs, Cd, Gs, Gd, Gi = params2C['Cs'], params2C['Cd'], params2C['Gs'], params2C['Gd'], params2C['Gj']
    # Gexp, DeltaT, VT   = params2C['Gexp'], params2C['DeltaT'], params2C['VT']
    # # Vr, Vth            = params2C['Vr'], params2C['Vth']
    # Delta, DeltaV      = params2C['Delta'], params2C['DeltaV']
    Cs, Cd, Gs, Gd, Gi, Gexp, DeltaT, VT, Delta, DeltaV = params2C
    vsn = vs + dt * (Gi * (vd - vs - Delta * E) - Gs * vs + Is + Gexp * DeltaT * np.exp((vs - VT) / DeltaT)) / Cs
    vdn = vd + dt * (-Gi * (vd - vs - Delta * E) - Gd * vd + Id) / Cd
    return vsn, vdn

@numba.njit
def runModel(params2C=np.ones(1), dt=0.01, tvar=0., Is=0.0, Id=0.0, Evar=0.0):
    Vr, Vth = params2C[-2], params2C[-1]
    vs, vd = 0, 0
    vsoma, vdend = np.zeros_like(tvar), np.zeros_like(tvar)
    for i, t in enumerate(tvar):
        vs, vd = twoCompModel(params2C[0:-2], vs, vd, Is=Is[i], Id=Id[i], E=Evar[i], dt=dt)
        if vs > Vth:
            vs = Vr
        vsoma[i], vdend[i] = vs, vd

    return vsoma, vdend


def runMultiSimulationForCorrelation_withoutEF(paratimes, parac, paramud, parasigmad):
    params2C = np.load('data/params2C.npy')
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 50  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = 0, 30, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Is = np.zeros_like(tvar)  # A
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12, Id2 * 1e-12  # A
    Evar = AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)

    return ans_c, ans_fr, ans_rou


def runMultiSimulationForCorrelation_withoutEF_AE10_fE10(paratimes, parac, paramud, parasigmad):
    params2C = np.load('data/params2C.npy')
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 40  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = 10, 10, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Is = np.zeros_like(tvar)  # A
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12, Id2 * 1e-12  # A
    Evar = AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)

    return ans_c, ans_fr, ans_rou

def runMultiSimulationForCorrelation_withEF(paratimes, parac, paramud, parasigmad, paraAE, parafE):
    params2C = np.load('data/params2C.npy')
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 30  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = paraAE, parafE, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Is = np.zeros_like(tvar)  # A
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12, Id2 * 1e-12  # A
    Evar = AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)
    if AE == 0:
        Evar = np.sin(2 * np.pi * fE * tvar + phiE)  # 用于计算plv
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    plv1, _, pluralPlv1 = calcTacsSpikePLV(vsoma1, Evar, tvar, Vth=19.0)
    preferPhase1 = (np.angle(pluralPlv1)*180/np.pi + 360) % 360
    plv2, _, pluralPlv2 = calcTacsSpikePLV(vsoma2, Evar, tvar, Vth=19.0)
    preferPhase2 = (np.angle(pluralPlv2) * 180 / np.pi + 360) % 360

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)
    ans_plv = np.sqrt(plv1 * plv2)
    ans_phase = np.sqrt(preferPhase1 * preferPhase2)

    return ans_c, ans_fr, ans_rou, ans_plv, ans_phase

def runMultiSimulation_inputCorrelation(paratimes, paraAE, parafE):
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 40  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = paraAE, parafE, 0
    c, mud, sigmad = 0.3, 14, 35  # pA
    Evar = AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc + Evar  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc + Evar  # pA
    cin    = pearsonr(Id1, Id2)[0]
    # Id1 = np.array(Id1)
    # Id2 = np.array(Id2)
    # meanin = np.sqrt(Id1.mean()*Id2.mean())
    # stdin  = np.sqrt(Id1.std()*Id2.std())

    return cin



def runMultiSimulation_withDendriticSineWave(paratimes, parac, paramud, parasigmad, paraAE, parafE):
    params2C = np.load('data/params2C.npy')
    Cs, Cd, Gs, Gd, Gi, Gexp, DeltaT, VT, Delta, DeltaV, Vr, Vth = params2C
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 30  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = paraAE, parafE, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Evar_sin = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    Is = np.zeros_like(tvar)  # A
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12 + Evar_sin, Id2 * 1e-12 + Evar_sin  # A
    Evar = np.zeros_like(tvar)
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)

    Evar = np.sin(2 * np.pi * fE * tvar + phiE)  # 用于计算plv
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    plv1, _, pluralPlv1 = calcTacsSpikePLV(vsoma1, Evar, tvar, Vth=19.0)
    preferPhase1 = (np.angle(pluralPlv1)*180/np.pi + 360) % 360
    plv2, _, pluralPlv2 = calcTacsSpikePLV(vsoma2, Evar, tvar, Vth=19.0)
    preferPhase2 = (np.angle(pluralPlv2) * 180 / np.pi + 360) % 360

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)
    ans_plv = np.sqrt(plv1 * plv2)
    ans_phase = np.sqrt(preferPhase1 * preferPhase2)
    ans_inputmu = np.sqrt(Id1.mean() * Id2.mean())*1e12
    ans_inputsigma = np.sqrt(Id1.std() * Id2.std())*1e12

    return ans_c, ans_fr, ans_rou, ans_plv, ans_phase, ans_inputmu, ans_inputsigma


def runMultiSimulation_withSomaticSineWave(paratimes, parac, paramud, parasigmad, paraAE, parafE):
    params2C = np.load('data/params2C.npy')
    Cs, Cd, Gs, Gd, Gi, Gexp, DeltaT, VT, Delta, DeltaV, Vr, Vth = params2C
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 40  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = paraAE, parafE, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Is = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar + phiE)
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12, Id2 * 1e-12  # A
    Evar = np.zeros_like(tvar)
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)

    Evar = np.sin(2 * np.pi * fE * tvar + phiE)  # 用于计算plv
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    plv1, _, pluralPlv1 = calcTacsSpikePLV(vsoma1, Evar, tvar, Vth=19.0)
    preferPhase1 = (np.angle(pluralPlv1)*180/np.pi + 360) % 360
    plv2, _, pluralPlv2 = calcTacsSpikePLV(vsoma2, Evar, tvar, Vth=19.0)
    preferPhase2 = (np.angle(pluralPlv2) * 180 / np.pi + 360) % 360

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)
    ans_plv = np.sqrt(plv1 * plv2)
    ans_phase = np.sqrt(preferPhase1 * preferPhase2)

    return ans_c, ans_fr, ans_rou, ans_plv, ans_phase


def runMultiSimulation_withDendriticAndSomaticSineWave(paratimes, parac, paramud, parasigmad, paraAE, parafE):
    params2C = np.load('data/params2C.npy')
    Cs, Cd, Gs, Gd, Gi, Gexp, DeltaT, VT, Delta, DeltaV, Vr, Vth = params2C
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 40  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = paraAE, parafE, 0
    c, mud, sigmad = parac, paramud, parasigmad  # pA
    Evar_sin = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar + phiE)  # V/m
    Is = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar + phiE + np.pi)
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12 + Evar_sin, Id2 * 1e-12 + Evar_sin  # A
    Evar = np.zeros_like(tvar)
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)

    Evar = np.sin(2 * np.pi * fE * tvar + phiE)  # 用于计算plv
    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)
    plv1, _, pluralPlv1 = calcTacsSpikePLV(vsoma1, Evar, tvar, Vth=19.0)
    preferPhase1 = (np.angle(pluralPlv1)*180/np.pi + 360) % 360
    plv2, _, pluralPlv2 = calcTacsSpikePLV(vsoma2, Evar, tvar, Vth=19.0)
    preferPhase2 = (np.angle(pluralPlv2) * 180 / np.pi + 360) % 360

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)
    ans_plv = np.sqrt(plv1 * plv2)
    ans_phase = np.sqrt(preferPhase1 * preferPhase2)
    ans_inputmu = np.sqrt(Id1.mean() * Id2.mean())*1e12
    ans_inputsigma = np.sqrt(Id1.std() * Id2.std())*1e12

    return ans_c, ans_fr, ans_rou, ans_plv, ans_phase, ans_inputmu, ans_inputsigma


def runMultiSimulation_varyingPhiBetweenSinewave(paratimes, parafE, paraphi):
    params2C = np.load('data/params2C.npy')
    Cs, Cd, Gs, Gd, Gi, Gexp, DeltaT, VT, Delta, DeltaV, Vr, Vth = params2C
    temptime = paratimes
    dt = 0.01 / 1e3  # second
    duration = 35  # second
    tvar = np.arange(0, duration, dt)  # second
    AE, fE, phiE = 20, parafE, paraphi
    c, mud, sigmad = 0.3, 8, 55  # pA
    Idsin = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar)  # V/m
    Is    = Gi * Delta * AE * np.sin(2 * np.pi * fE * tvar + phiE)
    Idc = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(c) + mud  # pA
    Id1 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id2 = generateOUNoise_Liu(tvar * 1e3) * sigmad * np.sqrt(1 - c) + Idc  # pA
    Id1, Id2 = Id1 * 1e-12 + Idsin, Id2 * 1e-12 + Idsin  # A
    Evar = np.zeros_like(tvar)
    vsoma1, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id1, Evar=Evar)
    vsoma2, _ = runModel(params2C=params2C, dt=dt, tvar=tvar, Is=Is, Id=Id2, Evar=Evar)

    tvar, vsoma1, vsoma2 = tvar * 1e3, vsoma1 * 1e3, vsoma2 * 1e3
    _, fr1, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma1, tvar, Vth=19.0)
    _, fr2, _, _ = calcSpikeNumberAndSpikeTimeAndFiringRate_IfSpike(vsoma2, tvar, Vth=19.0)

    ans_c = pearsonr(Id1, Id2)[0]
    ans_fr = np.sqrt(fr1 * fr2)
    ans_rou = calcSpikeTrainCorrelation(vsoma1, vsoma2, tvar, Tin=10, Vth=19.0)
    ans_inputmu = np.sqrt(Id1.mean() * Id2.mean())*1e12
    ans_inputsigma = np.sqrt(Id1.std() * Id2.std())*1e12

    return ans_c, ans_fr, ans_rou, ans_inputmu, ans_inputsigma




if __name__ == '__main__':
    start_time = time.time()
    casemark = 100
    if casemark == 1:
        '''模型验证 及 输入参数范围确定'''
        listRepetitionTimes = np.arange(0, 30, 1)
        list_c = [0.0, 0.1, 0.2, 0.3]
        list_mu = np.arange(8, 20+0.1, 1)
        list_sigma = sorted(np.arange(15, 55+1, 2).tolist() + [30])  # 为了包含30
        paras = np.array([[i, j, k, m] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withoutEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        results[i][j][k][m] = res[num]
                        num = num + 1
        results = np.array(results)
        np.save('data/times20_c00-03_mu8-20pA_sigma15-55pA.npy', results)


    if casemark == 2:
        listRepetitionTimes = np.arange(0, 30, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = np.arange(0, 20+1, 2.5)
        list_fE = [2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/times30_c03_mu14pA_sigma35pA_AE0-20_fE2-100.npy', results)


    if casemark == 3:
        '''c=0.3的条件下，引入电场'''
        listRepetitionTimes = np.arange(0, 20, 1)
        list_c = [0.3]
        list_mu = np.arange(8, 20+0.1, 1)
        list_sigma = np.arange(15, 55+1, 4)
        paras = np.array([[i, j, k, m] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withoutEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        results[i][j][k][m] = res[num]
                        num = num + 1
        results = np.array(results)
        np.save('data/times20_c03_mu8-20pA_sigma15-55pA_AE10_fE30.npy', results)

    if casemark == 4:
        # spending time = 26373 seconds = 7.32 hours
        listRepetitionTimes = np.arange(0, 10, 1)
        list_c = [0.3]
        list_mu = np.arange(8, 20 + 0.1, 2)
        list_sigma = np.arange(15, 55 + 1, 5)
        list_AE = np.arange(2.5, 20+1, 2.5)
        list_fE = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/times10_c03_mu8-20pA_sigma15-55pA_AE0-20_fE5-100.npy', results)

    if casemark == 5:
        # Figure 3: 引入电场能否改变相关性
        listRepetitionTimes = np.arange(0, 100, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [0, 10]
        list_fE = [10]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/times30_c03_mu14pA_sigma35pA_AE0-10_fE10.npy', results)

    if casemark == 6:
        # Figure 4: 相关性对电场强度 AE 的依赖性
        listRepetitionTimes = np.arange(0, 50, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = np.arange(0, 20+1, 2.5)
        list_fE = [5, 10, 20, 50]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/times50_c03_mu14pA_sigma35pA_AE0-20_fE5-10-20-50.npy', results)

    if casemark == 7:
        # Figure 4: 相关性对电场频率 fE 的依赖性
        listRepetitionTimes = np.arange(0, 50, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [5, 10, 20]
        list_fE = np.hstack((np.arange(2, 60, 2), np.arange(60, 101, 4)))
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/times50_c03_mu14pA_sigma35pA_AE5-10-20_fE2-100.npy', results)


    if casemark == 8:
        # Figure 6: 加入扰动输入对输入相关性的影响
        # run time: 8467 seconds
        listRepetitionTimes = np.arange(0, 30, 1)
        list_AE = np.arange(0, 51, 1)
        list_fE = np.arange(2, 101, 2)
        paras = np.array([[i, j, k] for i in listRepetitionTimes for j in list_AE for k in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_inputCorrelation, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_AE)):
                for k in range(len(list_fE)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times30_c03_mu14pA_sigma35pA_AE0-50_fE2-100_inputcorrelation.npy', results)

    if casemark == 9:
        # 仅仅加入树突正弦输入
        listRepetitionTimes = np.arange(0, 40, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [10]
        list_fE = np.arange(1, 100, 2)
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withDendriticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/onlyDendriticSineWave_times40_c03_mu14pA_sigma35pA_AE10_fE1-100.npy', results)


        list_AE = np.arange(0, 21, 1)
        list_fE = [10]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withDendriticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/onlyDendriticSineWave_times40_c03_mu14pA_sigma35pA_AE0-20_fE10.npy', results)

    if casemark == 10:
        # 仅仅加入胞体正弦输入
        listRepetitionTimes = np.arange(0, 40, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [10]
        list_fE = np.arange(1, 100, 2)
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withSomaticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/onlySomaticSineWave_times40_c03_mu14pA_sigma35pA_AE10_fE1-100.npy', results)

        list_AE = np.arange(0, 21, 1)
        list_fE = [10]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withSomaticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/onlySomaticSineWave_times40_c03_mu14pA_sigma35pA_AE0-20_fE10.npy', results)
        print('Running time:', time.time() - start_time)

    if casemark == 11:
        # 加入胞体正弦输入和树突正弦输入，两者反向
        listRepetitionTimes = np.arange(0, 40, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [10]
        list_fE = np.arange(1, 100, 2)
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withDendriticAndSomaticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/SomaticAndDendriticSineWave_times40_c03_mu14pA_sigma35pA_AE10_fE1-100.npy', results)

        list_AE = np.arange(0, 21, 1)
        list_fE = [10]
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_withDendriticAndSomaticSineWave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/SomaticAndDendriticSineWave_times40_c03_mu14pA_sigma35pA_AE0-20_fE10.npy', results)




    if casemark == 12:
        # 引入电场而非正弦刺激
        listRepetitionTimes = np.arange(0, 20, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = np.arange(0, 31, 2.5)
        list_fE = np.arange(5, 101, 10)
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma), len(list_AE), len(list_fE) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        for ii in range(len(list_AE)):
                            for jj in range(len(list_fE)):
                                results[i][j][k][m][ii][jj] = res[num]
                                num = num + 1
        results = np.array(results)
        np.save('data/EfieldStimulation_times20_c03_mu14pA_sigma35pA_AE0-30_fE5-10-100.npy', results)

    if casemark == 13:
        # 探究不同相位差对共振的影响
        # Is项加phi
        listRepetitionTimes = np.arange(0, 40, 1)
        list_fE = np.hstack((np.arange(1, 41, 2), np.arange(41, 100+4, 4)))
        list_phi = [0]
        paras = np.array([[i, j, k] for i in listRepetitionTimes for j in list_fE for k in list_phi])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulation_varyingPhiBetweenSinewave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(listRepetitionTimes), len(list_fE), len(list_phi))).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_fE)):
                for k in range(len(list_phi)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times40_c03_mu14pA_sigma35pA_AE20_fE0-100_phi-00pi.npy', results)

    if casemark == 14:
        # 探究不同相位差对共振的影响（同13）
        # 只在电场作用下, 应当和phi=np.pi时结果一致
        listRepetitionTimes = np.arange(0, 40, 1)
        list_c = [0.3]
        list_mu = [14]
        list_sigma = [35]
        list_AE = [20]
        list_fE = np.hstack((np.arange(1, 41, 2), np.arange(41, 100+4, 4)))
        paras = np.array([[i, j, k, m, ii, jj] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma for ii in list_AE for jj in list_fE])
        pool = multiprocessing.Pool(processes=6)
        res = pool.starmap(runMultiSimulationForCorrelation_withEF, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(listRepetitionTimes), len(list_fE))).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_fE)):
                results[i][j] = res[num]
                num = num + 1
        results = np.array(results)
        np.save('data/times40_c03_mu14pA_sigma35pA_AE20_fE0-100_Evar.npy', results)

    if casemark == 15:
        # 探究不同相位差对共振的影响,mu=14, sigma=35, AE=20
        # 探究不同相位差对共振的影响,mu=20, sigma=15, AE=20
        # 探究不同相位差对共振的影响,mu=8, sigma=55, AE=20
        # 1212s
        listRepetitionTimes = np.arange(0, 40, 1)
        list_fE = np.hstack((np.arange(1, 41, 2), np.arange(41, 100+4, 4)))
        list_phi = np.arange(0, np.pi+0.01, np.pi/4)
        paras = np.array([[i, j, k] for i in listRepetitionTimes for j in list_fE for k in list_phi])
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(runMultiSimulation_varyingPhiBetweenSinewave, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros((len(listRepetitionTimes), len(list_fE), len(list_phi))).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_fE)):
                for k in range(len(list_phi)):
                    results[i][j][k] = res[num]
                    num = num + 1
        results = np.array(results)
        np.save('data/times30_c03_mu8pA_sigma55pA_AE20_fE0-100_phi0-pi.npy', results)


    if casemark == 16:
        '''在10V/m，10Hz电场作用下模型验证'''
        listRepetitionTimes = np.arange(0, 30, 1)
        list_c = [0.0, 0.1, 0.2, 0.3]
        list_mu = np.arange(8, 20+0.1, 1)
        list_sigma = np.arange(15, 55+1, 2)
        paras = np.array([[i, j, k, m] for i in listRepetitionTimes for j in list_c for k in list_mu for m in list_sigma])
        pool = multiprocessing.Pool(processes=5)
        res = pool.starmap(runMultiSimulationForCorrelation_withoutEF_AE10_fE10, paras)
        pool.close()
        pool.join()
        num = 0
        results = np.zeros(( len(listRepetitionTimes), len(list_c), len(list_mu), len(list_sigma) )).tolist()
        for i in range(len(listRepetitionTimes)):
            for j in range(len(list_c)):
                for k in range(len(list_mu)):
                    for m in range(len(list_sigma)):
                        results[i][j][k][m] = res[num]
                        num = num + 1
        results = np.array(results)
        np.save('data/times30_c00-03_mu8-20pA_sigma15-55pA_AE10_fE10.npy', results)

    print('Running time:', time.time() - start_time)
