//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// QEngineHybrid is a header-only wrapper that selects between QEngineCPU and QEngineOCL implementations based on a
// qubit threshold for maximum performance.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <thread>

#include "qengine_cpu.hpp"
#include "qengine_opencl.hpp"

namespace Qrack {

#define QENGINGEHYBRID_CALL(method)                                                                                    \
    if (isLocked) {                                                                                                    \
        return QEngineCPU::method;                                                                                     \
    } else {                                                                                                           \
        return QEngineOCL::method;                                                                                     \
    }

#define QENGINGEHYBRID_WAIT_RETURN(method, toRet)                                                                      \
    if (isLocked) {                                                                                                    \
        toRet = QEngineCPU::method;                                                                                    \
    } else {                                                                                                           \
        toRet = QEngineOCL::method;                                                                                    \
    }

#define QENGINGEHYBRID_CALL_VOID(method)                                                                               \
    if (isLocked) {                                                                                                    \
        QEngineCPU::method;                                                                                            \
    } else {                                                                                                           \
        QEngineOCL::method;                                                                                            \
    }

class QEngineHybrid;
typedef std::shared_ptr<QEngineHybrid> QEngineHybridPtr;

class QEngineHybrid : public QEngineOCL, QEngineCPU {
protected:
    bitLenInt minimumOCLBits;
    bool isLocked;

    void Lock()
    {
        if (isLocked) {
            return;
        }

        LockSync(CL_MAP_READ | CL_MAP_WRITE);
        isLocked = true;
    }

    void Unlock()
    {
        if (!isLocked) {
            return;
        }

        UnlockSync();
        isLocked = false;
    }

    void SyncToOther(QInterfacePtr oth)
    {
        QEngineHybridPtr other = std::dynamic_pointer_cast<QEngineHybrid>(oth);

        if (isLocked != other->isLocked) {
            if (isLocked) {
                Unlock();
            } else {
                Lock();
            }
        }
    }

    void AdjustToLengthChange()
    {
        if (qubitCount <= minimumOCLBits) {
            Lock();
        } else {
            Unlock();
        }
    }

public:
    // Only the constructor and compose/decompose methods need special implementations. All other methods in the public
    // interface just switch between QEngineCPU and QEngineOCL
    QEngineHybrid(bitLenInt qBitCount, bitCapInt initState, qrack_rand_gen_ptr rgp = nullptr,
        complex phaseFac = CMPLX_DEFAULT_ARG, bool doNorm = false, bool randomGlobalPhase = true,
        bool useHostMem = false, int devID = -1, bool useHardwareRNG = true, bool ignored = false,
        real1 norm_thresh = REAL1_DEFAULT_ARG, std::vector<bitLenInt> ignored2 = {})
        : QEngine(qBitCount, rgp, doNorm, randomGlobalPhase, useHostMem, useHardwareRNG)
        , QEngineOCL(qBitCount, initState, rgp, phaseFac, doNorm, randomGlobalPhase, useHostMem, devID, useHardwareRNG)
        , minimumOCLBits(5)
        , isLocked(false)
    {
        SetConcurrencyLevel(std::thread::hardware_concurrency());

        if (qubitCount < minimumOCLBits) {
            Lock();
        }
    }

    virtual ~QEngineHybrid()
    {
        Finish();
        Unlock();
    }

    /**
     * \defgroup Externally syncing implementations
     *@{
     */

    virtual bitLenInt Compose(QInterfacePtr toCopy, bool isConsumed = false)
    {
        bitLenInt toRet;
        SyncToOther(toCopy);
        QENGINGEHYBRID_WAIT_RETURN(Compose(toCopy, false), toRet);
        AdjustToLengthChange();
        return toRet;
    }
    virtual bitLenInt Compose(QInterfacePtr toCopy, bitLenInt start, bool isConsumed = false)
    {
        bitLenInt toRet;
        SyncToOther(toCopy);
        QENGINGEHYBRID_WAIT_RETURN(Compose(toCopy, start, isConsumed), toRet);
        AdjustToLengthChange();
        return toRet;
    }
    virtual std::map<QInterfacePtr, bitLenInt> Compose(std::vector<QInterfacePtr> toCopy, bool isConsumed = false)
    {
        return QInterface::Compose(toCopy, isConsumed);
    }
    virtual void Decompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        SyncToOther(dest);
        QENGINGEHYBRID_CALL_VOID(Decompose(start, length, dest));
        AdjustToLengthChange();
    }
    virtual void Dispose(bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL_VOID(Dispose(start, length));
        AdjustToLengthChange();
    }
    virtual bool TryDecompose(bitLenInt start, bitLenInt length, QInterfacePtr dest)
    {
        bool toRet;
        SyncToOther(dest);
        QENGINGEHYBRID_WAIT_RETURN(TryDecompose(start, length, dest), toRet);
        AdjustToLengthChange();
        return toRet;
    }
    virtual bool ApproxCompare(QInterfacePtr toCompare)
    {
        bool toRet;
        SyncToOther(toCompare);
        QENGINGEHYBRID_WAIT_RETURN(ApproxCompare(toCompare), toRet);
        AdjustToLengthChange();
        return toRet;
    }
    virtual QInterfacePtr Clone()
    {
        throw "QEngineHybrid::Clone() not implemented!";
    }

    virtual void ResetStateVec(complex* nStateVec)
    {
        throw "QEngineHybrid::ResetStateVec() not implemented!";
        //QEngineOCL::ResetStateVec(nStateVec);
        //if (isLocked)
        //    QEngineOCL::ResetStateBuffer(QEngineOCL::MakeStateVecBuffer(nStateVec));
        //}
    }

    /** @} */

    /**
     * \defgroup QEngine overrides
     *@{
     */

    virtual StateVectorPtr AllocStateVec(bitCapInt elemCount, bool doForceAlloc = false)
    {
        throw "QEngineHybrid::AllocStateVec not implemented!";
        //QENGINGEHYBRID_CALL(AllocStateVec(elemCount, doForceAlloc));
    }

    virtual void Apply2x2(bitCapInt offset1, bitCapInt offset2, const complex* mtrx, const bitLenInt bitCount,
        const bitCapInt* qPowersSorted, bool doCalcNorm, real1 norm_thresh = REAL1_DEFAULT_ARG)
    {
        QENGINGEHYBRID_CALL(Apply2x2(offset1, offset2, mtrx, bitCount, qPowersSorted, doCalcNorm, norm_thresh));
    }

    virtual void INCDECC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECC(toMod, inOutStart, length, carryIndex));
    }
    virtual void INCDECSC(bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length,
        const bitLenInt& overflowIndex, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECSC(toMod, inOutStart, length, overflowIndex, carryIndex));
    }
    virtual void INCDECSC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECSC(toMod, inOutStart, length, carryIndex));
    }
    virtual void INCDECBCDC(
        bitCapInt toMod, const bitLenInt& inOutStart, const bitLenInt& length, const bitLenInt& carryIndex)
    {
        QENGINGEHYBRID_CALL(INCDECBCDC(toMod, inOutStart, length, carryIndex));
    }

    virtual void IMULModNOut(
        bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(IMULModNOut(toMul, modN, inStart, outStart, length));
    }

    virtual void CIMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CIMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen));
    }

    virtual void FullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        QENGINGEHYBRID_CALL(FullAdd(inputBit1, inputBit2, carryInSumOut, carryOut));
    }

    virtual void IFullAdd(bitLenInt inputBit1, bitLenInt inputBit2, bitLenInt carryInSumOut, bitLenInt carryOut)
    {
        QENGINGEHYBRID_CALL(IFullAdd(inputBit1, inputBit2, carryInSumOut, carryOut));
    }

    virtual void ApplyM(bitCapInt regMask, bool result, complex nrm)
    {
        QENGINGEHYBRID_CALL(ApplyM(regMask, result, nrm));
    }

    virtual void ApplyM(bitCapInt regMask, bitCapInt result, complex nrm)
    {
        QENGINGEHYBRID_CALL(ApplyM(regMask, result, nrm));
    }

    /** @} */

    /**
     * \defgroup QEngineOCL overrides
     *@{
     */
    virtual void Finish() { QEngineOCL::Finish(); }
    virtual bool isFinished() { return QEngineOCL::isFinished(); }

    /** @} */

    /**
     * \defgroup QInterface pure virtuals (not overriden by QEngine)
     *@{
     */
    virtual void SetQuantumState(const complex* inputState) { QENGINGEHYBRID_CALL(SetQuantumState(inputState)); }
    virtual void GetQuantumState(complex* outputState) { QENGINGEHYBRID_CALL(GetQuantumState(outputState)); }
    virtual void GetProbs(real1* outputProbs) { QENGINGEHYBRID_CALL(GetProbs(outputProbs)); }
    virtual complex GetAmplitude(bitCapInt perm) { QENGINGEHYBRID_CALL(GetAmplitude(perm)); }
    virtual void SetAmplitude(bitCapInt perm, complex amp) { QENGINGEHYBRID_CALL(SetAmplitude(perm, amp)); }
    virtual void SetPermutation(bitCapInt perm, complex phaseFac = CMPLX_DEFAULT_ARG)
    {
        QENGINGEHYBRID_CALL(SetPermutation(perm, phaseFac));
    }

    virtual void CSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void CSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CSqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCSqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCSqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void CISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(CISqrtSwap(controls, controlLen, qubit1, qubit2));
    }
    virtual void AntiCISqrtSwap(
        const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
    {
        QENGINGEHYBRID_CALL(AntiCISqrtSwap(controls, controlLen, qubit1, qubit2));
    }

    virtual void INC(bitCapInt toAdd, bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(INC(toAdd, start, length));
    }
    virtual void CINC(
        bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CINC(toAdd, inOutStart, length, controls, controlLen));
    }
    virtual void INCS(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length, bitLenInt overflowIndex)
    {
        QENGINGEHYBRID_CALL(INCS(toAdd, inOutStart, length, overflowIndex));
    }
    virtual void INCBCD(bitCapInt toAdd, bitLenInt inOutStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(INCBCD(toAdd, inOutStart, length));
    }
    virtual void MUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(MUL(toMul, inOutStart, carryStart, length));
    }
    virtual void DIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(DIV(toDiv, inOutStart, carryStart, length));
    }
    virtual void MULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(MULModNOut(toMul, modN, inStart, outStart, length));
    }
    virtual void POWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(POWModNOut(base, modN, inStart, outStart, length));
    }
    virtual void CMUL(bitCapInt toMul, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CMUL(toMul, inOutStart, carryStart, length, controls, controlLen));
    }
    virtual void CDIV(bitCapInt toDiv, bitLenInt inOutStart, bitLenInt carryStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CDIV(toDiv, inOutStart, carryStart, length, controls, controlLen));
    }
    virtual void CMULModNOut(bitCapInt toMul, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CMULModNOut(toMul, modN, inStart, outStart, length, controls, controlLen));
    }
    virtual void CPOWModNOut(bitCapInt base, bitCapInt modN, bitLenInt inStart, bitLenInt outStart, bitLenInt length,
        bitLenInt* controls, bitLenInt controlLen)
    {
        QENGINGEHYBRID_CALL(CPOWModNOut(base, modN, inStart, outStart, length, controls, controlLen));
    }

    virtual void ZeroPhaseFlip(bitLenInt start, bitLenInt length) { QENGINGEHYBRID_CALL(ZeroPhaseFlip(start, length)); }
    virtual void CPhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length, bitLenInt flagIndex)
    {
        QENGINGEHYBRID_CALL(CPhaseFlipIfLess(greaterPerm, start, length, flagIndex));
    }
    virtual void PhaseFlipIfLess(bitCapInt greaterPerm, bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(PhaseFlipIfLess(greaterPerm, start, length));
    }
    virtual void PhaseFlip() { QENGINGEHYBRID_CALL(PhaseFlip()); }

    virtual bitCapInt IndexedLDA(
        bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart, bitLenInt valueLength, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedLDA(indexStart, indexLength, valueStart, valueLength, values));
    }
    virtual bitCapInt IndexedADC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedADC(indexStart, indexLength, valueStart, valueLength, carryIndex, values));
    }
    virtual bitCapInt IndexedSBC(bitLenInt indexStart, bitLenInt indexLength, bitLenInt valueStart,
        bitLenInt valueLength, bitLenInt carryIndex, unsigned char* values)
    {
        QENGINGEHYBRID_CALL(IndexedSBC(indexStart, indexLength, valueStart, valueLength, carryIndex, values));
    }

    virtual void UpdateRunningNorm(real1 norm_thresh = REAL1_DEFAULT_ARG) { QENGINGEHYBRID_CALL(UpdateRunningNorm(norm_thresh)); }
    virtual void NormalizeState(real1 nrm = REAL1_DEFAULT_ARG, real1 norm_thresh = REAL1_DEFAULT_ARG) { QENGINGEHYBRID_CALL(NormalizeState(nrm, norm_thresh)); }

    /** @} */

    /**
     * \defgroup QEngineOCL/QEngineCPU default QInterface implementation overrides
     *@{
     */

    virtual void UniformlyControlledSingleBit(const bitLenInt* controls, const bitLenInt& controlLen,
        bitLenInt qubitIndex, const complex* mtrxs, const bitCapInt* mtrxSkipPowers, const bitLenInt mtrxSkipLen,
        const bitCapInt& mtrxSkipValueMask)
    {
        QENGINGEHYBRID_CALL(UniformlyControlledSingleBit(controls, controlLen, qubitIndex, mtrxs, mtrxSkipPowers, mtrxSkipLen, mtrxSkipValueMask));
    }

    virtual void X(bitLenInt start) { QENGINGEHYBRID_CALL(X(start)); }
    virtual void Z(bitLenInt start) { QENGINGEHYBRID_CALL(Z(start)); }

    virtual void ROL(bitLenInt shift, bitLenInt start, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(ROL(shift, start, length));
    }

    virtual void Swap(bitLenInt start1, bitLenInt start2, bitLenInt length)
    {
        QENGINGEHYBRID_CALL(Swap(start1, start2, length));
    }

    virtual real1 Prob(bitLenInt qubitIndex) { QENGINGEHYBRID_CALL(Prob(qubitIndex)); }
    virtual real1 ProbAll(bitCapInt fullRegister) { QENGINGEHYBRID_CALL(ProbAll(fullRegister)); }
    virtual real1 ProbReg(const bitLenInt& start, const bitLenInt& length, const bitCapInt& permutation)
    {
        QENGINGEHYBRID_CALL(ProbReg(start, length, permutation));
    }
    virtual real1 ProbMask(const bitCapInt& mask, const bitCapInt& permutation)
    {
        QENGINGEHYBRID_CALL(ProbMask(mask, permutation));
    }

    /** @} */
};
} // namespace Qrack
