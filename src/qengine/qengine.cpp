//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017, 2018. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#include "qengine.hpp"

namespace Qrack {

/// PSEUDO-QUANTUM - Acts like a measurement gate, except with a specified forced result.
bool QEngine::ForceM(bitLenInt qubit, bool result, bool doForce)
{
    if (doNormalize) {
        NormalizeState();
    }

    real1 oneChance = Prob(qubit);
    if (!doForce) {
        real1 prob = Rand();
        result = ((prob < oneChance) && (oneChance > ZERO_R1));
    }

    real1 nrmlzr;
    if (result) {
        nrmlzr = oneChance;
    } else {
        nrmlzr = ONE_R1 - oneChance;
    }

    bitCapInt qPower = 1 << qubit;
    ApplyM(qPower, result, GetNonunitaryPhase() / (real1)(std::sqrt(nrmlzr)));

    return result;
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceM(const bitLenInt* bits, const bitLenInt& length, const bool* values)
{
    // Single bit operations are better optimized for this special case:
    if (length == 1) {
        if (values == NULL) {
            if (M(bits[0])) {
                return (1U << bits[0]);
            } else {
                return 0U;
            }
        } else {
            if (ForceM(bits[0], values[0])) {
                return (1U << bits[0]);
            } else {
                return 0U;
            }
        }
    }

    if (doNormalize) {
        NormalizeState();
    }

    bitCapInt i;

    complex phase = GetNonunitaryPhase();

    bitCapInt* qPowers = new bitCapInt[length];
    bitCapInt regMask = 0;
    for (i = 0; i < length; i++) {
        qPowers[i] = 1U << bits[i];
        regMask |= qPowers[i];
    }
    std::sort(qPowers, qPowers + length);

    bitCapInt lengthPower = 1 << length;
    real1 nrmlzr = ONE_R1;
    bitCapInt lcv, result;
    complex nrm;

    if (values != NULL) {
        result = 0U;
        for (bitLenInt j = 0; j < length; j++) {
            result |= values[j] ? (1U << bits[j]) : 0U;
        }
        nrmlzr = ProbMask(regMask, result);
        nrm = phase / (real1)(std::sqrt(nrmlzr));
        ApplyM(regMask, result, nrm);

        // No need to check against probabilities:
        return result;
    }

    real1 prob = Rand();
    real1* probArray = new real1[lengthPower]();

    ProbMaskAll(regMask, probArray);

    lcv = 0;
    real1 lowerProb = ZERO_R1;
    real1 largestProb = ZERO_R1;
    result = lengthPower - 1;

    /*
     * The value of 'lcv' should not exceed lengthPower unless the stateVec is
     * in a bug-induced topology - some value in stateVec must always be a
     * vector.
     */
    while ((lowerProb < prob) && (lcv < lengthPower)) {
        lowerProb += probArray[lcv];
        if (largestProb <= probArray[lcv]) {
            largestProb = probArray[lcv];
            nrmlzr = largestProb;
            result = lcv;
        }
        lcv++;
    }
    if (lcv < lengthPower) {
        lcv--;
        result = lcv;
        nrmlzr = probArray[lcv];
    }

    delete[] probArray;

    i = 0;
    for (bitLenInt p = 0; p < length; p++) {
        if ((1U << p) & result) {
            i |= qPowers[p];
        }
    }
    result = i;

    delete[] qPowers;

    nrm = phase / (real1)(std::sqrt(nrmlzr));

    ApplyM(regMask, result, nrm);

    return result;
}

void QEngine::ApplySingleBit(const complex* mtrx, bool doCalcNorm, bitLenInt qubit)
{
    bitCapInt qPowers[1];
    qPowers[0] = 1 << qubit;
    Apply2x2(0, qPowers[0], mtrx, 1, qPowers, doCalcNorm);
}

void QEngine::ApplyControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (controlLen == 0) {
        ApplySingleBit(mtrx, true, target);
    } else {
        ApplyControlled2x2(controls, controlLen, target, mtrx, false);
        if (doNormalize) {
            UpdateRunningNorm();
        }
    }
}

void QEngine::ApplyAntiControlledSingleBit(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target, const complex* mtrx)
{
    if (controlLen == 0) {
        ApplySingleBit(mtrx, true, target);
    } else {
        ApplyAntiControlled2x2(controls, controlLen, target, mtrx, false);
        if (doNormalize) {
            UpdateRunningNorm();
        }
    }
}

void QEngine::CSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | (1U << qubit1), skipMask | (1U << qubit2), pauliX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(1U << qubit1, 1U << qubit2, pauliX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::CSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | (1U << qubit1), skipMask | (1U << qubit2), sqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCSqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(1U << qubit1, 1U << qubit2, sqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::CISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt skipMask = 0;
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
        skipMask |= qPowersSorted[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(skipMask | (1U << qubit1), skipMask | (1U << qubit2), iSqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::AntiCISqrtSwap(
    const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& qubit1, const bitLenInt& qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 2];
    for (bitLenInt i = 0; i < controlLen; i++) {
        qPowersSorted[i] = 1U << controls[i];
    }
    qPowersSorted[controlLen] = 1U << qubit1;
    qPowersSorted[controlLen + 1] = 1U << qubit2;
    std::sort(qPowersSorted, qPowersSorted + controlLen + 2);
    Apply2x2(1U << qubit1, 1U << qubit2, iSqrtX, 2 + controlLen, qPowersSorted, false);
    delete[] qPowersSorted;
}

void QEngine::ApplyControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex* mtrx, bool doCalcNorm)
{
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    bitCapInt fullMask = 0;
    bitCapInt controlMask;
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
        fullMask |= qPowers[i];
    }
    controlMask = fullMask;
    qPowers[controlLen] = 1 << target;
    fullMask |= qPowers[controlLen];
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(controlMask, fullMask, mtrx, controlLen + 1, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

void QEngine::ApplyAntiControlled2x2(const bitLenInt* controls, const bitLenInt& controlLen, const bitLenInt& target,
    const complex* mtrx, bool doCalcNorm)
{
    bitCapInt* qPowers = new bitCapInt[controlLen + 1];
    bitCapInt* qPowersSorted = new bitCapInt[controlLen + 1];
    for (int i = 0; i < controlLen; i++) {
        qPowers[i] = 1 << controls[i];
    }
    qPowers[controlLen] = 1 << target;
    std::copy(qPowers, qPowers + controlLen + 1, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + controlLen + 1);
    Apply2x2(0, qPowers[controlLen], mtrx, controlLen + 1, qPowersSorted, doCalcNorm);
    delete[] qPowers;
    delete[] qPowersSorted;
}

/// Swap values of two bits in register
void QEngine::Swap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex pauliX[4] = { complex(ZERO_R1, ZERO_R1), complex(ONE_R1, ZERO_R1), complex(ONE_R1, ZERO_R1),
        complex(ZERO_R1, ZERO_R1) };
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << qubit1;
    qPowers[1] = 1 << qubit2;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], pauliX, 2, qPowersSorted, false);
}

/// Square root of swap gate
void QEngine::SqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex sqrtX[4] = { complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2),
        complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2) };
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << qubit1;
    qPowers[1] = 1 << qubit2;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], sqrtX, 2, qPowersSorted, false);
}

/// Inverse of square root of swap gate
void QEngine::ISqrtSwap(bitLenInt qubit1, bitLenInt qubit2)
{
    if (qubit1 == qubit2) {
        return;
    }

    const complex iSqrtX[4] = { complex(ONE_R1 / 2, -ONE_R1 / 2), complex(ONE_R1 / 2, ONE_R1 / 2),
        complex(ONE_R1 / 2, ONE_R1 / 2), complex(ONE_R1 / 2, -ONE_R1 / 2) };
    bitCapInt qPowers[2];
    bitCapInt qPowersSorted[2];
    qPowers[0] = 1 << qubit1;
    qPowers[1] = 1 << qubit2;
    std::copy(qPowers, qPowers + 2, qPowersSorted);
    std::sort(qPowersSorted, qPowersSorted + 2);
    Apply2x2(qPowersSorted[0], qPowersSorted[1], iSqrtX, 2, qPowersSorted, false);
}

void QEngine::ProbRegAll(const bitLenInt& start, const bitLenInt& length, real1* probsArray)
{
    bitCapInt lengthPower = 1U << length;
    for (bitCapInt lcv = 0; lcv < lengthPower; lcv++) {
        probsArray[lcv] = ProbReg(start, length, lcv);
    }
}

void QEngine::ProbMaskAll(const bitCapInt& mask, real1* probsArray)
{
    bitCapInt v = mask; // count the number of bits set in v
    bitCapInt oldV;
    bitLenInt length;
    std::vector<bitCapInt> powersVec;
    for (length = 0; v; length++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
    }

    v = ~mask; // count the number of bits set in v
    bitCapInt power;
    bitLenInt len; // c accumulates the total bits set in v
    std::vector<bitCapInt> skipPowersVec;
    for (len = 0; v; len++) {
        oldV = v;
        v &= v - 1; // clear the least significant bit set
        power = (v ^ oldV) & oldV;
        if (power < mask) {
            skipPowersVec.push_back(power);
        } else {
            v = 0;
        }
    }

    bitCapInt lengthPower = 1 << length;
    bitCapInt lcv;

    bitLenInt p;
    bitCapInt i, iHigh, iLow;
    for (lcv = 0; lcv < lengthPower; lcv++) {
        iHigh = lcv;
        i = 0;
        for (p = 0; p < (skipPowersVec.size()); p++) {
            iLow = iHigh & (skipPowersVec[p] - 1);
            i |= iLow;
            iHigh = (iHigh ^ iLow) << 1;
            if (iHigh == 0) {
                break;
            }
        }
        i |= iHigh;
        probsArray[lcv] = ProbMask(mask, i);
    }
}

/// Measure permutation state of a register
bitCapInt QEngine::ForceMReg(bitLenInt start, bitLenInt length, bitCapInt result, bool doForce)
{
    // Single bit operations are better optimized for this special case:
    if (length == 1U) {
        if (ForceM(start, result & 1U, doForce)) {
            return 1U;
        } else {
            return 0U;
        }
    }

    if (doNormalize) {
        NormalizeState();
    }

    real1 prob = Rand();
    complex phase = GetNonunitaryPhase();
    bitCapInt lengthPower = 1 << length;
    bitCapInt regMask = (lengthPower - 1) << start;
    real1* probArray = new real1[lengthPower]();
    bitCapInt lcv;
    real1 nrmlzr = ONE_R1;

    ProbRegAll(start, length, probArray);

    if (!doForce) {
        lcv = 0;
        real1 lowerProb = ZERO_R1;
        real1 largestProb = ZERO_R1;
        result = lengthPower - 1;

        /*
         * The value of 'lcv' should not exceed lengthPower unless the stateVec is
         * in a bug-induced topology - some value in stateVec must always be a
         * vector.
         */
        while ((lowerProb < prob) && (lcv < lengthPower)) {
            lowerProb += probArray[lcv];
            if (largestProb <= probArray[lcv]) {
                largestProb = probArray[lcv];
                nrmlzr = largestProb;
                result = lcv;
            }
            lcv++;
        }
        if (lcv < lengthPower) {
            lcv--;
            result = lcv;
            nrmlzr = probArray[lcv];
        }
    }

    delete[] probArray;

    bitCapInt resultPtr = result << start;
    complex nrm = phase / (real1)(std::sqrt(nrmlzr));

    ApplyM(regMask, resultPtr, nrm);

    return result;
}

} // namespace Qrack
