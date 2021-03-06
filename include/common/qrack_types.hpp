//////////////////////////////////////////////////////////////////////////////////////
//
// (C) Daniel Strano and the Qrack contributors 2017-2019. All rights reserved.
//
// This is a multithreaded, universal quantum register simulation, allowing
// (nonphysical) register cloning and direct measurement of probability and
// phase, to leverage what advantages classical emulation of qubits can have.
//
// Licensed under the GNU Lesser General Public License V3.
// See LICENSE.md in the project root or https://www.gnu.org/licenses/lgpl-3.0.en.html
// for details.

#pragma once

#include <functional>
#include <memory>
#include <random>

#if QBCAPPOW < 8
#define bitLenInt uint8_t
#elif QBCAPPOW < 16
#define bitLenInt uint16_t
#elif QBCAPPOW < 32
#define bitLenInt uint32_t
#else
#define bitLenInt uint64_t
#endif

#if ENABLE_PURE32
#define bitCapIntOcl uint32_t
#define bitCapInt uint32_t
#define ONE_BCI 1U
#elif ENABLE_UINT128
#ifdef BOOST_AVAILABLE
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapIntOcl uint64_t
#define bitCapInt boost::multiprecision::uint128_t
#define ONE_BCI 1ULL
#else
#define bitCapIntOcl uint64_t
#define bitCapInt __uint128_t
#define ONE_BCI 1ULL
#endif
#elif QBCAPPOW > 7
#include <boost/multiprecision/cpp_int.hpp>
#define bitCapIntOcl uint64_t
#define bitCapInt                                                                                                      \
    boost::multiprecision::number<boost::multiprecision::cpp_int_backend<1 << QBCAPPOW, 1 << QBCAPPOW,                 \
        boost::multiprecision::unsigned_magnitude, boost::multiprecision::unchecked, void>>
#define ONE_BCI 1ULL
#else
#define bitCapIntOcl uint64_t
#define bitCapInt uint64_t
#define ONE_BCI 1ULL
#endif

#define bitsInByte 8
#define qrack_rand_gen std::mt19937_64
#define qrack_rand_gen_ptr std::shared_ptr<qrack_rand_gen>
#define QRACK_ALIGN_SIZE 64

#include "config.h"

#include <complex>

#if ENABLE_COMPLEX8
namespace Qrack {
typedef std::complex<float> complex;
typedef float real1;
} // namespace Qrack
#define ZERO_R1 0.0f
#define ONE_R1 1.0f
#define PI_R1 (real1) M_PI
// min_norm is the minimum probability neighborhood to check for exactly 1 or 0 probability. Values were chosen based on
// the results of the tests in accuracy.cpp.
#define min_norm 1e-14f
#define REAL1_DEFAULT_ARG -999.0f
#define REAL1_EPSILON FLT_EPSILON
#else
//#include "complex16simd.hpp"
namespace Qrack {
typedef std::complex<double> complex;
typedef double real1;
} // namespace Qrack
#define ZERO_R1 0.0
#define ONE_R1 1.0
#define PI_R1 M_PI
// min_norm is the minimum probability neighborhood to check for exactly 1 or 0 probability. Values were chosen based on
// the results of the tests in accuracy.cpp.
#define min_norm 1e-30
#define REAL1_DEFAULT_ARG -999.0
#define REAL1_EPSILON DBL_EPSILON
#endif

#define ONE_CMPLX complex(ONE_R1, ZERO_R1)
#define ZERO_CMPLX complex(ZERO_R1, ZERO_R1)
#define I_CMPLX complex(ZERO_R1, ONE_R1)
#define CMPLX_DEFAULT_ARG complex(REAL1_DEFAULT_ARG, REAL1_DEFAULT_ARG)

// approxcompare_error is the maximum acceptable sum of probability amplitude difference for ApproxCompare to return
// "true." When TrySeparate or TryDecohere is applied after the QFT followed by its inverse on a permutation, the sum of
// square errors of probability is generally less than 10^-11, for float accuracy. (A small number of trials return many
// orders larger error, but these cases should not be separated, as the code stands.)
#define approxcompare_error 1e-7f

namespace Qrack {
typedef std::shared_ptr<complex> BitOp;

/** Called once per value between begin and end. */
typedef std::function<void(const bitCapInt, const int cpu)> ParallelFunc;
typedef std::function<bitCapInt(const bitCapInt, const int cpu)> IncrementFunc;

class StateVector;
class StateVectorArray;
class StateVectorSparse;

typedef std::shared_ptr<StateVector> StateVectorPtr;
typedef std::shared_ptr<StateVectorArray> StateVectorArrayPtr;
typedef std::shared_ptr<StateVectorSparse> StateVectorSparsePtr;

// This is a buffer struct that's capable of representing controlled single bit gates and arithmetic, when subclassed.
class StateVector {
protected:
    bitCapInt capacity;

public:
    bool isReadLocked;

    StateVector(bitCapInt cap)
        : capacity(cap)
        , isReadLocked(true)
    {
    }
    virtual complex read(const bitCapInt& i) = 0;
    virtual void write(const bitCapInt& i, const complex& c) = 0;
    /// Optimized "write" that is only guaranteed to write if either amplitude is nonzero. (Useful for the result of 2x2
    /// tensor slicing.)
    virtual void write2(const bitCapInt& i1, const complex& c1, const bitCapInt& i2, const complex& c2) = 0;
    virtual void clear() = 0;
    virtual void copy_in(const complex* inArray) = 0;
    virtual void copy_out(complex* outArray) = 0;
    virtual void copy(StateVectorPtr toCopy) = 0;
    virtual void get_probs(real1* outArray) = 0;
    virtual bool is_sparse() = 0;
};

void mul2x2(complex* left, complex* right, complex* out);
void exp2x2(complex* matrix2x2, complex* outMatrix2x2);
void log2x2(complex* matrix2x2, complex* outMatrix2x2);
} // namespace Qrack
