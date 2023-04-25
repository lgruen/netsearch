#ifndef DETERMINANT_H
#define DETERMINANT_H

#include <emmintrin.h>
#include <iostream>
#include "DigitalNets.h"

namespace Determinant {
	typedef unsigned int rowtype;

	int det(unsigned int m, rowtype *c);
	int det2(const unsigned int &m, rowtype *c);

  // works correctly. works on the m least significant bits.
  // _warning_: overwrites the input during computation!
  inline int det3(const unsigned int& m, rowtype *c)
  {
    rowtype mask = 1<<(m-1);
    bool clear;
    for(unsigned int j=0;j<m;j++, mask >>= 1)
    {
      clear = false;
      for(unsigned int i=j;i<m;i++)
      {
        if(c[i] & mask)
        {
          if(clear) c[i] ^= c[j];
          else
          {
            clear = true;
            rowtype tmp = c[j];
            c[j] = c[i];
            c[i] = tmp;
          }
        }
      }
      if(!clear) return 0;
    }
    return 1;
  }

#ifdef __SSE2
	union vInt32 {
		__m128i m[8];
		int i[32];
	};

	// 32x32 fixed size SSE2 version
	inline int det4(const unsigned int * const c) {
		union {
			__m128i m[8];
			int i[32];
		} mc;

		for (unsigned int i = 0; i < 8; i++)  	 	 
			mc.m[i] = _mm_load_si128((__m128i *) c + i);

		for (unsigned int mask = 1UL << 31; mask; mask >>= 1) {
			unsigned int i;
			for (i = 0; i < 32; i++)
				if (mc.i[i] & mask)
					goto found;
			return 0; // nothing found

		found:
			__m128i x = _mm_set1_epi32(mc.i[i]);
			for (i >>= 2; i < 8; i++) { // start at first "block" that isn't cleared yet
				// create mask for those rows that have column bit set
				__m128i rowMask = _mm_cmpeq_epi32(_mm_and_si128(mc.m[i], _mm_set1_epi32(mask)), _mm_setzero_si128());

				// only xor those rows
				mc.m[i] = _mm_xor_si128(mc.m[i], _mm_andnot_si128(rowMask, x));
			}
		}

		return 1;
	}

	// 4x4 fixed size SSE2 version
	inline int detSize4(const unsigned int * const c) {
		union {
			__m128i m;
			int i[4];
		} mc;

		mc.m = _mm_load_si128((__m128i *) c);

		for (unsigned int mask = 1UL << 3; mask; mask >>= 1) {
			unsigned int i;
			for (i = 0; i < 4; i++)
				if (mc.i[i] & mask)
					goto found;
			return 0; // nothing found

		found:
			__m128i x = _mm_set1_epi32(mc.i[i]);
			__m128i rowMask = _mm_cmpeq_epi32(_mm_and_si128(mc.m, _mm_set1_epi32(mask)), _mm_setzero_si128());
			mc.m = _mm_xor_si128(mc.m, _mm_andnot_si128(rowMask, x));
		}

		return 1;
	}
#endif

  /** calculates the sub-determinant on the k most significiant bits. */
  inline int detMostSig(const unsigned int& m, const unsigned int * const cp, const unsigned int k = 32)
  {
    unsigned int c[32];
    memcpy(c, cp, m * sizeof(unsigned int));
    unsigned int mask = 1<<(k-1);
    for(unsigned int j=0;j<m;j++, mask >>= 1)
    {
      for(unsigned int i=j+1;i<m;i++)
      {
        if(c[i] & mask)
        {
          if(c[j] & mask) c[i] ^= c[j];
          else
          {
            rowtype tmp = c[j];
            c[j] = c[i];
            c[i] = tmp;
          }
        }
      }
      if(!(c[j] & mask)) return 0;
    }
    return 1;
  }

}

#endif
