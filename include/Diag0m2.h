#ifndef DIAG_0M2_H
#define DIAG_0M2_H

class Diag0m2 {
  public:
    Diag0m2(const unsigned int m) {
      n = 1 << m;
      m2 = (m + 1) >> 1;
      mask = ~-(1 << m2);

      const unsigned int sqrtN = 1 << (m >> 1);
      d = new unsigned int[sqrtN];

      if (m & 1) { // odd
        dx = dy = m >> 1;

        for (unsigned int k = 0; k < sqrtN; k++)
          d[k] = (radicalInverseBase2(k) >> (32 - m)) + k;
      }
      else { // even
        dx = (m >> 1) - 2;
        dy = m >> 1;

        for (unsigned int k = 0; k < sqrtN; k++)
          d[k] = (radicalInverseBase2(k) >> (32 - m)) + (k >> 2) + (1 << dx);
      }
    }

    ~Diag0m2() {
      delete [] d;
    }

    inline void operator()(const unsigned int i, unsigned int &x, unsigned int &y) {
      const unsigned int k = i >> m2;
      const unsigned int j = i & mask;

      // multiplication by shift, modulo by and
      x = (d[k] + (j << dx)) & (n - 1);
      y = k + (j << dy);
    }

  private:
    inline static unsigned int radicalInverseBase2(unsigned int bits) { // 32 bits version
      bits = (bits << 16) | (bits >> 16);
      bits = ((bits & 0x00ff00ff) << 8) | ((bits & 0xff00ff00) >> 8);
      bits = ((bits & 0x0f0f0f0f) << 4) | ((bits & 0xf0f0f0f0) >> 4);
      bits = ((bits & 0x33333333) << 2) | ((bits & 0xcccccccc) >> 2);
      bits = ((bits & 0x55555555) << 1) | ((bits & 0xaaaaaaaa) >> 1);
      return bits;
    }

    unsigned int n, m2, mask;
    unsigned int *d;
    unsigned int dx, dy;
};

#endif
