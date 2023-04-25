#include "Determinant.h"

namespace Determinant
{
  int det2(const unsigned int& m, rowtype *c)
  {
    rowtype used = 0xFFFFFFFF;
    for(rowtype mask=1<<(m-1); mask; mask >>= 1)
    {
      rowtype clear = 0;
      rowtype imask = 1;
      for(unsigned int i=0;i<m;i++, imask <<= 1)
      {
        //only uses c[i] to clear the other 1s at col j, if not used before.
        if(used & imask && (c[i] & mask))
        {
          if(clear) c[i] ^= clear;
          else
          {
            used ^= imask;
            clear = c[i];
          }
        }
      }
      if(!clear) return 0;
    }
    return 1;
  }
#if 0
  int det3(const unsigned int& m, rowtype *c)
  {
    rowtype mask = 1<<(m-1);
    for(unsigned int j=0;j<m;j++, mask >>= 1)
    {
      rowtype clear = 0;
      for(unsigned int i=j+1;i<m;i++)
      {
        if(c[i] & mask)
        {
          if(clear) c[i] ^= clear;
          else
          {
            clear = c[i];
            c[i] = c[j];
          }
        }
      }
      if(!clear) return 0;
    }
    return 1;
  }
#endif

  int det(unsigned int m, rowtype *c)
  {
    rowtype used = 0;
    for(rowtype mask=1<<(m-1); mask; mask >>= 1)
    {
      rowtype clear = 0;
      for(unsigned int i=0;i<m;i++)
      {
        //only uses c[i] to clear the other 1s at col j, if not used before.
        if((c[i] & mask) && !(used & (1<<i)))
        {
          clear = c[i];
          used |= 1<<i;
          for(i++;i<m;i++)
          {
            if(c[i] & mask) c[i] ^= clear;
          }
        }
        //printMatrix(m, c);
      }
      if(!clear) return 0;
    }
    return 1;
  }
}

