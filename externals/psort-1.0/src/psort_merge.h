/*
Copyright (c) 2009, David Cheng, Viral B. Shah.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef PSORT_MERGE_H
#define PSORT_MERGE_H
#include <vector>

namespace p_sort {
  template <typename MergeType>
  class Merge {
  public:
    template <typename _ValueType, class _Compare, typename _Distance>
    void merge(_ValueType *in,
               _ValueType *out,
               _Distance *disps,
               int nproc,
               _Compare comp,
               _Compare oppositeComp) {

      MergeType *m = static_cast<MergeType *>(this);
      m->real_merge(in, out, disps, nproc, comp, oppositeComp);
    }
  };

  // An out of place tree merge
  class OOPTreeMerge : public Merge<OOPTreeMerge> {
  public:
    template <typename _RandomAccessIter, class _Compare, typename _Distance>
    void real_merge(_RandomAccessIter in,
                    _RandomAccessIter out,
                    _Distance *disps,
                    int nproc,
                    _Compare comp,
                    _Compare oppositeComp) {

      if(nproc == 1) {
        copy(in, in + disps[nproc], out);
        return;
      }

      _RandomAccessIter bufs[2] = {in, out};
      //_Distance locs[nproc];

      std::vector<_Distance> locs(nproc, 0);

      _Distance next = 1;
      while(true) {
        _Distance stride = next * 2;
        if(stride >= nproc)
          break;

        for(_Distance i = 0; i + next < nproc; i += stride) {
          _Distance end_ind = min(i + stride, (_Distance)nproc);

          std::merge(bufs[locs[i]] + disps[i], bufs[locs[i]] + disps[i + next],
                     bufs[locs[i + next]] + disps[i + next],
                     bufs[locs[i + next]] + disps[end_ind],
                     bufs[1 - locs[i]] + disps[i], comp);
          locs[i] = 1 - locs[i];
        }

        next = stride;
      }

      // now have 4 cases for final merge
      if(locs[0] == 0) {
        // 00, 01 => out of place
        std::merge(in, in + disps[next], bufs[locs[next]] + disps[next],
                   bufs[locs[next]] + disps[nproc], out, comp);
      } else if(locs[next] == 0) {
        // 10 => backwards out of place
        std::merge(reverse_iterator<_RandomAccessIter>(in + disps[nproc]),
                   reverse_iterator<_RandomAccessIter>(in + disps[next]),
                   reverse_iterator<_RandomAccessIter>(out + disps[next]),
                   reverse_iterator<_RandomAccessIter>(out),
                   reverse_iterator<_RandomAccessIter>(out + disps[nproc]),
                   oppositeComp);
      } else {
        // 11 => in-place
        std::inplace_merge(out, out + disps[next], out + disps[nproc], comp);
      }
    }
  };
} // namespace p_sort

#endif /*PSORT_MERGE_H */
