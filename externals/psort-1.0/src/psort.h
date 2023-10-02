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

#ifndef PSORT_H
#define PSORT_H

#include "psort_alltoall.h"
#include "psort_merge.h"
#include "psort_splitters.h"

namespace p_sort {
  using namespace std;

  struct vertexToSort {
    double value;
    long long int globalId;
    long long int order;
  };

  bool comp1(const vertexToSort a, const vertexToSort b) {
    return (b.value > a.value)
           || (a.value == b.value && a.globalId < b.globalId);
  }

  bool oppositeComp1(const vertexToSort a, const vertexToSort b) {
    return (b.value < a.value)
           || (a.value == b.value && a.globalId > b.globalId);
  }

  /*
   SeqSort can be STLSort, STLStableSort
   Split can be MedianSplit, SampleSplit
   Merge can be FlatMerge, TreeMerge, OOPTreeMerge, FunnelMerge2, FunnelMerge4
  */
  template <typename _RandomAccessIter,
            typename _Compare,
            typename _SplitType,
            typename _MergeType>
  void parallel_sort(_RandomAccessIter first,
                     _RandomAccessIter last,
                     _Compare comp,
                     _Compare oppositeComp,
                     long *dist_in,
                     Split<_SplitType> &mysplit,
                     Merge<_MergeType> &mymerge,
                     MPI_Comm comm) {

    typedef typename iterator_traits<_RandomAccessIter>::value_type _ValueType;
    typedef
      typename iterator_traits<_RandomAccessIter>::difference_type _Distance;
    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    MPI_Datatype MPI_valueType, MPI_distanceType;
    MPI_Type_contiguous(sizeof(_ValueType), MPI_CHAR, &MPI_valueType);
    MPI_Type_commit(&MPI_valueType);
    MPI_Type_contiguous(sizeof(_Distance), MPI_CHAR, &MPI_distanceType);
    MPI_Type_commit(&MPI_distanceType);

    //_Distance dist[nproc];
    std::vector<_Distance> dist(nproc);
    for(int i = 0; i < nproc; ++i)
      dist[i] = (_Distance)dist_in[i];

    // Sort the data locally
    __gnu_parallel::sort(first, last, comp);

    if(nproc == 1)
      return;

    // Find splitters
    vector<vector<_Distance>> right_ends(
      nproc + 1, vector<_Distance>(nproc, 0));
    mysplit.split(first, last, dist.data(), comp, right_ends, MPI_valueType,
                  MPI_distanceType, comm);

    // Communicate to destination
    _Distance n_loc = last - first;
    std::vector<_ValueType> trans_data(n_loc);
    //_Distance boundaries[nproc+1];
    std::vector<_Distance> boundaries(nproc + 1);
    alltoall(right_ends, first, last, trans_data.data(), boundaries.data(),
             MPI_valueType, comm);

    mymerge.merge(trans_data.data(), &(*first), boundaries.data(), nproc, comp,
                  oppositeComp);

    MPI_Type_free(&MPI_valueType);
    MPI_Type_free(&MPI_distanceType);

    return;
  }

  template <typename T, typename _Compare>
  void parallel_sort(std::vector<T> &data,
                     _Compare comp,
                     _Compare oppositeComp,
                     long *dist,
                     MPI_Comm comm) {

    MedianSplit mysplit;
    OOPTreeMerge mymerge;

    parallel_sort(data.begin(), data.end(), comp, oppositeComp, dist, mysplit,
                  mymerge, comm);
  }

  template <typename T>
  void parallel_sort(std::vector<T> &data, long *dist, MPI_Comm comm) {

    MedianSplit mysplit;
    OOPTreeMerge mymerge;

    parallel_sort(data.begin(), data.end(), comp1, oppositeComp1, dist, mysplit,
                  mymerge, comm);
  }
} // namespace p_sort

#endif /* PSORT_H */
