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

#pragma once
#include <Debug.h>

namespace p_sort {

  template <typename T, class _Compare>
  class PermCompare {
  private:
    T *weights;
    _Compare comp;

  public:
    PermCompare(T *w, _Compare c) : weights(w), comp(c) {
    }
    bool operator()(int a, int b) {
      return comp(weights[a], weights[b]);
    }
  };

  template <class _Compare, typename _RandomAccessIter>
  void psort_split(_RandomAccessIter first,
                   _RandomAccessIter last,
                   ttk::SimplexId *dist,
                   _Compare comp,
                   std::vector<std::vector<ttk::SimplexId>> &right_ends,
                   MPI_Datatype &MPI_valueType,
                   MPI_Datatype &MPI_distanceType,
                   int nThreads) {

    typedef
      typename std::iterator_traits<_RandomAccessIter>::value_type dataType;

    int n_real = ttk::MPIsize_;
    for(int i = 0; i < ttk::MPIsize_; ++i)
      if(dist[i] == 0) {
        n_real = i;
        break;
      }

    std::copy(dist, dist + ttk::MPIsize_, right_ends[ttk::MPIsize_].begin());

    // union of [0, right_end[i+1]) on each processor produces dist[i] total
    // values
    // ttk::SimplexId targets[ttk::MPIsize_-1];
    std::vector<ttk::SimplexId> targets(ttk::MPIsize_ - 1);
    std::partial_sum(dist, dist + (ttk::MPIsize_ - 1), targets.data());

    // keep a list of ranges, trying to "activate" them at each branch
    std::vector<std::pair<_RandomAccessIter, _RandomAccessIter>> d_ranges(
      ttk::MPIsize_ - 1);
    std::vector<std::pair<ttk::SimplexId *, ttk::SimplexId *>> t_ranges(
      ttk::MPIsize_ - 1);
    d_ranges[0] = std::make_pair(first, last);
    t_ranges[0]
      = std::make_pair(targets.data(), targets.data() + ttk::MPIsize_ - 1);

    // invariant: subdist[i][ttk::MPIrank_] == d_ranges[i].second -
    // d_ranges[i].first amount of data each proc still has in the search
    std::vector<std::vector<ttk::SimplexId>> subdist(
      ttk::MPIsize_ - 1, std::vector<ttk::SimplexId>(ttk::MPIsize_));
    std::copy(dist, dist + ttk::MPIsize_, subdist[0].begin());

    // for each processor, d_ranges - first
    std::vector<std::vector<ttk::SimplexId>> outleft(
      ttk::MPIsize_ - 1, std::vector<ttk::SimplexId>(ttk::MPIsize_, 0));

    for(int n_act = 1; n_act > 0;) {

      for(int k = 0; k < n_act; ++k) {
        assert(subdist[k][ttk::MPIrank_]
               == d_ranges[k].second - d_ranges[k].first);
      }

      //------- generate n_act guesses

      // for the allgather, make a flat array of ttk::MPIsize_ chunks, each with
      // n_act elts
      std::vector<dataType> medians(ttk::MPIsize_ * n_act);
      for(int k = 0; k < n_act; ++k) {
        if(d_ranges[k].first != last) {
          dataType *ptr = &(*d_ranges[k].first);
          ttk::SimplexId index = subdist[k][ttk::MPIrank_] / 2;
          medians[ttk::MPIrank_ * n_act + k] = ptr[index];
        } else
          medians[ttk::MPIrank_ * n_act + k] = *(last - 1);
      }

      // std::cout << "Median allGather" << std::endl;
      MPI_Allgather(MPI_IN_PLACE, n_act, MPI_valueType, &medians[0], n_act,
                    MPI_valueType, ttk::MPIcomm_);

      // compute the weighted median of medians
      std::vector<dataType> queries(n_act);

      std::vector<ttk::SimplexId> ms_perm(n_real);
      for(int k = 0; k < n_act; ++k) {
        // ttk::SimplexId ms_perm[n_real];

        for(int i = 0; i < n_real; ++i)
          ms_perm[i] = i * n_act + k;
        TTK_PSORT(nThreads, ms_perm.data(), ms_perm.data() + n_real,
                  PermCompare<dataType, _Compare>(&medians[0], comp));

        ttk::SimplexId mid
          = std::accumulate(
              subdist[k].begin(), subdist[k].end(), (ttk::SimplexId)0)
            / 2;
        ttk::SimplexId query_ind = -1;
        for(int i = 0; i < n_real; ++i) {
          if(subdist[k][ms_perm[i] / n_act] == 0)
            continue;

          mid -= subdist[k][ms_perm[i] / n_act];
          if(mid <= 0) {
            query_ind = ms_perm[i];
            break;
          }
        }

        assert(query_ind >= 0);
        queries[k] = medians[query_ind];
      }
      //------- find min and max ranks of the guesses
      // ttk::SimplexId ind_local[2 * n_act];
      std::vector<ttk::SimplexId> ind_local(2 * n_act);

      for(int k = 0; k < n_act; ++k) {
        std::pair<_RandomAccessIter, _RandomAccessIter> ind_local_p
          = equal_range(
            d_ranges[k].first, d_ranges[k].second, queries[k], comp);

        ind_local[2 * k] = ind_local_p.first - first;
        ind_local[2 * k + 1] = ind_local_p.second - first;
      }

      // ttk::SimplexId ind_all[2 * n_act * ttk::MPIsize_];
      std::vector<ttk::SimplexId> ind_all(2 * n_act * ttk::MPIsize_);
      MPI_Allgather(ind_local.data(), 2 * n_act, MPI_distanceType,
                    ind_all.data(), 2 * n_act, MPI_distanceType, ttk::MPIcomm_);
      // sum to get the global range of indices
      std::vector<std::pair<ttk::SimplexId, ttk::SimplexId>> ind_global(n_act);
      for(int k = 0; k < n_act; ++k) {
        ind_global[k] = std::make_pair(0, 0);
        for(int i = 0; i < ttk::MPIsize_; ++i) {
          ind_global[k].first += ind_all[2 * (i * n_act + k)];
          ind_global[k].second += ind_all[2 * (i * n_act + k) + 1];
        }
      }

      // state to pass on to next iteration
      std::vector<std::pair<_RandomAccessIter, _RandomAccessIter>> d_ranges_x(
        ttk::MPIsize_ - 1);
      std::vector<std::pair<ttk::SimplexId *, ttk::SimplexId *>> t_ranges_x(
        ttk::MPIsize_ - 1);
      std::vector<std::vector<ttk::SimplexId>> subdist_x(
        ttk::MPIsize_ - 1, std::vector<ttk::SimplexId>(ttk::MPIsize_));
      std::vector<std::vector<ttk::SimplexId>> outleft_x(
        ttk::MPIsize_ - 1, std::vector<ttk::SimplexId>(ttk::MPIsize_, 0));
      int n_act_x = 0;

      for(int k = 0; k < n_act; ++k) {
        ttk::SimplexId *split_low = std::lower_bound(
          t_ranges[k].first, t_ranges[k].second, ind_global[k].first);
        ttk::SimplexId *split_high = std::upper_bound(
          t_ranges[k].first, t_ranges[k].second, ind_global[k].second);

        // iterate over targets we hit
        for(ttk::SimplexId *s = split_low; s != split_high; ++s) {
          assert(*s > 0);
          // a bit sloppy: if more than one target in range, excess won't zero
          // out
          ttk::SimplexId excess = *s - ind_global[k].first;
          // low procs to high take excess for stability
          for(int i = 0; i < ttk::MPIsize_; ++i) {
            ttk::SimplexId amount
              = std::min(ind_all[2 * (i * n_act + k)] + excess,
                         ind_all[2 * (i * n_act + k) + 1]);
            right_ends[(s - targets.data()) + 1][i] = amount;
            excess -= amount - ind_all[2 * (i * n_act + k)];
          }
        }

        if((split_low - t_ranges[k].first) > 0) {
          t_ranges_x[n_act_x] = std::make_pair(t_ranges[k].first, split_low);
          // lop off local_ind_low..end
          d_ranges_x[n_act_x]
            = std::make_pair(d_ranges[k].first, first + ind_local[2 * k]);
          for(int i = 0; i < ttk::MPIsize_; ++i) {
            subdist_x[n_act_x][i]
              = ind_all[2 * (i * n_act + k)] - outleft[k][i];
            outleft_x[n_act_x][i] = outleft[k][i];
          }
          ++n_act_x;
        }

        if((t_ranges[k].second - split_high) > 0) {
          t_ranges_x[n_act_x] = std::make_pair(split_high, t_ranges[k].second);
          // lop off begin..local_ind_high
          d_ranges_x[n_act_x]
            = std::make_pair(first + ind_local[2 * k + 1], d_ranges[k].second);
          for(int i = 0; i < ttk::MPIsize_; ++i) {
            subdist_x[n_act_x][i] = outleft[k][i] + subdist[k][i]
                                    - ind_all[2 * (i * n_act + k) + 1];
            outleft_x[n_act_x][i] = ind_all[2 * (i * n_act + k) + 1];
          }
          ++n_act_x;
        }
      }

      t_ranges = t_ranges_x;
      d_ranges = d_ranges_x;
      subdist = subdist_x;
      outleft = outleft_x;
      n_act = n_act_x;
    }
  }

  template <typename dataType>
  static void alltoall(std::vector<std::vector<ttk::SimplexId>> &right_ends,
                       std::vector<dataType> &data,
                       std::vector<dataType> &trans_data,
                       ttk::SimplexId *boundaries,
                       MPI_Datatype &MPI_valueType) {

    // Should be ttk::SimplexId, but MPI wants ints
    char errMsg[] = "32-bit limit for MPI has overflowed";
    ttk::SimplexId n_loc_ = data.size();
    bool overflowInt{false};
    int int_max_cus = INT_MAX;
    if(n_loc_ > int_max_cus) {
      std::cout << errMsg << std::endl;
      overflowInt = true;
    }
    int n_loc = static_cast<int>(n_loc_);
    // Calculate the counts for redistributing data
    // int send_counts[ttk::MPIsize_], send_disps[ttk::MPIsize_];
    std::vector<ttk::SimplexId> send_counts(ttk::MPIsize_);
    std::vector<ttk::SimplexId> recv_counts(ttk::MPIsize_);

    for(int i = 0; i < ttk::MPIsize_; ++i) {
      ttk::SimplexId scount
        = right_ends[i + 1][ttk::MPIrank_] - right_ends[i][ttk::MPIrank_];
      ttk::SimplexId rcount
        = right_ends[ttk::MPIrank_ + 1][i] - right_ends[ttk::MPIrank_][i];
      if(scount > int_max_cus || rcount > int_max_cus) {
        overflowInt = true;
        std::cout << errMsg << std::endl;
      }
      send_counts[i] = scount;
      recv_counts[i] = rcount;
    }

    if(!overflowInt) {
      std::vector<int> send_counts_int(ttk::MPIsize_);
      std::vector<int> send_disps_int(ttk::MPIsize_);
      std::vector<int> recv_counts_int(ttk::MPIsize_);
      std::vector<int> recv_disps_int(ttk::MPIsize_);

      for(int i = 0; i < ttk::MPIsize_; i++) {
        send_counts_int[i] = static_cast<int>(send_counts[i]);
        recv_counts_int[i] = static_cast<int>(recv_counts[i]);
      }

      recv_disps_int[0] = 0;
      std::partial_sum(recv_counts_int.data(),
                       recv_counts_int.data() + ttk::MPIsize_ - 1,
                       recv_disps_int.data() + 1);

      send_disps_int[0] = 0;
      std::partial_sum(send_counts_int.data(),
                       send_counts_int.data() + ttk::MPIsize_ - 1,
                       send_disps_int.data() + 1);
      // Do the transpose
      MPI_Alltoallv(data.data(), send_counts_int.data(), send_disps_int.data(),
                    MPI_valueType, trans_data.data(), recv_counts_int.data(),
                    recv_disps_int.data(), MPI_valueType, ttk::MPIcomm_);

      for(int i = 0; i < ttk::MPIsize_; ++i)
        boundaries[i] = (ttk::SimplexId)recv_disps_int[i];
      boundaries[ttk::MPIsize_] = (ttk::SimplexId)n_loc; // for the merging

      return;
    }

    std::vector<int> recv_counts_chunks(ttk::MPIsize_);
    std::vector<int> recv_disps_chunks(ttk::MPIsize_);
    std::vector<int> send_counts_chunks(ttk::MPIsize_);
    std::vector<int> send_disps_chunks(ttk::MPIsize_);
    std::vector<ttk::SimplexId> send_disps(ttk::MPIsize_);
    std::vector<ttk::SimplexId> recv_disps(ttk::MPIsize_);

    for(int i = 0; i < ttk::MPIsize_; i++) {
      recv_counts_chunks[i] = recv_counts[i] / int_max_cus;
      if(recv_counts[i] % int_max_cus != 0) {
        recv_counts_chunks[i]++;
      }
      send_counts_chunks[i] = send_counts[i] / int_max_cus;
      if(send_counts[i] % int_max_cus != 0) {
        send_counts_chunks[i]++;
      }
    }
    recv_disps_chunks[0] = 0;
    std::partial_sum(recv_counts_chunks.data(),
                     recv_counts_chunks.data() + ttk::MPIsize_ - 1,
                     recv_disps_chunks.data() + 1);
    send_disps_chunks[0] = 0;
    std::partial_sum(send_counts_chunks.data(),
                     send_counts_chunks.data() + ttk::MPIsize_ - 1,
                     send_disps_chunks.data() + 1);
    send_disps[0] = 0;
    std::partial_sum(send_counts.data(), send_counts.data() + ttk::MPIsize_ - 1,
                     send_disps.data() + 1);
    recv_disps[0] = 0;
    std::partial_sum(recv_counts.data(), recv_counts.data() + ttk::MPIsize_ - 1,
                     recv_disps.data() + 1);

    int bufferSize = std::accumulate(
      recv_counts_chunks.data(), recv_counts_chunks.data() + ttk::MPIsize_, 0);
    std::vector<dataType> send_buffer_64bits(bufferSize * int_max_cus);
    std::vector<dataType> recv_buffer_64bits(bufferSize * int_max_cus);

    for(int p = 0; p < ttk::MPIsize_; p++) {
      send_buffer_64bits.insert(
        send_buffer_64bits.begin() + send_disps_chunks[p] * int_max_cus,
        data.data() + send_disps[p],
        data.data() + send_disps[p] + send_counts[p]);
    }

    // Create chunk type
    MPI_Datatype MPI_valueChunkType;
    MPI_Type_contiguous(int_max_cus, MPI_valueType, &MPI_valueChunkType);
    MPI_Type_commit(&MPI_valueChunkType);

    MPI_Alltoallv(send_buffer_64bits.data(), send_counts_chunks.data(),
                  send_disps_chunks.data(), MPI_valueChunkType,
                  recv_buffer_64bits.data(), recv_counts_chunks.data(),
                  recv_disps_chunks.data(), MPI_valueChunkType, ttk::MPIcomm_);
    for(int p = 0; p < ttk::MPIsize_; p++) {
      trans_data.insert(
        trans_data.begin() + recv_disps[p],
        recv_buffer_64bits.data() + recv_disps_chunks[p] * int_max_cus,
        recv_buffer_64bits.data() + recv_disps_chunks[p] * int_max_cus
          + recv_counts[p]);
    }
    for(int i = 0; i < ttk::MPIsize_; ++i)
      boundaries[i] = recv_disps[i];
    boundaries[ttk::MPIsize_] = n_loc_; // for the merging

    return;
  }

  template <class _Compare, typename _RandomAccessIter>
  void psort_merge(_RandomAccessIter in,
                   _RandomAccessIter out,
                   ttk::SimplexId *disps,
                   _Compare comp,
                   _Compare oppositeComp) {

    if(ttk::MPIsize_ == 1) {
      std::copy(in, in + disps[ttk::MPIsize_], out);
      return;
    }

    _RandomAccessIter bufs[2] = {in, out};
    // ttk::SimplexId locs[ttk::MPIsize_];

    std::vector<ttk::SimplexId> locs(ttk::MPIsize_, 0);

    ttk::SimplexId next = 1;
    while(true) {
      ttk::SimplexId stride = next * 2;
      if(stride >= ttk::MPIsize_)
        break;

      for(ttk::SimplexId i = 0; i + next < ttk::MPIsize_; i += stride) {
        ttk::SimplexId end_ind
          = std::min(i + stride, (ttk::SimplexId)ttk::MPIsize_);

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
                 bufs[locs[next]] + disps[ttk::MPIsize_], out, comp);
    } else if(locs[next] == 0) {
      // 10 => backwards out of place
      std::merge(
        std::reverse_iterator<_RandomAccessIter>(in + disps[ttk::MPIsize_]),
        std::reverse_iterator<_RandomAccessIter>(in + disps[next]),
        std::reverse_iterator<_RandomAccessIter>(out + disps[next]),
        std::reverse_iterator<_RandomAccessIter>(out),
        std::reverse_iterator<_RandomAccessIter>(out + disps[ttk::MPIsize_]),
        oppositeComp);
    } else {
      // 11 => in-place
      std::inplace_merge(
        out, out + disps[next], out + disps[ttk::MPIsize_], comp);
    }
  }

  template <typename dataType, typename _Compare>
  void parallel_sort(std::vector<dataType> &data,
                     _Compare comp,
                     _Compare oppositeComp,
                     std::vector<ttk::SimplexId> &dist,
                     MPI_Datatype &MPI_valueType,
                     MPI_Datatype &MPI_distanceType,
                     int nThreads) {

    // Sort the data locally
    TTK_PSORT(nThreads, data.begin(), data.end(), comp);

    if(ttk::MPIsize_ == 1)
      return;

    // Find splitters
    std::vector<std::vector<ttk::SimplexId>> right_ends(
      ttk::MPIsize_ + 1, std::vector<ttk::SimplexId>(ttk::MPIsize_, 0));
    psort_split<_Compare>(data.begin(), data.end(), dist.data(), comp,
                          right_ends, MPI_valueType, MPI_distanceType,
                          nThreads);

    // Communicate to destination
    ttk::SimplexId n_loc = data.size();
    std::vector<dataType> trans_data(n_loc);

    std::vector<ttk::SimplexId> boundaries(ttk::MPIsize_ + 1);
    alltoall(right_ends, data, trans_data, boundaries.data(), MPI_valueType);

    psort_merge<_Compare>(
      trans_data.data(), data.data(), boundaries.data(), comp, oppositeComp);

    return;
  }

} // namespace p_sort
