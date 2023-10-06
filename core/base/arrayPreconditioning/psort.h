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

  template <typename _ValueType, typename _Distance, typename _Compare>
  void parallel_sort(std::vector<_ValueType> &data,
                     _Compare comp,
                     _Compare oppositeComp,
                     long *dist_in,
                     MPI_Comm comm) {

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
    __gnu_parallel::sort(data.begin(), data.end(), comp);

    if(nproc == 1)
      return;

    // Find splitters
    std::vector<vector<_Distance>> right_ends(
      nproc + 1, vector<_Distance>(nproc, 0));
    psort_split(data.begin(), data.end(), dist.data(), comp, right_ends,
                MPI_valueType, MPI_distanceType, comm);

    // Communicate to destination
    _Distance n_loc = data.size();
    std::vector<_ValueType> trans_data(n_loc);
    //_Distance boundaries[nproc+1];
    std::vector<_Distance> boundaries(nproc + 1);
    alltoall(
      right_ends, data, trans_data, boundaries.data(), MPI_valueType, comm);

    psort_merge(trans_data.data(), data.data(), boundaries.data(), nproc, comp,
                oppositeComp);

    MPI_Type_free(&MPI_valueType);
    MPI_Type_free(&MPI_distanceType);

    return;
  }

  template <typename _ValueType, typename _Distance>
  void parallel_sort(std::vector<_ValueType> &data, long *dist, MPI_Comm comm) {
    parallel_sort<_ValueType, _Distance>(
      data, comp1, oppositeComp1, dist, comm);
  }

  template <typename _ValueType, typename _Distance>
  static void alltoall(std::vector<std::vector<_Distance>> &right_ends,
                       std::vector<_ValueType> &data,
                       std::vector<_ValueType> &trans_data,
                       _Distance *boundaries,
                       MPI_Datatype &MPI_valueType,
                       MPI_Comm comm) {

    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    // Should be _Distance, but MPI wants ints
    char errMsg[] = "32-bit limit for MPI has overflowed";
    _Distance n_loc_ = data.size();
    bool overflowInt{false};
    if(n_loc_ > INT_MAX) {
      std::cout << errMsg << std::endl;
      overflowInt = true;
    }
    int n_loc = static_cast<int>(n_loc_);
    // Calculate the counts for redistributing data
    // int send_counts[nproc], send_disps[nproc];
    std::vector<_Distance> send_counts(nproc);
    std::vector<_Distance> recv_counts(nproc);

    for(int i = 0; i < nproc; ++i) {
      _Distance scount = right_ends[i + 1][rank] - right_ends[i][rank];
      _Distance rcount = right_ends[rank + 1][i] - right_ends[rank][i];
      if(scount > INT_MAX || rcount > INT_MAX) {
        overflowInt = true;
        std::cout << errMsg << std::endl;
      }
      send_counts[i] = scount;
      recv_counts[i] = rcount;
    }
    if(!overflowInt) {
      std::vector<int> send_counts_int(nproc);
      std::vector<int> send_disps_int(nproc);
      std::vector<int> recv_counts_int(nproc);
      std::vector<int> recv_disps_int(nproc);

      for(int i = 0; i < nproc; i++) {
        send_counts_int[i] = static_cast<int>(send_counts[i]);
        recv_counts_int[i] = static_cast<int>(recv_counts[i]);
      }

      recv_disps_int[0] = 0;
      std::partial_sum(recv_counts_int.data(),
                       recv_counts_int.data() + nproc - 1,
                       recv_disps_int.data() + 1);

      send_disps_int[0] = 0;
      std::partial_sum(send_counts_int.data(),
                       send_counts_int.data() + nproc - 1,
                       send_disps_int.data() + 1);
      // Do the transpose
      MPI_Alltoallv(data.data(), send_counts_int.data(), send_disps_int.data(),
                    MPI_valueType, trans_data.data(), recv_counts_int.data(),
                    recv_disps_int.data(), MPI_valueType, comm);

      for(int i = 0; i < nproc; ++i)
        boundaries[i] = (_Distance)recv_disps_int[i];
      boundaries[nproc] = (_Distance)n_loc; // for the merging

      return;
    }

    std::vector<int> recv_counts_chunks(nproc);
    std::vector<int> recv_disps_chunks(nproc);
    std::vector<int> send_counts_chunks(nproc);
    std::vector<int> send_disps_chunks(nproc);
    std::vector<_Distance> send_disps(nproc);
    std::vector<_Distance> recv_disps(nproc);

    for(int i = 0; i < nproc; i++) {
      recv_counts_chunks[i] = recv_counts[i] / INT_MAX;
      if(recv_counts[i] % INT_MAX != 0) {
        recv_counts_chunks[i]++;
      }
      send_counts_chunks[i] = send_counts[i] / INT_MAX;
      if(send_counts[i] % INT_MAX != 0) {
        send_counts_chunks[i]++;
      }
    }
    recv_disps_chunks[0] = 0;
    std::partial_sum(recv_counts_chunks.data(),
                     recv_counts_chunks.data() + nproc - 1,
                     recv_disps_chunks.data() + 1);
    send_disps_chunks[0] = 0;
    std::partial_sum(send_counts_chunks.data(),
                     send_counts_chunks.data() + nproc - 1,
                     send_disps_chunks.data() + 1);
    send_disps[0] = 0;
    std::partial_sum(send_counts.data(), send_counts.data() + nproc - 1,
                     send_disps.data() + 1);
    recv_disps[0] = 0;
    std::partial_sum(recv_counts.data(), recv_counts.data() + nproc - 1,
                     recv_disps.data() + 1);

    int bufferSize = accumulate(
      recv_counts_chunks.data(), recv_counts_chunks.data() + nproc, 0);
    std::vector<_ValueType> send_buffer_64bits(bufferSize * INT_MAX);
    std::vector<_ValueType> recv_buffer_64bits(bufferSize * INT_MAX);

    for(int p = 0; p < nproc; p++) {
      send_buffer_64bits.insert(
        send_buffer_64bits.begin() + send_disps_chunks[p] * INT_MAX,
        data.data() + send_disps[p],
        data.data() + send_disps[p] + send_counts[p]);
    }

    // Create chunk type
    MPI_Datatype MPI_valueChunkType;
    MPI_Type_contiguous(
      sizeof(_ValueType) * INT_MAX, MPI_CHAR, &MPI_valueChunkType);
    MPI_Type_commit(&MPI_valueChunkType);

    MPI_Alltoallv(send_buffer_64bits.data(), send_counts_chunks.data(),
                  send_disps_chunks.data(), MPI_valueChunkType,
                  recv_buffer_64bits.data(), recv_counts_chunks.data(),
                  recv_disps_chunks.data(), MPI_valueChunkType, comm);
    for(int p = 0; p < nproc; p++) {
      trans_data.insert(
        trans_data.begin() + recv_disps[p],
        recv_buffer_64bits.data() + recv_disps_chunks[p] * INT_MAX,
        recv_buffer_64bits.data() + recv_disps_chunks[p] * INT_MAX
          + recv_counts[p]);
    }
    for(int i = 0; i < nproc; ++i)
      boundaries[i] = recv_disps[i];
    boundaries[nproc] = (_Distance)n_loc; // for the merging

    return;
  }

  template <typename _RandomAccessIter, class _Compare, typename _Distance>
  void psort_merge(_RandomAccessIter in,
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

  template <typename _RandomAccessIter, class _Compare, typename _Distance>
  void psort_split(_RandomAccessIter first,
                   _RandomAccessIter last,
                   _Distance *dist,
                   _Compare comp,
                   vector<vector<_Distance>> &right_ends,
                   MPI_Datatype &MPI_valueType,
                   MPI_Datatype &MPI_distanceType,
                   MPI_Comm comm) {

    typedef typename iterator_traits<_RandomAccessIter>::value_type _ValueType;

    int nproc, rank;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &rank);

    int n_real = nproc;
    for(int i = 0; i < nproc; ++i)
      if(dist[i] == 0) {
        n_real = i;
        break;
      }

    copy(dist, dist + nproc, right_ends[nproc].begin());

    // union of [0, right_end[i+1]) on each processor produces dist[i] total
    // values
    //_Distance targets[nproc-1];
    std::vector<_Distance> targets(nproc - 1);
    partial_sum(dist, dist + (nproc - 1), targets.data());

    // keep a list of ranges, trying to "activate" them at each branch
    std::vector<pair<_RandomAccessIter, _RandomAccessIter>> d_ranges(nproc - 1);
    std::vector<pair<_Distance *, _Distance *>> t_ranges(nproc - 1);
    d_ranges[0] = make_pair(first, last);
    t_ranges[0] = make_pair(targets.data(), targets.data() + nproc - 1);

    // invariant: subdist[i][rank] == d_ranges[i].second - d_ranges[i].first
    // amount of data each proc still has in the search
    std::vector<std::vector<_Distance>> subdist(
      nproc - 1, std::vector<_Distance>(nproc));
    copy(dist, dist + nproc, subdist[0].begin());

    // for each processor, d_ranges - first
    std::vector<std::vector<_Distance>> outleft(
      nproc - 1, std::vector<_Distance>(nproc, 0));

    for(int n_act = 1; n_act > 0;) {

      for(int k = 0; k < n_act; ++k) {
        assert(subdist[k][rank] == d_ranges[k].second - d_ranges[k].first);
      }

      //------- generate n_act guesses

      // for the allgather, make a flat array of nproc chunks, each with n_act
      // elts
      std::vector<_ValueType> medians(nproc * n_act);
      for(int k = 0; k < n_act; ++k) {
        if(d_ranges[k].first != last) {
          _ValueType *ptr = &(*d_ranges[k].first);
          _Distance index = subdist[k][rank] / 2;
          medians[rank * n_act + k] = ptr[index];
        } else
          medians[rank * n_act + k] = *(last - 1);
      }

      // std::cout << "Median allGather" << std::endl;
      MPI_Allgather(MPI_IN_PLACE, n_act, MPI_valueType, &medians[0], n_act,
                    MPI_valueType, comm);
      // MPI_Allgather (&medians[rank*n_act], n_act, MPI_valueType,
      //     &medians[0], n_act, MPI_valueType, comm);
      // std::cout << "Median allGather2" << std::endl;

      // compute the weighted median of medians
      std::vector<_ValueType> queries(n_act);

      std::vector<_Distance> ms_perm(n_real);
      for(int k = 0; k < n_act; ++k) {
        //_Distance ms_perm[n_real];

        for(int i = 0; i < n_real; ++i)
          ms_perm[i] = i * n_act + k;
        sort(ms_perm.data(), ms_perm.data() + n_real,
             PermCompare<_ValueType, _Compare>(&medians[0], comp));

        _Distance mid
          = accumulate(subdist[k].begin(), subdist[k].end(), (_Distance)0) / 2;
        _Distance query_ind = -1;
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
      //_Distance ind_local[2 * n_act];
      std::vector<_Distance> ind_local(2 * n_act);

      for(int k = 0; k < n_act; ++k) {
        std::pair<_RandomAccessIter, _RandomAccessIter> ind_local_p
          = equal_range(
            d_ranges[k].first, d_ranges[k].second, queries[k], comp);

        ind_local[2 * k] = ind_local_p.first - first;
        ind_local[2 * k + 1] = ind_local_p.second - first;
      }

      //_Distance ind_all[2 * n_act * nproc];
      std::vector<_Distance> ind_all(2 * n_act * nproc);
      MPI_Allgather(ind_local.data(), 2 * n_act, MPI_distanceType,
                    ind_all.data(), 2 * n_act, MPI_distanceType, comm);
      // sum to get the global range of indices
      std::vector<std::pair<_Distance, _Distance>> ind_global(n_act);
      for(int k = 0; k < n_act; ++k) {
        ind_global[k] = make_pair(0, 0);
        for(int i = 0; i < nproc; ++i) {
          ind_global[k].first += ind_all[2 * (i * n_act + k)];
          ind_global[k].second += ind_all[2 * (i * n_act + k) + 1];
        }
      }

      // state to pass on to next iteration
      std::vector<pair<_RandomAccessIter, _RandomAccessIter>> d_ranges_x(nproc
                                                                         - 1);
      std::vector<std::pair<_Distance *, _Distance *>> t_ranges_x(nproc - 1);
      std::vector<std::vector<_Distance>> subdist_x(
        nproc - 1, vector<_Distance>(nproc));
      std::vector<std::vector<_Distance>> outleft_x(
        nproc - 1, vector<_Distance>(nproc, 0));
      int n_act_x = 0;

      for(int k = 0; k < n_act; ++k) {
        _Distance *split_low = lower_bound(
          t_ranges[k].first, t_ranges[k].second, ind_global[k].first);
        _Distance *split_high = upper_bound(
          t_ranges[k].first, t_ranges[k].second, ind_global[k].second);

        // iterate over targets we hit
        for(_Distance *s = split_low; s != split_high; ++s) {
          assert(*s > 0);
          // a bit sloppy: if more than one target in range, excess won't zero
          // out
          _Distance excess = *s - ind_global[k].first;
          // low procs to high take excess for stability
          for(int i = 0; i < nproc; ++i) {
            _Distance amount = min(ind_all[2 * (i * n_act + k)] + excess,
                                   ind_all[2 * (i * n_act + k) + 1]);
            right_ends[(s - targets.data()) + 1][i] = amount;
            excess -= amount - ind_all[2 * (i * n_act + k)];
          }
        }

        if((split_low - t_ranges[k].first) > 0) {
          t_ranges_x[n_act_x] = make_pair(t_ranges[k].first, split_low);
          // lop off local_ind_low..end
          d_ranges_x[n_act_x]
            = make_pair(d_ranges[k].first, first + ind_local[2 * k]);
          for(int i = 0; i < nproc; ++i) {
            subdist_x[n_act_x][i]
              = ind_all[2 * (i * n_act + k)] - outleft[k][i];
            outleft_x[n_act_x][i] = outleft[k][i];
          }
          ++n_act_x;
        }

        if((t_ranges[k].second - split_high) > 0) {
          t_ranges_x[n_act_x] = make_pair(split_high, t_ranges[k].second);
          // lop off begin..local_ind_high
          d_ranges_x[n_act_x]
            = make_pair(first + ind_local[2 * k + 1], d_ranges[k].second);
          for(int i = 0; i < nproc; ++i) {
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

} // namespace p_sort
