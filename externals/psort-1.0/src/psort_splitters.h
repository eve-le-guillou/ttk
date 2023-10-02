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

#ifndef PSORT_SPLITTERS_H
#define PSORT_SPLITTERS_H

namespace p_sort {
  using namespace std;

  template <typename SplitType>
  class Split {
  public:
    template <typename _RandomAccessIter, class _Compare, typename _Distance>
    void split(_RandomAccessIter first,
               _RandomAccessIter last,
               _Distance *dist,
               _Compare comp,
               vector<vector<_Distance>> &right_ends,
               MPI_Datatype &MPI_valueType,
               MPI_Datatype &MPI_distanceType,
               MPI_Comm comm) {

      SplitType *s = static_cast<SplitType *>(this);
      s->real_split(first, last, dist, comp, right_ends, MPI_valueType,
                    MPI_distanceType, comm);
    }
  };

  class MedianSplit : public Split<MedianSplit> {
  public:

    template <typename _RandomAccessIter, class _Compare, typename _Distance>
    void real_split(_RandomAccessIter first,
                    _RandomAccessIter last,
                    _Distance *dist,
                    _Compare comp,
                    vector<vector<_Distance>> &right_ends,
                    MPI_Datatype &MPI_valueType,
                    MPI_Datatype &MPI_distanceType,
                    MPI_Comm comm) {

      typedef
        typename iterator_traits<_RandomAccessIter>::value_type _ValueType;

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
      std::vector<pair<_RandomAccessIter, _RandomAccessIter>> d_ranges(nproc
                                                                       - 1);
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
            = accumulate(subdist[k].begin(), subdist[k].end(), (_Distance)0)
              / 2;
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

  private:
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
  };

} // namespace p_sort

#endif /* PSORT_SPLITTERS_H */
