/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef FIX_CLASS
// clang-format off
FixStyle(plumed/kk,FixPlumedKokkos<LMPDeviceType>);
FixStyle(plumed/kk/device,FixPlumedKokkos<LMPDeviceType>);
FixStyle(plumed/kk/host,FixPlumedKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_FIX_PLUMED_KOKKOS_H
#define LMP_FIX_PLUMED_KOKKOS_H

#include "fix_plumed.h"
#include "kokkos_type.h"

#include <vector>

namespace LAMMPS_NS {

struct TagFixPlumedAddBias {};      // add PLUMED's forces back, all local atoms
struct TagFixPlumedGather {};       // compact: rebuild the local mapping
struct TagFixPlumedCheck {};        // cheap: is the cached mapping still valid
struct TagFixPlumedPos {};          // cheap: refill positions through the mapping
struct TagFixPlumedScatter {};      // compact: add PLUMED's forces back

template<class DeviceType>
class FixPlumedKokkos : public FixPlumed {
  public:
    typedef ArrayTypes<DeviceType> AT;

    // PLUMED reads and writes double on the host whatever KK_FLOAT is; these are
    // allocated by this style, so rule 3's DAT:: restriction does not apply
    typedef Kokkos::DualView<double*, Kokkos::LayoutRight, DeviceType> tdual_f64_1d;
    typedef Kokkos::DualView<int*, Kokkos::LayoutRight, DeviceType> tdual_i32_1d;

    FixPlumedKokkos(class LAMMPS *, int, char **);

    int setmask() override;
    void post_run() override;
    int modify_param(int, char **) override;

    double memory_usage() override;

// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(TagFixPlumedAddBias, const int) const;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(TagFixPlumedGather, const int, int &, const bool) const;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(TagFixPlumedCheck, const int, int &) const;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(TagFixPlumedPos, const int) const;
// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    void operator()(TagFixPlumedScatter, const int) const;

  protected:
    void set_plumed_atoms() override;
    void unpack_plumed_forces() override;

    void set_plumed_atoms_full();      // every local atom, PLUMED writes atom->f
    void set_plumed_atoms_subset();    // only the atoms PLUMED asked for
    void set_plumed_atoms_none();      // PLUMED wants nothing this step
    void grow_full(int);
    void grow_subset(int);
    int  refresh_needed_list();        // returns 1 if the requested set changed
    int  mapping_is_valid();           // O(nsub) check that the cache still holds
    void rebuild_mapping();            // O(nlocal) device scan, only when needed

    // --- full path (v3 behaviour) ---
    tdual_f64_1d k_fbias;
    int nmax_full;

    // --- subset path ---
    tdual_i32_1d k_glist;              // PLUMED's requested global indices, sorted
    int nglobal;                       // entries in k_glist
    tdual_i32_1d k_gather_local;       // slot -> local index
    tdual_i32_1d k_gather_gat;         // slot -> global index (PLUMED's gatindex)
    tdual_f64_1d k_xsub, k_fsub;       // 3*nsub
    tdual_f64_1d k_msub, k_qsub;       // nsub
    std::vector<int> cached_list;      // last list PLUMED asked for
    int nsub;                          // local atoms in that list
    int nlocal_cached;                 // atom->nlocal when the mapping was built
    int nsub_active;                   // atoms actually handed over this step
    int nsub_told;                     // last count PLUMED was given
    int nmax_sub;

    int mode;                          // 0 = full, 1 = subset
    int subset_enable;                 // fix_modify subset yes/no
    int stat_enable;                   // fix_modify stats yes/no

    // per-run statistics, reported by post_run()
    bigint n_calls, n_subset, n_none, n_full, n_rebuild;
    bigint n_gathered;                 // total atoms handed over on subset steps

    // views used inside the kernels
    typename AT::t_kkacc_1d_3 f;
    typename AT::t_kkfloat_1d_3_lr x;
    typename AT::t_tagint_1d tag;
    typename tdual_f64_1d::t_dev d_fbias, d_xsub, d_fsub;
    typename tdual_i32_1d::t_dev d_glist, d_gather_local, d_gather_gat;

// NOLINTNEXTLINE
    KOKKOS_INLINE_FUNCTION
    int wanted(const int gid) const {
      // k_glist comes from PLUMED's createFullList(), which builds it with
      // Tools::mergeSortedVectors() and so returns it sorted ascending
      int lo = 0, hi = nglobal - 1;
      while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        const int v = d_glist(mid);
        if (v == gid) return 1;
        if (v < gid) lo = mid + 1; else hi = mid - 1;
      }
      return 0;
    }
};

}    // namespace LAMMPS_NS

#endif
#endif
