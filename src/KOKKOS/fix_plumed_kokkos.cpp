// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   PLUMED is a host library taking raw double pointers, so the collective
   variables are always evaluated on the host.  What this style keeps off the
   host is the per-atom traffic.

   A collective variable usually touches a handful of atoms.  PLUMED will say
   which ones: after prepareDependencies() has decided what is active this step,
   createFullList() returns the global indices it needs.  This style intersects
   that with the atoms this rank owns, gathers just those into compact buffers
   with a device kernel, and adds PLUMED's forces back with another - so neither
   x nor f is moved in full.

   Two cases fall back to handing over every local atom:
     - PLUMED wants the energy this step, because a biased ENERGY action
       rescales the whole force array rather than adding to it, which only
       works if the array is the real atom->f (see Atoms::share()).
     - the requested set is everything anyway, as it is on the first step,
       where PLUMED collects masses and charges once.
------------------------------------------------------------------------- */

#include "fix_plumed_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "comm.h"
#include "error.h"
#include "neighbor.h"
#include "update.h"

#include <algorithm>
#include <cstring>

#include "plumed/wrapper/Plumed.h"

using namespace LAMMPS_NS;

namespace {

/* ----------------------------------------------------------------------
   Copy the first n elements between the two halves of a DualView.

   DualView::sync_host() / sync_device() copy the WHOLE allocation.  These
   buffers are sized to nglobal -- the number of atoms PLUMED asked for
   globally, which bounds what any one rank can own -- so on a rank holding
   fewer than that, or on any rank once the buffers have grown to a
   high-water mark, most of that allocation is not in use.  Copying only the
   prefix is what keeps the transfer proportional to the subset rather than
   to the buffer.
------------------------------------------------------------------------- */

template <class DstView, class SrcView>
inline void copy_prefix(const DstView &dst, const SrcView &src, int n)
{
  if (n <= 0) return;
  Kokkos::deep_copy(Kokkos::subview(dst, Kokkos::make_pair(0, n)),
                    Kokkos::subview(src, Kokkos::make_pair(0, n)));
}

}    // namespace


template<class DeviceType>
FixPlumedKokkos<DeviceType>::FixPlumedKokkos(LAMMPS *lmp, int narg, char **arg) :
  FixPlumed(lmp, narg, arg),
  nmax_full(0), nglobal(0), nsub(-1), nlocal_cached(0), nsub_active(0),
  nsub_told(-1), nmax_sub(0), mode(0), subset_enable(1), stat_enable(1),
  n_calls(0), n_subset(0), n_none(0), n_full(0), n_rebuild(0), n_gathered(0)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;

  // all syncing is explicit, in set_plumed_atoms() and unpack_plumed_forces()

  datamask_read = EMPTY_MASK;
  datamask_modify = EMPTY_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixPlumedKokkos<DeviceType>::setmask()
{
  return FixPlumed::setmask() | FixConst::POST_RUN;
}

/* ----------------------------------------------------------------------
   fix_modify subset yes/no   - use the gather at all
   fix_modify stats yes/no    - print the summary at the end of the run
------------------------------------------------------------------------- */

template<class DeviceType>
int FixPlumedKokkos<DeviceType>::modify_param(int narg, char **arg)
{
  if (narg >= 2 && strcmp(arg[0], "subset") == 0) {
    subset_enable = (strcmp(arg[1], "yes") == 0);
    return 2;
  }
  if (narg >= 2 && strcmp(arg[0], "stats") == 0) {
    stat_enable = (strcmp(arg[1], "yes") == 0);
    return 2;
  }
  return FixPlumed::modify_param(narg, arg);
}

/* ----------------------------------------------------------------------
   where the steps went, and how often the cached mapping had to be rebuilt
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::post_run()
{
  if (!stat_enable || comm->me != 0 || n_calls == 0) return;

  const double pct = 100.0 / (double) n_calls;
  utils::logmesg(lmp, "fix {} (plumed/kk): {} calls - subset {:.1f}%, idle {:.1f}%, "
                 "full {:.1f}%\n", id, n_calls, pct*n_subset, pct*n_none, pct*n_full);
  utils::logmesg(lmp, "  mapping rebuilds {} ({:.1f}% of calls), "
                 "mean atoms gathered {:.1f}\n", n_rebuild, pct*n_rebuild,
                 n_subset ? (double) n_gathered / (double) n_subset : 0.0);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::grow_full(int n)
{
  if (n <= nmax_full) return;
  nmax_full = n;
  k_fbias = tdual_f64_1d("FixPlumedKokkos::fbias", 3*nmax_full);
}

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::grow_subset(int n)
{
  if (n <= nmax_sub) return;
  nmax_sub = n;
  k_gather_local = tdual_i32_1d("FixPlumedKokkos::gather_local", nmax_sub);
  k_gather_gat   = tdual_i32_1d("FixPlumedKokkos::gather_gat", nmax_sub);
  k_xsub = tdual_f64_1d("FixPlumedKokkos::xsub", 3*nmax_sub);
  k_fsub = tdual_f64_1d("FixPlumedKokkos::fsub", 3*nmax_sub);
  k_msub = tdual_f64_1d("FixPlumedKokkos::msub", nmax_sub);
  k_qsub = tdual_f64_1d("FixPlumedKokkos::qsub", nmax_sub);
}

/* ----------------------------------------------------------------------
   mark the global indices PLUMED asked for; returns 1 when the set changed
------------------------------------------------------------------------- */

template<class DeviceType>
int FixPlumedKokkos<DeviceType>::refresh_needed_list()
{
  int nfull = 0;
  p->cmd("createFullList", &nfull);
  const int *list = nullptr;
  p->cmd("getFullList", &list);

  // an empty request means PLUMED wants nothing this step - a strided CV does
  // this on every step it is inactive.  Leave the cached mapping alone so it
  // survives until the CV comes back, instead of invalidating it every step.

  if (nfull == 0) {
    p->cmd("clearFullList");
    nglobal = 0;
    return 0;
  }

  int changed = 0;
  if ((int) cached_list.size() != nfull) changed = 1;
  else for (int i = 0; i < nfull; i++)
    if (cached_list[i] != list[i]) { changed = 1; break; }

  if (changed) {
    cached_list.assign(list, list + nfull);
    if (nfull > (int) k_glist.extent(0))
      k_glist = tdual_i32_1d("FixPlumedKokkos::glist", nfull);
    // wanted() binary-searches this list, so it has to be sorted ascending.  It
    // already is - createFullList() builds it with Tools::mergeSortedVectors() -
    // but relying on that is a silent-wrong-answer dependency on a PLUMED
    // internal: an unsorted list would make the search miss atoms rather than
    // fail.  The check below is O(nfull) on a path that is already O(nfull) and
    // runs only when the requested set changed, so verifying costs nothing; the
    // sort is there so a future PLUMED that stops sorting is handled rather than
    // merely detected.
    auto h_glist = k_glist.view_host();
    int sorted = 1;
    for (int i = 0; i < nfull; i++) {
      h_glist(i) = cached_list[i];
      if (i && h_glist(i) < h_glist(i-1)) sorted = 0;
    }
    if (!sorted) std::sort(h_glist.data(), h_glist.data() + nfull);
    k_glist.modify_host();
    k_glist.template sync<DeviceType>();
  }
  nglobal = nfull;

  p->cmd("clearFullList");
  return changed;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::set_plumed_atoms()
{
  n_calls++;

  if (subset_enable && !plumedNeedsEnergy) {
    const int changed = refresh_needed_list();
    if (nglobal == 0) { set_plumed_atoms_none(); return; }
    if (nglobal < atom->natoms) {
      if (changed) nsub = -1;               // requested set moved: rebuild the mapping
      set_plumed_atoms_subset();
      return;
    }
  }

  set_plumed_atoms_full();
}

/* ----------------------------------------------------------------------
   every local atom, PLUMED accumulates into a private buffer unless it may
   rescale, in which case it is given atom->f itself
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::set_plumed_atoms_full()
{
  n_full++;
  if (mode != 0) { nlocal = -1; nsub_told = -1; mode = 0; }     // make the base re-send gatindex

  if (plumedNeedsEnergy)
    atomKK->sync(Host, X_MASK|F_MASK|TAG_MASK|TYPE_MASK|RMASS_MASK|Q_MASK);
  else
    atomKK->sync(Host, X_MASK|TAG_MASK|TYPE_MASK|RMASS_MASK|Q_MASK);

  update_local_atoms();

  p->cmd("setPositions", &atom->x[0][0]);

  if (plumedNeedsEnergy) {
    p->cmd("setForces", &atom->f[0][0]);
  } else {
    grow_full(nlocal);
    double *h = k_fbias.view_host().data();
    if (nlocal > 0) std::memset(h, 0, sizeof(double)*3*nlocal);
    p->cmd("setForces", h);
  }

  p->cmd("setMasses", &masses[0]);
  p->cmd("setCharges", &charges[0]);
}

/* ----------------------------------------------------------------------
   only the atoms PLUMED asked for
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::set_plumed_atoms_none()
{
  // Nothing is active this step, so PlumedMain::shareData() returns without
  // reading anything.  Issue no commands at all: in particular do not touch
  // setAtomsGatindex, which clears PLUMED's natoms-sized g2l map every call
  // (Atoms.cpp) and would cost an O(N) host pass for no reason.

  mode = 1;
  nsub_active = 0;
  n_none++;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int FixPlumedKokkos<DeviceType>::mapping_is_valid()
{
  if (nsub < 0) return 0;
  if (nsub == 0) return 1;

  atomKK->sync(execution_space, TAG_MASK);
  tag = atomKK->k_tag.template view<DeviceType>();
  d_gather_local = k_gather_local.template view<DeviceType>();
  d_gather_gat = k_gather_gat.template view<DeviceType>();

  int bad = 0;
  copymode = 1;
  Kokkos::parallel_reduce(Kokkos::RangePolicy<DeviceType, TagFixPlumedCheck>(0, nsub),
                          *this, bad);
  copymode = 0;

  return bad == 0;
}

/* ----------------------------------------------------------------------
   O(nlocal) device scan: find the locally owned atoms PLUMED asked for
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::rebuild_mapping()
{
  // nsub is bounded by nglobal -- a rank can only own a subset of what PLUMED
  // asked for -- and nglobal is known before the scan.  Size to that, so the
  // scan never needs a reallocation on the hot path.  High-water mark growth,
  // so a rank whose share shrinks does not thrash; the subview copies below
  // move only the nsub prefix actually in use.
  const int nlocal_all = atom->nlocal;
  grow_subset(nglobal > 0 ? nglobal : 1);

  atomKK->sync(execution_space, TAG_MASK);
  tag = atomKK->k_tag.template view<DeviceType>();
  d_glist = k_glist.template view<DeviceType>();
  d_gather_local = k_gather_local.template view<DeviceType>();
  d_gather_gat = k_gather_gat.template view<DeviceType>();

  int count = 0;
  copymode = 1;
  Kokkos::parallel_scan(Kokkos::RangePolicy<DeviceType, TagFixPlumedGather>(0, nlocal_all),
                        *this, count);
  copymode = 0;

  nsub = count;
  nlocal_cached = nlocal_all;
  copy_prefix(k_gather_gat.view_host(), k_gather_gat.template view<DeviceType>(), nsub);
}

/* ----------------------------------------------------------------------
   only the atoms PLUMED asked for.  The local mapping is cached: each step it
   is checked in O(nsub) and rebuilt with the O(nlocal) scan only when atoms
   have moved between ranks or been reordered by Atom::sort(), or when PLUMED
   has changed what it is asking for.
------------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::set_plumed_atoms_subset()
{
  const int was_full = (mode != 1);
  mode = 1;

  // Atoms change ranks only during an exchange, and Atom::sort() reorders only
  // there too, so a neighbour rebuild is the one step on which the mapping can
  // go stale.  Checking the cached slots alone is not enough: it would miss an
  // atom that has just *arrived* on this rank, since every existing slot still
  // points where it did.

  const int exchanged = (neighbor->ago == 0);

  int rebuilt = 0;
  if (was_full || nsub < 0 || exchanged || !mapping_is_valid()) {
    rebuild_mapping();
    rebuilt = 1;
    n_rebuild++;
  }

  atomKK->sync(execution_space, X_MASK);
  x = atomKK->k_x.template view<DeviceType>();
  d_gather_local = k_gather_local.template view<DeviceType>();
  d_xsub = k_xsub.template view<DeviceType>();

  if (nsub > 0) {
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPlumedPos>(0, nsub), *this);
    copymode = 0;
    copy_prefix(k_xsub.view_host(), k_xsub.template view<DeviceType>(), 3*nsub);
  }

  // masses and charges only matter on the steps PLUMED collects them, and it
  // caches them after the first, so refresh them only when the mapping moved

  if (rebuilt && nsub > 0) {
    copy_prefix(k_gather_local.view_host(), k_gather_local.template view<DeviceType>(), nsub);
    atomKK->sync(Host, TYPE_MASK|RMASS_MASK|Q_MASK);
    auto h_loc = k_gather_local.view_host();
    auto h_m = k_msub.view_host();
    auto h_q = k_qsub.view_host();
    for (int j = 0; j < nsub; j++) {
      const int i = h_loc(j);
      h_m(j) = atom->rmass_flag ? atom->rmass[i] : atom->mass[atom->type[i]];
      h_q(j) = atom->q_flag ? atom->q[i] : 0.0;
    }
    k_msub.modify_host();
    k_qsub.modify_host();
  }

  double *h_fsub = k_fsub.view_host().data();
  if (nsub > 0) std::memset(h_fsub, 0, sizeof(double)*3*nsub);

  nsub_active = nsub;
  n_subset++;
  n_gathered += nsub;

  // setAtomsGatindex() clears PLUMED's natoms-sized g2l map on every call, so
  // only hand the set over when it has actually changed - which is exactly
  // when the mapping was rebuilt.

  if (rebuilt || nsub != nsub_told) {
    p->cmd("setAtomsNlocal", &nsub);
    p->cmd("setAtomsGatindex", k_gather_gat.view_host().data());
    nsub_told = nsub;
  }

  p->cmd("setPositions", k_xsub.view_host().data());
  p->cmd("setForces", h_fsub);
  p->cmd("setMasses", k_msub.view_host().data());
  p->cmd("setCharges", k_qsub.view_host().data());
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void FixPlumedKokkos<DeviceType>::unpack_plumed_forces()
{
  if (mode == 0) {
    if (plumedNeedsEnergy) {                 // PLUMED wrote into atom->f itself
      atomKK->modified(Host, F_MASK);
      return;
    }
    if (nlocal == 0) return;
    k_fbias.modify_host();
    k_fbias.template sync<DeviceType>();
    d_fbias = k_fbias.template view<DeviceType>();
    atomKK->sync(execution_space, F_MASK);
    f = atomKK->k_f.template view<DeviceType>();
    copymode = 1;
    Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPlumedAddBias>(0, nlocal),
                         *this);
    copymode = 0;
    atomKK->modified(execution_space, F_MASK);
    return;
  }

  if (nsub_active <= 0) return;

  copy_prefix(k_fsub.template view<DeviceType>(), k_fsub.view_host(), 3*nsub_active);
  d_fsub = k_fsub.template view<DeviceType>();
  d_gather_local = k_gather_local.template view<DeviceType>();

  atomKK->sync(execution_space, F_MASK);
  f = atomKK->k_f.template view<DeviceType>();

  copymode = 1;
  Kokkos::parallel_for(Kokkos::RangePolicy<DeviceType, TagFixPlumedScatter>(0, nsub_active), *this);
  copymode = 0;

  atomKK->modified(execution_space, F_MASK);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixPlumedKokkos<DeviceType>::operator()(TagFixPlumedAddBias, const int i) const
{
  f(i,0) += static_cast<KK_ACC_FLOAT>(d_fbias(3*i+0));
  f(i,1) += static_cast<KK_ACC_FLOAT>(d_fbias(3*i+1));
  f(i,2) += static_cast<KK_ACC_FLOAT>(d_fbias(3*i+2));
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixPlumedKokkos<DeviceType>::operator()(TagFixPlumedGather, const int i,
                                             int &slot, const bool final) const
{
  const int gid = (int) tag(i) - 1;
  const int want = wanted(gid);
  if (final && want) {
    d_gather_local(slot) = i;
    d_gather_gat(slot) = gid;
  }
  slot += want;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixPlumedKokkos<DeviceType>::operator()(TagFixPlumedCheck, const int j, int &bad) const
{
  // the cached slot still has to point at the atom it was built for; a rank
  // exchange or an Atom::sort() breaks this and forces a rebuild
  const int i = d_gather_local(j);
  if (i >= nlocal_cached || (int) tag(i) - 1 != d_gather_gat(j)) bad += 1;
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixPlumedKokkos<DeviceType>::operator()(TagFixPlumedPos, const int j) const
{
  const int i = d_gather_local(j);
  d_xsub(3*j+0) = static_cast<double>(x(i,0));
  d_xsub(3*j+1) = static_cast<double>(x(i,1));
  d_xsub(3*j+2) = static_cast<double>(x(i,2));
}

template<class DeviceType>
// NOLINTNEXTLINE
KOKKOS_INLINE_FUNCTION
void FixPlumedKokkos<DeviceType>::operator()(TagFixPlumedScatter, const int j) const
{
  const int i = d_gather_local(j);
  f(i,0) += static_cast<KK_ACC_FLOAT>(d_fsub(3*j+0));
  f(i,1) += static_cast<KK_ACC_FLOAT>(d_fsub(3*j+1));
  f(i,2) += static_cast<KK_ACC_FLOAT>(d_fsub(3*j+2));
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
double FixPlumedKokkos<DeviceType>::memory_usage()
{
  double bytes = FixPlumed::memory_usage();
  bytes += (double) k_glist.extent(0) * sizeof(int) * 2.0;
  bytes += (double) 3*nmax_full * sizeof(double) * 2.0;
  bytes += (double) nmax_sub * (8.0*sizeof(double) + 2.0*sizeof(int)) * 2.0;
  return bytes;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class FixPlumedKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class FixPlumedKokkos<LMPHostType>;
#endif
}
