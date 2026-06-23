from collections import defaultdict

import numpy as np
from rtree.index import Index, Property

from pyfr.cache import memoize
from pyfr.mpiutil import autofree, get_comm_rank_root, get_start_end_csize, mpi
from pyfr.polys import get_polybasis
from pyfr.shapes import BaseShape
from pyfr.util import subclass_where


class PointLocator:
    def __init__(self, mesh, fine_order=6):
        self.mesh = mesh
        self.fine_order = fine_order

    def _reduce_elocs(self, npts, elocator):
        comm, rank, root = get_comm_rank_root()

        # Allocate the location buffer
        dtype = [('dist', float), ('cidx', np.int16), ('eidx', np.int64),
                 ('tloc', float, self.mesh.ndims)]
        locs = np.zeros(npts, dtype=dtype)
        locs['dist'] = np.inf

        # Reduce over each of our element types
        for etype, eidxs in self.mesh.eidxs.items():
            cidx = self.mesh.codec.index(f'eles/{etype}')

            for i, (dist, eidx, tloc) in elocator(etype).items():
                l = locs[i]

                if dist < l['dist']:
                    l['dist'], l['tloc'] = dist, tloc
                    l['cidx'], l['eidx'] = cidx, eidxs[eidx]

        # Reduce over all ranks
        self._minloc(comm.Allreduce, mpi.IN_PLACE, locs, ndim=3)

        return locs

    def _locate_nnode(self, pts):
        print("\n[_locate_nnode] entering nearest-node fallback")
        print("[_locate_nnode] npts =", len(pts))

        # Find the nearest node to each query point
        nearest = self._find_closest_node(pts)

        print("[_locate_nnode] nearest-node distances min/max =",
            nearest['dist'].min(), nearest['dist'].max())

        for i, n in enumerate(nearest[:10]):
            print("    nearest for local failed pt =", i)
            print("        pt =", pts[i])
            print("        nearest node global idx =", n['idx'])
            print("        nearest node dist =", n['dist'])

        # Callback to identify the nearest elements to these nodes
        def elocator(etype):
            return self._find_closest_element_nnode(etype, pts, nearest)

        locs = self._reduce_elocs(len(pts), elocator)

        print("[_locate_nnode] finite nnode locs =",
            np.count_nonzero(np.isfinite(locs['dist'])))
        print("[_locate_nnode] inf nnode locs =",
            np.count_nonzero(np.isinf(locs['dist'])))

        return locs

    def _locate_bbox(self, pts):
        print("\n[_locate_bbox] entering bbox locator")
        print("[_locate_bbox] npts =", len(pts))

        def elocator(etype):
            print(f"\n[_locate_bbox] checking etype = {etype}")
            return self._find_closest_element_bbox(etype, pts)

        locs = self._reduce_elocs(len(pts), elocator)

        print("\n[_locate_bbox] reduced bbox results")
        print("[_locate_bbox] finite =", np.count_nonzero(np.isfinite(locs['dist'])))
        print("[_locate_bbox] inf =", np.count_nonzero(np.isinf(locs['dist'])))

        return locs

    def locate(self, pts):
        print("\n[locate] starting point location")
        print("[locate] npts =", len(pts))
        print("[locate] pts min =", np.min(pts, axis=0))
        print("[locate] pts max =", np.max(pts, axis=0))

        # Attempt to locate points using a bounding box approach
        locs = self._locate_bbox(pts)

        print("\n[locate] after _locate_bbox")
        print("[locate] finite bbox locs =", np.count_nonzero(np.isfinite(locs['dist'])))
        print("[locate] inf bbox locs =", np.count_nonzero(np.isinf(locs['dist'])))

        # Print per-point bbox result
        for i, l in enumerate(locs):
            print("    [locate/bbox-result]")
            print("        idx  =", i)
            print("        pt   =", pts[i])
            print("        dist =", l['dist'])
            print("        cidx =", l['cidx'])
            print("        eidx =", l['eidx'])
            print("        tloc =", l['tloc'])

        # Fall back to nearest-node locator for failed bbox points
        if np.any(infp := np.isinf(locs['dist'])):
            failed = np.where(infp)[0]

            print("\n[locate] bbox failed point indices =", failed[:50])
            print("[locate] first failed bbox points:")
            for i in failed[:10]:
                print("    idx =", i, "pt =", pts[i])

            nlocs = self._locate_nnode(pts[infp])
            locs[infp] = nlocs

            print("\n[locate] after _locate_nnode fallback")
            print("[locate] finite total locs =", np.count_nonzero(np.isfinite(locs['dist'])))
            print("[locate] inf total locs =", np.count_nonzero(np.isinf(locs['dist'])))

            for local_i, global_i in enumerate(failed[:10]):
                l = nlocs[local_i]
                print("    [locate/nnode-result]")
                print("        original idx =", global_i)
                print("        pt           =", pts[global_i])
                print("        dist         =", l['dist'])
                print("        cidx         =", l['cidx'])
                print("        eidx         =", l['eidx'])
                print("        tloc         =", l['tloc'])

        # Final validation
        for i, l in enumerate(locs):
            if l['dist'] == np.inf:
                ploc = ', '.join(str(p) for p in pts[i])
                print("\n[locate] FINAL FAILURE")
                print("    idx  =", i)
                print("    pt   =", pts[i])
                print("    dist =", l['dist'])
                print("    cidx =", l['cidx'])
                print("    eidx =", l['eidx'])
                print("    tloc =", l['tloc'])
                raise ValueError(f'Unable to locate point ({ploc})')

        return locs

    @memoize
    def _get_shape_basis(self, etype, nspts):
        shape = subclass_where(BaseShape, name=etype)
        order = shape.order_from_npts(nspts)
        basis = get_polybasis(etype, order + 1, shape.std_ele(order))

        return shape, basis

    def _minloc(self, coll, x, y, ndim=None):
        dtype = y.dtype
        fields = list(dtype.fields)[:ndim]

        def op(pmem, qmem, dt):
            p = np.frombuffer(pmem, dtype=dtype)
            q = np.frombuffer(qmem, dtype=dtype)

            lmask = p[fields[0]] < q[fields[0]]
            emask = p[fields[0]] == q[fields[0]]

            for f in fields[1:]:
                lmask |= emask & (p[f] < q[f])
                emask &= p[f] == q[f]

            q[lmask] = p[lmask]

        sbuf = (x, mpi.BYTE) if x is not mpi.IN_PLACE else x
        rbuf = (y, mpi.BYTE)

        coll(sbuf, rbuf, op=autofree(mpi.Op.Create(op, commute=False)))

    @memoize
    def _get_nodes_off_tree(self):
        comm, rank, root = get_comm_rank_root()

        # Read our portion of the nodes table
        start, end, _ = get_start_end_csize(comm, len(self.mesh.raw['nodes']))
        nodes = self.mesh.raw['nodes'][start:end]['location']

        # Insert these points into a spatial index
        tree = Index((np.arange(len(nodes)), nodes, nodes),
                     properties=Property(dimension=self.mesh.ndims))

        return nodes, start, tree

    @memoize
    def _get_bbox_tree(self, etype, *, scale=1.05):
        spts = self.mesh.spts[etype]

        print(f"\n[_get_bbox_tree] etype = {etype}")
        print("[_get_bbox_tree] spts shape =", spts.shape)
        print("[_get_bbox_tree] raw spts min =", spts.min(axis=(0, 1)))
        print("[_get_bbox_tree] raw spts max =", spts.max(axis=(0, 1)))

        # Compute the bounding boxes
        smin, smax = spts.min(axis=0), spts.max(axis=0)

        print("[_get_bbox_tree] number of elements =", len(smin))
        print("[_get_bbox_tree] bbox global min before expand =", smin.min(axis=0))
        print("[_get_bbox_tree] bbox global max before expand =", smax.max(axis=0))

        # Expand by the scale factor to better account for strong curvature
        expand = np.abs((0.5*(scale - 1))*(smin + smax))
        smin -= expand
        smax += expand

        print("[_get_bbox_tree] bbox global min after expand =", smin.min(axis=0))
        print("[_get_bbox_tree] bbox global max after expand =", smax.max(axis=0))

        # Insert these boxes into a spatial index
        return Index((np.arange(len(smin)), smin, smax),
                    properties=Property(dimension=self.mesh.ndims))

    def _find_closest_node(self, pts):
        comm, rank, root = get_comm_rank_root()

        # Query the node index to find our closest node
        nodes, off, tree = self._get_nodes_off_tree()
        nearest = tree.nearest_v(pts, pts, strict=True)[0]

        buf = np.empty(len(pts), dtype=[('dist', float), ('idx', int)])
        buf['dist'] = np.linalg.norm(pts - nodes[nearest], axis=1)
        buf['idx'] = nearest + off

        self._minloc(comm.Allreduce, mpi.IN_PLACE, buf)

        return buf
    def _debug_print_element_geometry(self, etype, ei, label='candidate'):
        """Print geometry for one local element index.

        Notes
        -----
        self.mesh.spts[etype][:, ei, :] are the physical solution/support
        point coordinates used by the locator.

        self.mesh.spts_nodes[etype][ei] are the mesh node IDs associated
        with the element, when available.
        """
        spts = self.mesh.spts[etype][:, ei, :]

        bbox_min = spts.min(axis=0)
        bbox_max = spts.max(axis=0)

        eidxs_global = self.mesh.eidxs[etype]
        global_ei = eidxs_global[ei]

        print(f"        [{label} element geometry]")
        print("            etype              =", etype)
        print("            element local idx  =", ei)
        print("            element global idx =", global_ei)

        print("            bbox min from spts =", bbox_min)
        print("            bbox max from spts =", bbox_max)

        print("            solution/support points:")
        for j, x in enumerate(spts):
            print("                spt", j, "=", x)

        # Mesh-node connectivity, if available
        if hasattr(self.mesh, 'spts_nodes') and etype in self.mesh.spts_nodes:
            try:
                node_ids = self.mesh.spts_nodes[etype][ei]

                print("            mesh node ids =", node_ids)

                # Raw node locations, if node IDs index into mesh.raw['nodes']
                try:
                    node_locs = self.mesh.raw['nodes'][node_ids]['location']

                    print("            mesh node locations:")
                    for j, (nid, x) in enumerate(zip(node_ids, node_locs)):
                        print("                node", j, "global id =", nid, "loc =", x)
                except Exception as e:
                    print("            could not print raw node locations:", repr(e))

            except Exception as e:
                print("            could not print spts_nodes:", repr(e))
        else:
            print("            no mesh.spts_nodes available for this etype")

    def _find_closest_element_nnode(self, etype, pts, nearest):
        print(f"\n[_find_closest_element_nnode] etype = {etype}")

        nodes = self.mesh.spts_nodes[etype]

        # See which of our elements contain the nearest node
        eidx, nidx = np.isin(nodes, nearest['idx']).nonzero()

        print("[_find_closest_element_nnode] matching element-node pairs =",
            len(eidx))

        # Create a map from node number to element indices
        neles = defaultdict(set)
        for ei, ni in zip(eidx, nodes[eidx, nidx]):
            neles[ni].add(ei)

        print("[_find_closest_element_nnode] unique nearest nodes with local elems =",
            len(neles))

        # Useful for printing nearest node physical locations
        raw_nodes = self.mesh.raw['nodes']

        # Use this to form the set of candidate elements for each point
        pidx, sidx = [], []

        for i, (di, ni) in enumerate(nearest):
            elems = sorted(neles.get(ni, []))

            print("\n    [_find_closest_element_nnode/point]")
            print("        local point idx =", i)
            print("        query pt        =", pts[i])
            print("        nearest node idx =", ni)
            print("        nearest node dist =", di)

            # Print nearest-node physical location
            try:
                nloc = raw_nodes[ni]['location']
                print("        nearest node loc =", nloc)
                print("        vector query - node =", pts[i] - nloc)
            except Exception as e:
                print("        could not read nearest node location:", repr(e))

            print("        attached local elems on this rank =", len(elems))

            if not elems:
                print("        NO attached elements for this nearest node on this rank")

            # Print geometry for every candidate element attached to this nearest node
            for ei in elems:
                self._debug_print_element_geometry(
                    etype,
                    ei,
                    label='nearest-node candidate'
                )

                pidx.append(i)
                sidx.append(ei)

        print("[_find_closest_element_nnode] candidate point-element pairs =",
            len(pidx))

        return self._find_closest_element(etype, pts, pidx, sidx)

    def _find_closest_element_bbox(self, etype, pts):
        print(f"\n[_find_closest_element_bbox] etype = {etype}")
        print("[_find_closest_element_bbox] npts =", len(pts))

        # Query the index to find intersecting elements
        tree = self._get_bbox_tree(etype)
        sidx, icounts = tree.intersection_v(pts, pts)

        print("[_find_closest_element_bbox] total candidate hits =", len(sidx))

        if len(icounts):
            print("[_find_closest_element_bbox] icounts min =", icounts.min())
            print("[_find_closest_element_bbox] icounts max =", icounts.max())
            print("[_find_closest_element_bbox] points with zero bbox hits =",
                np.count_nonzero(icounts == 0))

            zero = np.where(icounts == 0)[0]
            if len(zero):
                print("[_find_closest_element_bbox] first zero-hit point indices =", zero[:20])
                for i in zero[:10]:
                    print("    zero-hit idx =", i, "pt =", pts[i])

            nonzero = np.where(icounts > 0)[0]
            if len(nonzero):
                print("[_find_closest_element_bbox] first nonzero-hit point indices =", nonzero[:10])
                for i in nonzero[:5]:
                    print("    nonzero-hit idx =", i,
                        "pt =", pts[i],
                        "nhits =", icounts[i])

        pidx = np.repeat(np.arange(len(pts)), icounts.astype(int)).tolist()
        sidx = sidx.tolist()

        print("[_find_closest_element_bbox] len(pidx) =", len(pidx))
        print("[_find_closest_element_bbox] len(sidx) =", len(sidx))

        out = self._find_closest_element(etype, pts, pidx, sidx)

        hit_points = set(pidx)
        located_points = set(out)
        zero_hit_points = set(np.where(icounts == 0)[0])
        hit_but_failed_points = sorted(hit_points - located_points)

        print("[_find_closest_element_bbox] located finite points for this etype =",
            len(out))
        print("[_find_closest_element_bbox] zero-hit points =",
            len(zero_hit_points))
        print("[_find_closest_element_bbox] hit-but-all-candidates-failed points =",
            len(hit_but_failed_points))

        if hit_but_failed_points:
            print("[_find_closest_element_bbox] first hit-but-failed point indices =",
                hit_but_failed_points[:20])
            for i in hit_but_failed_points[:10]:
                print("    hit-but-failed idx =", i, "pt =", pts[i])

        return out

    def _find_closest_element(self, etype, pts, pidx, sidx):
        print(f"\n[_find_closest_element] etype = {etype}")
        print("[_find_closest_element] candidate point-element pairs =", len(pidx))

        if len(pidx) == 0:
            print("[_find_closest_element] no candidates; returning empty")
            return {}

        spts_all = self.mesh.spts[etype]
        eidxs_global = self.mesh.eidxs[etype]

        print("[_find_closest_element] first candidate pairs:")
        for pi, ei in list(zip(pidx, sidx))[:10]:
            print("    point idx =", pi,
                "element local idx =", ei,
                "element global idx =", eidxs_global[ei],
                "pt =", pts[pi])

        # Obtain the closest location inside each candidate element
        dists, tlocs = self._compute_tlocs(
            etype,
            spts_all[:, sidx],
            pts[pidx],
            pidx=pidx,
            sidx=sidx
        )

        finite = np.isfinite(dists)
        print("[_find_closest_element] finite candidate dists after _compute_tlocs =",
            np.count_nonzero(finite))
        print("[_find_closest_element] inf candidate dists after _compute_tlocs =",
            np.count_nonzero(~finite))

        if np.any(finite):
            print("[_find_closest_element] finite dist min/max =",
                dists[finite].min(), dists[finite].max())

        # Group candidate results by original point index
        by_point = defaultdict(list)
        for cand_i, (pi, ei, dist, tloc) in enumerate(zip(pidx, sidx, dists, tlocs)):
            by_point[pi].append((cand_i, ei, dist, tloc))

        print("[_find_closest_element] unique points with bbox candidates =",
            len(by_point))

        # Summarize candidate status per point
        out = {}
        failed_points = []

        for pi, cand_list in by_point.items():
            finite_cands = [c for c in cand_list if np.isfinite(c[2])]
            inf_cands = [c for c in cand_list if not np.isfinite(c[2])]

            print("    [_find_closest_element/point-summary]")
            print("        point idx      =", pi)
            print("        pt             =", pts[pi])
            print("        num candidates =", len(cand_list))
            print("        finite cands   =", len(finite_cands))
            print("        failed cands   =", len(inf_cands))

            if finite_cands:
                # Pick finite candidate with smallest distance
                cand_i, ei, dist, tloc = min(finite_cands, key=lambda c: c[2])

                out[pi] = (dist, ei, tloc)

                print("        SELECTED candidate")
                print("            candidate idx      =", cand_i)
                print("            element local idx  =", ei)
                print("            element global idx =", eidxs_global[ei])
                print("            dist               =", dist)
                print("            tloc               =", tloc)
            else:
                # All candidate elements failed; do not return this point
                failed_points.append(pi)

                print("        NO FINITE CANDIDATE FOUND")
                print("        failed candidates with geometry:")

                for cand_i, ei, dist, tloc in inf_cands[:10]:
                    print("            candidate idx      =", cand_i)
                    print("            element local idx  =", ei)
                    print("            element global idx =", eidxs_global[ei])
                    print("            dist               =", dist)
                    print("            tloc               =", tloc)

                    self._debug_print_element_geometry(
                        etype,
                        ei,
                        label='failed candidate'
                    )

        print("[_find_closest_element] returning finite located points =", len(out))

        if failed_points:
            print("[_find_closest_element] points with candidates but all failed =",
                failed_points[:50])
            for pi in failed_points[:10]:
                print("    failed point idx =", pi, "pt =", pts[pi])

        return out

    def _find_closest_elementdelete(self, etype, pts, pidx, sidx):
        print(f"\n[_find_closest_element] etype = {etype}")
        print("[_find_closest_element] candidate point-element pairs =", len(pidx))

        if len(pidx) == 0:
            print("[_find_closest_element] no candidates; returning empty")
            return {}

        print("[_find_closest_element] first candidate pairs:")
        for pi, ei in list(zip(pidx, sidx))[:10]:
            print("    point idx =", pi, "element local idx =", ei, "pt =", pts[pi])

        spts = self.mesh.spts[etype]

        # Obtain the closest location inside each of these elements
        dists, tlocs = self._compute_tlocs(etype, spts[:, sidx], pts[pidx])

        print("[_find_closest_element] finite dists after _compute_tlocs =",
            np.count_nonzero(np.isfinite(dists)))
        print("[_find_closest_element] inf dists after _compute_tlocs =",
            np.count_nonzero(np.isinf(dists)))

        if len(dists):
            finite = np.isfinite(dists)
            if np.any(finite):
                print("[_find_closest_element] finite dist min/max =",
                    dists[finite].min(), dists[finite].max())

            bad = np.where(np.isinf(dists))[0]
            if len(bad):
                print("[_find_closest_element] first invalid candidate indices =",
                    bad[:20])
                for j in bad[:10]:
                    print("    candidate j =", j,
                        "point idx =", pidx[j],
                        "element local idx =", sidx[j],
                        "pt =", pts[pidx[j]],
                        "tloc =", tlocs[j])

        # For each query point identify the most promising element
        closest = {}
        for i, (pi, dist, tloc) in enumerate(zip(pidx, dists, tlocs)):
            if pi not in closest or dist < closest[pi][0]:
                closest[pi] = (dist, i)

        pidx = list(closest)
        tidx = [i for d, i in closest.values()]
        sidx = [sidx[i] for i in tidx]

        out = dict(zip(pidx, zip(dists[tidx], sidx, tlocs[tidx])))

        print("[_find_closest_element] returning located points =", len(out))

        return out

    def _initial_tlocs(self, etype, spts, plocs):
        shape, basis = self._get_shape_basis(etype, len(spts))
        tpts = np.array(shape.std_ele(self.fine_order))

        # Obtain a fine sampling of points inside each element
        fop = basis.nodal_basis_at(tpts)
        fpts = fop @ spts.reshape(len(spts), -1)
        fpts = fpts.reshape(len(fop), *spts.shape[1:])

        # Find the closest fine sample point to each query point
        dists = np.linalg.norm(fpts - plocs, axis=2)

        # Return this sample point in transformed space
        return tpts[dists.argmin(axis=0)]

    def _compute_tlocs(self, etype, spts, plocs, pidx=None, sidx=None):
        shape, basis = self._get_shape_basis(etype, len(spts))

        ncands = len(plocs)

        print(f"\n[_compute_tlocs] etype = {etype}")
        print("[_compute_tlocs] ncandidates =", ncands)
        print("[_compute_tlocs] spts shape =", spts.shape)
        print("[_compute_tlocs] plocs shape =", plocs.shape)

        if ncands == 0:
            return np.empty(0), np.empty((0, self.mesh.ndims))

        # Evaluate the initial guesses
        ktlocs = self._initial_tlocs(etype, spts, plocs)
        kplocs = np.einsum('ij,jik->ik',
                        basis.nodal_basis_at(ktlocs, clean=False), spts)

        init_dists = np.linalg.norm(kplocs - plocs, axis=1)

        print("[_compute_tlocs] initial dist min/max =",
            init_dists.min(), init_dists.max())

        failed_newton = np.zeros(ncands, dtype=bool)

        # Apply three iterations of Newton's method
        for k in range(3):
            jac_ops = basis.jac_nodal_basis_at(ktlocs, clean=False)

            A = np.einsum('ijk,jkl->kli', jac_ops, spts)
            b = kplocs - plocs

            try:
                delta = np.linalg.solve(A, b[..., None]).squeeze()
            except np.linalg.LinAlgError as e:
                print("[_compute_tlocs] Newton solve failed globally")
                print("[_compute_tlocs] iter =", k)
                print("[_compute_tlocs] error =", e)

                # Fall back to marking everything as failed
                dists = np.full(ncands, np.inf)
                return dists, ktlocs

            ktlocs -= delta

            ops = basis.nodal_basis_at(ktlocs, clean=False)
            np.einsum('ij,jik->ik', ops, spts, out=kplocs)

            iter_dists = np.linalg.norm(kplocs - plocs, axis=1)

            print(f"[_compute_tlocs] Newton iter {k}")
            print("    delta norm min/max =",
                np.linalg.norm(delta, axis=1).min(),
                np.linalg.norm(delta, axis=1).max())
            print("    dist min/max =",
                iter_dists.min(), iter_dists.max())

            bad = ~np.all(np.isfinite(ktlocs), axis=1)
            if np.any(bad):
                failed_newton |= bad
                print("    non-finite tloc count =", np.count_nonzero(bad))

        # Compute the final distances
        dists_raw = np.linalg.norm(kplocs - plocs, axis=1)
        dists = dists_raw.copy()

        invalid_spt = np.zeros(ncands, dtype=bool)
        nonfinite = ~np.isfinite(dists_raw) | ~np.all(np.isfinite(ktlocs), axis=1)

        # Prune invalid points
        for i, t in enumerate(ktlocs):
            if nonfinite[i] or not shape.valid_spt(t):
                invalid_spt[i] = True
                dists[i] = np.inf

        finite = np.isfinite(dists)

        print("[_compute_tlocs] final raw dist finite count =",
            np.count_nonzero(np.isfinite(dists_raw)))
        print("[_compute_tlocs] final accepted finite count =",
            np.count_nonzero(finite))
        print("[_compute_tlocs] final rejected count =",
            np.count_nonzero(~finite))
        print("[_compute_tlocs] rejected by nonfinite count =",
            np.count_nonzero(nonfinite))
        print("[_compute_tlocs] rejected by valid_spt/nonfinite count =",
            np.count_nonzero(invalid_spt))

        if np.any(finite):
            print("[_compute_tlocs] accepted dist min/max =",
                dists[finite].min(), dists[finite].max())

        # Print accepted examples
        good = np.where(finite)[0]
        if len(good):
            print("[_compute_tlocs] first accepted candidates:")
            for j in good[:10]:
                print("    candidate j =", j)
                if pidx is not None:
                    print("        point idx =", pidx[j])
                if sidx is not None:
                    print("        element local idx =", sidx[j])
                print("        ploc =", plocs[j])
                print("        tloc =", ktlocs[j])
                print("        mapped ploc =", kplocs[j])
                print("        raw dist =", dists_raw[j])

        # Print rejected examples
        bad = np.where(~finite)[0]
        if len(bad):
            print("[_compute_tlocs] first rejected candidates:")
            for j in bad[:20]:
                print("    candidate j =", j)
                if pidx is not None:
                    print("        point idx =", pidx[j])
                if sidx is not None:
                    print("        element local idx =", sidx[j])
                print("        ploc =", plocs[j])
                print("        tloc =", ktlocs[j])
                print("        mapped ploc =", kplocs[j])
                print("        raw dist =", dists_raw[j])
                print("        nonfinite =", bool(nonfinite[j]))
                print("        valid_spt =", False if nonfinite[j] else shape.valid_spt(ktlocs[j]))

        return dists, ktlocs


class PointSampler:
    def __init__(self, mesh, spts, slocs=None):
        locf = ['cidx', 'eidx', 'tloc']
        self.mesh = mesh

        # Named point set
        if isinstance(spts, str):
            comm, rank, root = get_comm_rank_root()

            if rank == root:
                sinfo = mesh.raw[f'plugins/sampler/{spts}'][:]
            else:
                sinfo = None

            sinfo = comm.bcast(sinfo, root=root)

            self.pts, self.locs = sinfo['ploc'], sinfo[locf]
        # Points with location data
        elif slocs is not None:
            self.pts, self.locs = spts, slocs[locf]
        # Points without location data
        else:
            self.pts = np.array(spts)
            self.locs = PointLocator(mesh).locate(self.pts)[locf]

    def configure_with_intg_nvars(self, intg, nvars):
        # Get the solution bases from the system
        ubases = {etype: eles.basis.ubasis
                  for etype, eles in intg.system.ele_map.items()}

        self._configure_ubases_nvars(ubases, nvars)

    def configure_with_cfg_nvars(self, cfg, nvars):
        ubases = {}

        for etype in self.mesh.eidxs:
            shapecls = subclass_where(BaseShape, name=etype)
            ubases[etype] = shapecls(None, cfg).ubasis

        self._configure_ubases_nvars(ubases, nvars)

    def _configure_ubases_nvars(self, ubases, nvars):
        self.nvars = nvars
        locs = self.locs

        comm, rank, root = get_comm_rank_root()
        ptsrank, pinfo = [], defaultdict(list)

        for j, (etype, eidxs) in enumerate(self.mesh.eidxs.items()):
            eimap = np.argsort(eidxs)
            ubasis = ubases[etype]

            # Filter points which do not belong to this element type
            elocs = locs['cidx'] == self.mesh.codec.index(f'eles/{etype}')
            elocs = elocs.nonzero()[0]

            # See what points we have
            esrch = np.searchsorted(eidxs, locs[elocs]['eidx'], sorter=eimap)
            for i, k, l in zip(elocs, esrch, locs[elocs]):
                if k < eimap.size and eidxs[eimap[k]] == l['eidx']:
                    op = ubasis.nodal_basis_at(l['tloc'][None], clean=False)

                    pinfo[j, eimap[k]].append((len(ptsrank), op))
                    ptsrank.append(i)

        # Group points according to the element they're inside
        self.pinfo, self.pcount = [], len(ptsrank)
        for (et, ei), info in pinfo.items():
            if len(info) == 1:
                self.pinfo.append((et, ei, *info[0]))
            else:
                idxs, ops = zip(*info)
                self.pinfo.append((et, ei, np.array(idxs), np.vstack(ops)))

        # Tell the root rank which points we are responsible for
        ptsrank = comm.gather(ptsrank, root=root)

        if rank == root:
            # Allocate a buffer to store the sampled points
            self._ptsbuf = ptsbuf = np.empty((len(self.pts), nvars))

            # Compute the counts and displacements, sans nvars
            ptscounts = np.array([len(pr) for pr in ptsrank])
            ptsdisps = np.concatenate(([0], np.cumsum(ptscounts[:-1])))

            if ptscounts.sum() != len(self.pts):
                raise RuntimeError('Missing points in solution')

            # Form the MPI Gatherv receive buffer tuple
            self._ptsrecv = (ptsbuf, (nvars*ptscounts, nvars*ptsdisps))

            # Form the reordering list
            self._ptsinv = np.argsort([i for pr in ptsrank for i in pr])

    def sample(self, solns, process=None):
        comm, rank, root = get_comm_rank_root()

        # Perform the sampling
        samples = np.empty((self.pcount, self.nvars))
        for et, ei, idxs, ops in self.pinfo:
            samples[idxs] = ops @ solns[et][:, :, ei]

        # Post-process the samples
        if process:
            samples = np.ascontiguousarray(process(samples))

        # Gather to the root rank and return
        if rank == root:
            comm.Gatherv(samples, self._ptsrecv, root=root)
            return self._ptsbuf[self._ptsinv]
        else:
            comm.Gatherv(samples, None, root=root)
            return None
