from sklearn.neighbors import NearestNeighbors
from collections import defaultdict, deque

# ---------------- Shared neighbor & voting utilities ----------------
def _neighbors_within_islet(adata, i_global, group_key, coord_key, k, r_auto, min_need):
    '''
    Find the closest neighbors of a given spot, restricted to the same islet.
    
    Parameters
    ----------
    adata : AnnData
        The spatial dataset containing `.obs` metadata and `.obsm[coord_key]` coordinates.
    i_global : int
        The global index (row number in `adata.obs_names`) of the focal spot.
    group_key : str
        Column in `.obs` defining islet membership (e.g. "islet").
    coord_key : str
        Key in `.obsm` containing spatial coordinates (e.g. "spatial").
    k : int or None
        Maximum number of nearest neighbors to return (cap). If None, only radius is used.
    r_auto : float or None
        Precomputed radius for this islet. If None, radius is estimated automatically
        as ≈ 2.5 × median 2nd-nearest-neighbor spacing within the islet.
    min_need : int
        Minimum number of neighbors desired. Used to ensure we query enough
        candidates even if k is small.

    Returns
    -------
    gi : ndarray of int
        Global indices of all spots in the same islet as the focal spot.
    ind : ndarray of int
        Local indices (into `Xi`) of the chosen neighbors of the focal spot,
        after applying radius and k cap.
    Xi : ndarray of shape (n_islet, 2)
        Coordinates of all spots in the islet.
    i_local : int
        Local index of the focal spot within `Xi`.
    r_use : float or None
        Radius actually used for filtering neighbors of this spot.
    g : str
        The islet ID of the focal spot.
    '''
    obs = adata.obs
    coords = np.asarray(adata.obsm[coord_key], dtype=float) # coordiates of all spots
    g = obs.iloc[i_global][group_key] # islet ID of the spot
    gi = np.where(obs[group_key].astype(str).values == str(g))[0] # all global indices of spots in that islet
    if gi.size < 2: # avoid islet with only 1 spot
        return gi, np.array([], dtype=int), None, None, None, g

    Xi = coords[gi] # coordinates of spots in that islet
    loc = np.where(gi == i_global)[0] # the global index of a chosen spot
    if loc.size == 0: # the chosen spot’s index wasn’t found inside gi
        return gi, np.array([], dtype=int), None, None, None, g
    i_local = loc[0] # the position of the spot within its islet’s local coordinate array

    # auto radius per-islet if None: ≈ 2.5 × median 2nd-NN spacing
    if r_auto is None: # if no spefified radius
        # For each spot in Xi, find its 2 nearest neighbors (n_neighbors=3 --> itself + its 2 closest neighbors).
        nn_small = NearestNeighbors(n_neighbors=min(3, len(gi))).fit(Xi)
        d, _ = nn_small.kneighbors(Xi) # distances to those neighbors
        spacing = float(np.median(d[:, 1])) if Xi.shape[0] >= 3 else 0.0 # take the 2nd smallest distance for each spot
        r_use = 2.5 * spacing if spacing > 0 else None
    else:
        r_use = float(r_auto)

    # build larger NN to be safe; we’ll filter by radius then cap to k
    want = (k or min_need) + 1
    nn = NearestNeighbors(n_neighbors=min(max(want, 15), len(gi))).fit(Xi)
    d, ind = nn.kneighbors(Xi[i_local:i_local+1], n_neighbors=min(want, len(gi)))
    d = d.ravel()[1:]; ind = ind.ravel()[1:]  # drop self

    if r_use is not None:
        keep = d <= r_use
        d = d[keep]; ind = ind[keep]

    if (k is not None) and (ind.size > k):
        d = d[:k]; ind = ind[:k]

    return gi, ind, Xi, i_local, r_use, g


def _gauss_weights(Xi, idxs, center, sigma):
    """
    Compute Gaussian distance weights for neighbors of a focal spot.

    This models distance effects so that closer neighbors have stronger influence,
    ensuring relabeling decisions are local in a biological sense.

    Parameters
    ----------
    Xi : ndarray of shape (n_spot_in_islet, 2)
        Coordinates of all spots in the islet.
    idxs : ndarray of int
        Local indices (into Xi) of the neighbor spots.
    center : ndarray of shape (2,)
        Coordinates of the focal spot (or cluster centroid).
    sigma : float
        Characteristic length scale (e.g. typical bead spacing).
        We often use r_auto / 2.5, where r_auto ≈ 2.5 × spacing.

    Returns
    -------
    weights : ndarray of float, shape (len(idxs),)
        Distance-based weights in [0, 1], decaying with distance
        according to a Gaussian kernel.
    """
    if sigma is None or sigma <= 0 or len(idxs) == 0:
        return np.ones(len(idxs), dtype=float)
    d = np.linalg.norm(Xi[idxs] - center, axis=1) # Euclidean distance from focal spot
    return np.exp(-(d**2) / (2.0 * sigma**2))     # Gaussian kernel -- w = exp(-d^2 / (2σ^2))

def _vote_counts(
    neigh_labels,
    parts,
    *, # All arguments after this point must be passed by keyword, not position
    neigh_ids=None,          
    frac=None,            
    proportional_mixed: bool = False,
    sep: str = "_",
    laplace: float = 0.5,
    weight_mixed: float = 0.5,
    exclude_threeway: bool = False,
    threeway_weight: float = 0.2,
    dweights=None,
):
    """
    Aggregate neighbor labels into weighted votes over a given set of classes.

    Each neighbor contributes to the classes present in its label. Pure labels
    (e.g., 'α') contribute a full vote; mixed labels (e.g., 'α_β') split their
    contribution evenly (by default), optionally down-weighting 3-way mixes.
    Distance weights (dweights) multiply each neighbor's contribution.

    Parameters
    ----------
    neigh_labels : Sequence[str]
        Labels of neighbors (some single, some mixed like 'α_β' or 'α_β_δ').
    parts : Sequence[str]
        The classes to vote over (e.g., ['α','β'] or ['α','β', 'δ']).
    neigh_ids : Sequence[str]
        obs_names of the neighbors (same order as neigh_labels)
    frac : DataFrame (rows=spot_id, cols=classes)
        DataFrame with normalized fractions of ('α','β', 'γ', 'δ') per spot (rows=spot_id, cols=classes) 
    proportional_mixed : bool, default False
        If True and frac_df available, split mixed votes by fractions
    sep : str, default '_'
        Separator used in labels to denote mixtures.       
    laplace : float, default 0.5
        Laplace smoothing added to every class count to avoid zeros.
    weight_mixed : float, default 0.5
        Total weight assigned to a mixed neighbor before splitting across its parts.
        (Pure neighbors always contribute 1.0 × distance × type_weight.)
    exclude_threeway : bool, default False
        If True, neighbors with 3 or more parts are ignored entirely.
    threeway_weight : float, default 0.2
        If not excluding 3-way, multiply their contribution by this factor.
    dweights : array-like or None
        Per-neighbor distance weights (from _gauss_weights). If None, all 1.0.

    Returns
    -------
    dict[str, float]
        Mapping {class -> weighted vote}. A dictionary mapping each class to its weighted vote count.
    """
    # Set of classes we are voting over (e.g., {"α","β"} or {"α","β","δ"})
    P = set(parts) 
    
    # Initialize counts for each class with Laplace smoothing
    # (prevents any class from staying at exactly zero)
    cnt = {p: float(laplace) for p in parts}
    
    # If no distance weights provided, treat all neighbors equally
    if dweights is None:
        dweights = np.ones(len(neigh_labels), dtype=float)
    
    # Flag: can we use frac to split mixed votes proportionally?
    use_frac = proportional_mixed and (frac is not None) and (neigh_ids is not None)

    # Loop over neighbors with their corresponding distance weighs
    for idx, (nl, w_phys) in enumerate(zip(neigh_labels, dweights)):
        # Split the neighbor label by sep (e.g., "α_β" -> ["α","β"])
        # Keep only tokens that are in the parts we care about
        toks = [t for t in nl.split(sep) if t in P]
        if not toks:
            continue # skip neighbors with no relevant classes
            
        # 3+ mix handling (e.g., "α_β_δ")
        if nl.count(sep) >= 2: # how many times the separator '-' appears in the label
            if exclude_threeway:
                continue # skip 3-way neighbors entirely
            type_w = threeway_weight # otherwise downweight them
        else:
            type_w = 1.0 # pure or 2-way mixes keep full weight

        # Case 1: neighbor is pure (single class, e.g., "α")
        if len(toks) == 1:
            cnt[toks[0]] += 1.0 * w_phys * type_w
            
        # Case 2: neighbor is mixed (2-way or 3-way)
        else:
            if use_frac:
                 # If proportional mode is enabled, use frac to split votes
                sid = neigh_ids[idx]
                if sid in frac.index:
                    # Get this neighbor’s fractions, restricted to our parts
                    f = frac.loc[sid, list(P)].fillna(0.0)
                    s = float(f[toks].sum()) # total of relevant tokens
                    if s > 0:
                        for t in toks:
                            # Contribution ∝ fraction of that token
                            cnt[t] += (weight_mixed * (f[t] / s)) * w_phys * type_w
                        continue # done with this neighbor, skip equal-split fallback
                        
            # fallback: equal split across toks
            split = (weight_mixed / len(toks)) * w_phys * type_w
            for t in toks:
                cnt[t] += split
                
    # Return dictionary: {class → weighted vote count}
    return cnt

def wilson_lower(p_hat: float, n: float, z: float = 1.64) -> float:
    """
    Lower bound of Wilson score interval for a binomial proportion.

    Parameters
    ----------
    p_hat : float
        Observed proportion in [0,1].
    n : float
        (Effective) sample size. Must be > 0.
    z : float
        Normal z for two-sided CI: ~1.28 (80%), 1.64 (90%), 1.96 (95%), 2.58 (99%).

    Returnsd
    -------
    float
        Wilson lower confidence bound.
    """
    n = max(float(n), 1.0)
    z2 = z * z
    denom  = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    rad    = z * np.sqrt((p_hat * (1.0 - p_hat) / n) + (z2 / (4.0 * n * n)))
    return (center - rad) / denom


def kish_effective_n(weights: np.ndarray) -> float:
    """
    Kish effective sample size for (possibly) unequal weights.

    n_eff = (sum w)^2 / sum(w^2)

    If all weights are 1, n_eff = number of neighbors.
    """
    w = np.asarray(weights, dtype=float)
    if w.size == 0:
        return 0.0
    s1 = np.sum(w)
    s2 = np.sum(w * w)
    return 0.0 if s2 == 0 else (s1 * s1) / s2

def _normalize_indices(candidates_idx, n_total):
    """Return a 1-D numpy array of integer indices to iterate over."""
    if candidates_idx is None:
        return np.arange(n_total, dtype=int)
    return np.asarray(candidates_idx, dtype=int).ravel()

def _pass_2way(
    adata, labels, 
    *,  # keyword-only from here
    mode,                           # "smooth" or "sharpen"
    force_resolve_set=("δ","γ"),    # any mix touching these is always relabeled
    frac=frac, proportional_mixed=True,
    sep="_", group_key="islet", coord_key="spatial",
    k=10, radius=None, min_neighbors=8, min_neighbors_lo=3,
    majority_thresh=0.50, majority_thresh_lo=2/3,
    use_wilson=False, z_wilson=1.64,
    laplace=0.5, weight_mixed=0.5, use_distance_weights=True,
    min_anchor_pure=3,
    candidates_idx=None
):
    """
    Run a relabeling pass for 2-way mixtures.
    
    Depending on the mode:
        - If mode == "smooth": encourage local homogeneity → assign to majority class.
        - If mode == "sharpen": preserve local complexity → assign to minority class.

    Parameters
    ----------
    adata : AnnData
        Spatial dataset containing .obs[group_key] and .obsm[coord_key].
    labels : pd.Series
        Current labels for all spots (mutable copy).
    mode : {"smooth","sharpen"}
        Strategy to apply.
    force_resolve_set: set (e.g. ("δ","γ"))
        any mix touching these is always relabeled
    frac : DataFrame (rows=spot_id, cols=classes)
        DataFrame with normalized fractions of ('α','β', 'γ', 'δ') per spot (rows=spot_id, cols=classes) 
    proportional_mixed : bool, default False
        If True and frac available, split mixed votes by fractions
    sep : str, default "_"
        Separator used in labels for mixtures.
    group_key : str, default "islet"
        Column in .obs that defines islet membership.
    coord_key : str, default "spatial"
        Key in .obsm containing spatial coordinates.
    k : int, default 10
        Maximum number of neighbors to consider (cap).
    radius : float or None
        Radius to use; if None, auto-computed per islet.
    min_neighbors : int, default 8
        Typical neighbor count required for normal thresholding.
    min_neighbors_lo : int, default 3
        Minimum neighbors for a decision (stricter threshold used).
    majority_thresh : float, default 0.5
        Fraction threshold for majority decisions (normal).
    majority_thresh_lo : float, default 2/3
        Fraction threshold for majority decisions (low-N).
    use_wilson: bool, default True
        whether to apply wilson threshold.
    z_wilson : float, default 1.64
        Normal z for two-sided CI: ~1.28 (80%), 1.64 (90%), 1.96 (95%), 2.58 (99%).
    laplace : float, default 0.5
        Laplace smoothing to avoid zero votes.
    weight_mixed : float, default 0.5
        Total weight assigned to a mixed neighbor before splitting votes.
    use_distance_weights : bool, default True
        Whether to apply Gaussian distance weighting.
    min_anchor_pure : int
        Require at least this many anchor neighbors (pure/2-way) to act in 3-way.

    Returns
    -------
    labels : pd.Series
        Updated labels after relabeling pass.
    changes : pd.DataFrame
        Records of changes with columns [spot, old, new, reason, frac, nN].
    unchanges : pd.DataFrame
        Records of changes with columns [spot, old, new, reason, frac, nN].
    """
    if mode not in {"smooth", "sharpen"}:
        raise ValueError(f"Invalid mode={mode!r}. Choose 'smooth' or 'sharpen'.")
        
    # normalize once at the top
    if force_resolve_set is None:
        force_resolve_set = frozenset()          # empty, cheap
    elif not isinstance(force_resolve_set, (set, frozenset)):
        force_resolve_set = frozenset(force_resolve_set)
    
    idx = adata.obs_names  # all spot IDs
    changes = []  # collect relabeling events
    no_changes = []  # collect not relabeling events

     # ↓↓↓ iterate only candidates if provided
    iter_indices = _normalize_indices(candidates_idx, len(idx))
    
    # Iterate over all spots
    for i_global in iter_indices:
        lab = labels.iat[i_global]               # current label (string, e.g., "α_β")
        parts = [p for p in lab.split(sep) if p] # split into component classes
        if len(parts) != 2:                      # only handle 2-way mixes here
            continue

        a, b = parts
        pair_sorted = sep.join(sorted([a, b]))
        pair = frozenset(parts) # immutable set (cannot be changed once created)
        touches_complex = bool(set(parts) & force_resolve_set) # if must relabelled cell type involved
        
        # Defaults for potential early exits
        nN_val = None
        anchors_val = None
        cluster_size_val = None
        frac_val = None
        
        # Get neighbors within the same islet
        gi, loc, Xi, i_local, r_use, _ = _neighbors_within_islet(
            adata, i_global, group_key, coord_key, k, radius, min_neighbors
        )
        if loc.size < min_neighbors_lo or Xi is None:
            no_changes.append({
                "spot": idx[i_global], "old": lab, "new": lab,
                "reason": "skip_neighbors_too_few_or_no_coords", "frac": None,
                "nN": int(loc.size) if hasattr(loc, "size") else None,
                "anchors": None, "cluster_size": None
            })
            continue  # too few neighbors or no coords

        nN_val = int(loc.size)
        
        # Focal spot coordinate
        center = Xi[i_local]
        # Define sigma for Gaussian weights (≈ spacing)
        sigma = (r_use / 2.5) if (use_distance_weights and r_use and r_use > 0) else None
        dweights = _gauss_weights(Xi, loc, center, sigma) if sigma else None

        # Get neighbor labels
        neigh_idx  = gi[loc]
        neigh_labs = labels.iloc[neigh_idx].tolist()
        neigh_ids  = adata.obs_names[neigh_idx].tolist()
        
        # ---- anchors (pure or 2-way of any pair) ----
        n_anchor = 0
        for nl in neigh_labs:
            c = nl.count(sep)
            if c == 0 or c == 1:
                n_anchor += 1
        if n_anchor < min_anchor_pure:
            no_changes.append({
                "spot": idx[i_global], "old": lab, "new": lab,
                "reason": "skip_not_enough_anchor_neighbors",
                "frac": None, "nN": nN_val, "anchors": anchors_val, "cluster_size": None
            })
            continue  # too flimsy to act
            
        # ---- quick local "cluster size" for this 2-way pair ----
        # count neighbors that share the SAME 2-way label (order-insensitive)
        same_pair = 0
        for nl in neigh_labs:
            if nl.count(sep) == 1:
                pa, pb = [p for p in nl.split(sep) if p]
                if sep.join(sorted([pa, pb])) == pair_sorted:
                    same_pair += 1
        cluster_size = 1 + same_pair  # include self
        cluster_size_val = int(cluster_size)

        # Count votes from neighbors for {a,b}
        cnt = _vote_counts(
            neigh_labs, [a, b],
            neigh_ids=neigh_ids, frac=frac, proportional_mixed=proportional_mixed,
            sep=sep, laplace=laplace, weight_mixed=weight_mixed,
            exclude_threeway=False, dweights=dweights
        )
        
        # Total votes for a+b
        total = cnt[a] + cnt[b]
        if total <= 0:
            no_changes.append({
                "spot": idx[i_global], "old": lab, "new": lab,
                "reason": "keep_no_votes",
                "frac": None, "nN": nN_val, "anchors": anchors_val,
                "cluster_size": cluster_size_val
            })
            continue

        # majority / minority & fraction
        if cnt[a] > cnt[b]:
            maj, mino = a, b
        elif cnt[b] > cnt[a]:
            maj, mino = b, a
        else:  # exact tie
            # tie-breaker prefers the complex member if present
            if a in force_resolve_set and b not in force_resolve_set:
                maj, mino = a, b
            elif b in force_resolve_set and a not in force_resolve_set:
                maj, mino = b, a
            else:
                # stable deterministic fallback
                maj, mino = sorted([a, b])  # alphabetical
        frac_maj = cnt[maj] / total # How strong the majority is
        frac_val = float(frac_maj)

        # Decision logic
        if touches_complex:
            # always relabel mixes involving δ/γ
            new_lab = maj if mode == "smooth" else mino
            if new_lab != lab:
                labels.iat[i_global] = new_lab
                changes.append({
                    "spot": idx[i_global], "old": lab, "new": new_lab,
                    "reason": f"2way_{mode}_force_complex",
                    "frac": float(frac_maj), "nN": int(loc.size),
                    "anchors": int(n_anchor), "cluster_size": cluster_size
                }) 
            else:
                # Already consistent with rule → no change, but record it
                no_changes.append({
                    "spot": idx[i_global], "old": lab, "new": lab,
                    "reason": "keep_complex_already_consistent",
                    "frac": frac_val, "nN": nN_val,
                    "anchors": anchors_val, "cluster_size": cluster_size_val
                })                
        else:
            # non-complex mixes (e.g., α_β): optional Wilson or fixed threshold
            if mode not in {"smooth","sharpen"}:
                raise ValueError(
                    f"Invalid mode={mode!r}. Please select strategy from {{'smooth','sharpen'}}."
                )

            if use_wilson:
                n_eff = kish_effective_n(dweights) if dweights is not None else float(loc.size)
                lb = wilson_lower(frac_maj, max(n_eff, 1.0), z=z_wilson)
                thr_base = 0.50  # simple majority among two classes
                if (lb - thr_base) > 1e-6:
                    new_lab = maj if mode == "smooth" else mino
                else:
                    new_lab = lab  # keep
                    no_changes.append({
                        "spot": idx[i_global], "old": lab, "new": lab,
                        "reason": "keep_wilson_not_significant",
                        "frac": frac_val, "nN": nN_val,
                        "anchors": anchors_val, "cluster_size": cluster_size_val
                    })
            else:
                thr = majority_thresh if loc.size >= min_neighbors else majority_thresh_lo
                if (frac_maj - thr) > 1e-6:
                    new_lab = maj if mode == "smooth" else mino
                else:
                    new_lab = lab
                    no_changes.append({
                        "spot": idx[i_global], "old": lab, "new": lab,
                        "reason": "keep_frac_below_threshold",
                        "frac": frac_val, "nN": nN_val,
                        "anchors": anchors_val, "cluster_size": cluster_size_val
                    })
            if new_lab != lab:
                labels.iat[i_global] = new_lab
                changes.append({
                    "spot": idx[i_global], "old": lab, "new": new_lab,
                    "reason": f"2way_{mode}" + ("_wilson" if use_wilson else ""),
                    "frac": float(frac_maj), "nN": int(loc.size),
                    "anchors": int(n_anchor), "cluster_size": cluster_size
                })
    changes_df = pd.DataFrame(changes, 
                              columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])
    no_changes_df = pd.DataFrame(no_changes, 
                                 columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])
    
    return labels, changes_df, no_changes_df

from collections import defaultdict, deque

# ---------------- 3-way joint passes ----------------
def _pass_3way_joint(
    adata, labels, 
    *, # keyword-only from here
    mode,                           # "smooth" or "sharpen"
    pairs_to_r,                     # iterable of pairs that define r in α_β_r (order-insensitive)
    # voting / neighbors
    frac=None, proportional_mixed=True,
    sep="_", group_key="islet", coord_key="spatial",
    k=10, radius=None, min_neighbors=8, min_neighbors_lo=3,
    majority_thresh=0.50, majority_thresh_lo=2/3,   # if you want fixed thr
    use_wilson=False, z_wilson=1.64,                # optional Wilson gate (smooth)
    cluster_radius_factor=0.5,
    laplace=0.5, weight_mixed=0.5,
    use_distance_weights=True,
    exclude_threeway=True, threeway_weight=0.2,
    min_anchor_pure=3,
    # tie-breaker when γ and δ are equal under sharpen fallback:
    sharpen_gamma_delta_priority=("δ","γ"),
    candidates_idx=None
):
    """
    Jointly relabel connected components of 3-way mixed spots.

    Behaviour
    ---------
    mode="smooth":
        - For each connected component (same trio, spatially adjacent), compute votes
          among the 3 parts; assign the component to the majority winner.
        - Optionally require Wilson lower bound > base threshold (use_wilson=True).

    mode="sharpen":
        - If trio matches α_β_r (i.e., pairs_to_r subset), assign to r.
        - Otherwise (trio contains γ and/or δ but not αβ together):
            pick argmax between {γ, δ} by neighbor votes (proportional if enabled).
            On tie, use deterministic priority (e.g., ('δ','γ')).

    Parameters
    ----------
    adata : AnnData
        Spatial data with obs[group_key] and obsm[coord_key].
    labels : pd.Series
        Current labels (mutable copy).
    mode : {"smooth","sharpen"}
        Strategy for 3-way handling.
    pairs_to_r : iterable[tuple[str,str]]
        Pairs that define a third member r in α_β_r trios (order-insensitive).
    frac : pd.DataFrame or None
        Per-spot absolute fractions; rows=spot_id, cols=classes (e.g. α,β,γ,δ).
    proportional_mixed : bool, default True
        Use `frac` to split mixed neighbor votes proportionally (else equal split).
    sep : str, default "_"
        Separator used in labels.
    group_key : str, default "islet"
        Islet column in obs.
    coord_key : str, default "spatial"
        2D coordinates in obsm.
    k : int, default 10
        Neighbor cap per spot.
    radius : float or None
        Neighborhood radius; if None, auto per-islet (≈ 2.5× median 2NN).
    min_neighbors : int, default 8
        Typical neighbor count; affects fixed-threshold branch in smooth mode.
    min_neighbors_lo : int, default 3
        Minimal neighbors to proceed.
    majority_thresh : float, default 0.5
        Fixed threshold for smooth mode (normal N) when not using Wilson.
    majority_thresh_lo : float, default 2/3
        Fixed threshold for smooth mode (low N) when not using Wilson.
    use_wilson : bool, default False
        If True (smooth mode), require Wilson lower bound over base threshold.
    z_wilson : float, default 1.64
        Normal z for two-sided CI: ~1.28 (80%), 1.64 (90%), 1.96 (95%), 2.58 (99%).
    cluster_radius_factor : float, default 0.5
        Component adjacency radius as factor of r_use.
    laplace, weight_mixed, use_distance_weights, exclude_threeway, threeway_weight
        Voting and weighting knobs passed to _vote_counts.
    weight_mixed : float, default 0.5
        Total weight assigned to a mixed neighbor before splitting votes.
    use_distance_weights : bool, default True
        Whether to apply Gaussian distance weighting.
    exclude_threeway : bool, default False
        If True, neighbors with 3 or more parts are ignored entirely.
    threeway_weight : float, default 0.2
        If not excluding 3-way, multiply their contribution by this factor.
    min_anchor_pure : int, default 3
        Require at least this many anchor neighbors (pure or trusted 2-way) to act.
    sharpen_gamma_delta_priority : set (e.g. ("δ","γ"))
        Tie-breaker when γ and δ are equal under sharpen fallback.

    Returns
    -------
    labels : pd.Series
        Updated labels after 3-way pass.
    changes : pd.DataFrame
        Log of changes with metadata (reason, frac, nN, anchors, cluster_size).
    """
    if mode not in {"smooth", "sharpen"}:
        raise ValueError(f"Invalid mode={mode!r}. Choose 'smooth' or 'sharpen'.")

    idx = adata.obs_names.to_numpy()
    pairs_to_r = {frozenset(p) for p in pairs_to_r}
    changes, no_changes = [], []

    # --- bucket 3-mix spots by (islet, trio) ---
    buckets = defaultdict(list)  # (islet, frozenset(parts)) -> list of (i_global, gi, Xi, i_local, r_use)
    trio_to_rpair = {}

    # --- bucket 3-mix spots only from candidates if provided ---
    iter_indices = _normalize_indices(candidates_idx, len(idx))
    for i_global in iter_indices:
        name = idx[i_global]
        lab = labels.iat[i_global]
        parts = [p for p in lab.split(sep) if p]
        if len(parts) < 3:
            no_changes.append({
                "spot": name, "old": lab, "new": lab,
                "reason": "skip_not_3way", "frac": None, "nN": None,
                "anchors": None, "cluster_size": None
            })
            continue

        gi, loc, Xi, i_local, r_use, g = _neighbors_within_islet(
            adata, i_global, group_key, coord_key, k, radius, min_neighbors
        )
        if Xi is None:
            no_changes.append({
                "spot": name, "old": lab, "new": lab,
                "reason": "skip_no_coords", "frac": None, "nN": None,
                "anchors": None, "cluster_size": None
            })
            continue

        trio = frozenset(parts)
        S = set(parts)
        pair = None; r_cls = None
        for cp in pairs_to_r:
            if cp.issubset(S):            # e.g., {'α','β'} ⊂ {'α','β','δ'}
                pair = cp
                r_cls = list(S - set(cp))[0]
                break
        if pair is not None:
            trio_to_rpair[trio] = (pair, r_cls)

        buckets[(g, trio)].append((i_global, gi, Xi, i_local, r_use))

    processed = set()

    for (g, trio), recs in buckets.items():
        gi = recs[0][1]; Xi = recs[0][2]; r_use = recs[0][4]
        members_global = [r[0] for r in recs]
        members_local  = [r[3] for r in recs]

        # --- connected components among this trio's 3-mix members ---
        cluster_r = (cluster_radius_factor * r_use) if (r_use and r_use > 0) else None
        if cluster_r is None or len(recs) == 1:
            components = [[j] for j in range(len(recs))]
        else:
            sub_X = Xi[members_local]
            nn_sub = NearestNeighbors(radius=cluster_r).fit(sub_X)
            nbrs = nn_sub.radius_neighbors(sub_X, return_distance=False)
            components, seen = [], set()
            for j in range(len(recs)):
                if j in seen: 
                    continue
                q = deque([j]); comp=[]
                seen.add(j)
                while q:
                    u = q.popleft(); comp.append(u)
                    for v in nbrs[u]:
                        if v not in seen:
                            seen.add(v); q.append(v)
                components.append(comp)

        for comp in components:
            comp_globals = [members_global[j] for j in comp]
            comp_locals  = [members_local[j]  for j in comp]
            centers = Xi[comp_locals]

            if any(m in processed for m in comp_globals):
                continue

            # union neighborhood around component (exclude members)
            union_local = set()
            for gidx in comp_globals:
                gi2, loc2, Xi2, i_loc2, _r2, _ = _neighbors_within_islet(
                    adata, gidx, group_key, coord_key, k, radius, min_neighbors
                )
                if Xi2 is None:
                    continue
                union_local.update(loc2.tolist())
            union_local = np.array(sorted(set(union_local) - set(comp_locals)), dtype=int)

            # If too few neighbors, record and skip
            if union_local.size < min_neighbors_lo:
                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    no_changes.append({
                        "spot": adata.obs_names[gidx], "old": old, "new": old,
                        "reason": "skip_component_neighbors_too_few",
                        "frac": None, "nN": int(union_local.size) if hasattr(union_local, "size") else None,
                        "anchors": None, "cluster_size": len(comp_globals),
                    })
                processed.update(comp_globals)
                continue

            neigh_globals = gi[union_local]
            neigh_labs = labels.iloc[neigh_globals].tolist()

            # anchor requirement: need at least some pure/2-way neighbors
            n_anchor = 0
            for nl in neigh_labs:
                if nl.count(sep) == 0 or nl.count(sep) == 1:
                    n_anchor += 1
            if n_anchor < min_anchor_pure:
                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    no_changes.append({
                        "spot": adata.obs_names[gidx], "old": old, "new": old,
                        "reason": "skip_not_enough_anchor_neighbors",
                        "frac": None, "nN": int(union_local.size),
                        "anchors": int(n_anchor), "cluster_size": len(comp_globals),
                    })
                processed.update(comp_globals)
                continue

            # distance weights (min distance to any member)
            if use_distance_weights and (r_use and r_use > 0):
                sigma = r_use / 2.5
                dmin = np.min(np.linalg.norm(Xi[union_local,None,:] - centers[None,:,:], axis=2), axis=1)
                dweights = np.exp(-(dmin**2) / (2.0 * sigma**2))
            else:
                dweights = np.ones(len(neigh_labs), dtype=float)

            trio_list = list(trio)

            if mode == "smooth":
                # Vote among the trio
                cnt = _vote_counts(
                    neigh_labs, trio_list,
                    neigh_ids=adata.obs_names[neigh_globals].tolist(),
                    frac=frac, proportional_mixed=proportional_mixed,
                    sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                    exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                    dweights=dweights
                )
                total = sum(cnt.values())
                if total <= 0:
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        no_changes.append({
                            "spot": adata.obs_names[gidx], "old": old, "new": old,
                            "reason": "keep_no_votes_smooth",
                            "frac": None, "nN": int(union_local.size),
                            "anchors": int(n_anchor), "cluster_size": len(comp_globals),
                        })
                    processed.update(comp_globals)
                    continue

                winner = max(trio_list, key=lambda c: cnt[c])
                p_win = cnt[winner] / total

                # Gate by Wilson or fixed majority (if use_wilson=False, pass_cut ignored for resolution)
                if use_wilson:
                    n_eff = kish_effective_n(dweights) if dweights is not None else float(union_local.size)
                    lb = wilson_lower(p_win, max(n_eff, 1.0), z=z_wilson)
                    pass_cut = (lb - 0.50) > 1e-6
                else:
                    thr = majority_thresh if union_local.size >= min_neighbors else majority_thresh_lo
                    pass_cut = (p_win - thr) > 1e-6

                if pass_cut or (not use_wilson):
                    # Resolve the component to winner (record keeps where already consistent)
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        if old != winner:
                            labels.iat[gidx] = winner
                            changes.append({
                                "spot": adata.obs_names[gidx], "old": old, "new": winner,
                                "reason": "3way_joint_majority" + ("_wilson" if use_wilson else ""),
                                "frac": float(p_win), "nN": int(union_local.size),
                                "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                            })
                        else:
                            no_changes.append({
                                "spot": adata.obs_names[gidx], "old": old, "new": old,
                                "reason": "keep_already_consistent_smooth",
                                "frac": float(p_win), "nN": int(union_local.size),
                                "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                            })
                    processed.update(comp_globals)
                else:
                    # Did not pass the gate → keep as is with reason
                    reason_keep = "keep_smooth_wilson_not_significant" if use_wilson else "keep_smooth_frac_below_threshold"
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        no_changes.append({
                            "spot": adata.obs_names[gidx], "old": old, "new": old,
                            "reason": reason_keep,
                            "frac": float(p_win), "nN": int(union_local.size),
                            "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                        })
                    processed.update(comp_globals)

            elif mode == "sharpen":
                # Decide target label for the whole component
                # 1) Case A: αβr → r rule
                if {"α","β"}.issubset(trio):
                    r_cls = [c for c in trio if c not in {"α","β"}][0]
                    cnt_trio = _vote_counts(
                        neigh_labs, trio,
                        neigh_ids=adata.obs_names[neigh_globals].tolist(),
                        frac=frac, proportional_mixed=proportional_mixed,
                        sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                        exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                        dweights=dweights
                    )
                    total_trio = sum(cnt_trio.values())
                    p_win = (cnt_trio[r_cls] / total_trio) if total_trio > 0 else 0.0
                    target = r_cls
                    reason = "3way_sharpen_rule_ab_to_r"
                else:
                    # 2) Case B: vote between {γ, δ} only
                    gd = [c for c in trio if c in ("γ","δ")]
                    if len(gd) == 2:
                        cnt_gd = _vote_counts(
                            neigh_labs, gd,
                            neigh_ids=adata.obs_names[neigh_globals].tolist(),
                            frac=frac, proportional_mixed=proportional_mixed,
                            sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                            exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                            dweights=dweights
                        )
                        total_gd = max(cnt_gd[gd[0]] + cnt_gd[gd[1]], 1e-9)
                        if cnt_gd[gd[0]] > cnt_gd[gd[1]]:
                            target, p_win = gd[0], cnt_gd[gd[0]] / total_gd
                        elif cnt_gd[gd[1]] > cnt_gd[gd[0]]:
                            target, p_win = gd[1], cnt_gd[gd[1]] / total_gd
                        else:
                            # tie: deterministic priority
                            target = None
                            for pref in sharpen_gamma_delta_priority:
                                if pref in gd:
                                    target = pref; break
                            if target is None:
                                target = gd[0]
                            p_win = 0.5
                        reason = "3way_sharpen_vote_gd"
                    else:
                        # 3) Fallback: vote among the trio
                        cnt_trio = _vote_counts(
                            neigh_labs, trio,
                            neigh_ids=adata.obs_names[neigh_globals].tolist(),
                            frac=frac, proportional_mixed=proportional_mixed,
                            sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                            exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                            dweights=dweights
                        )
                        if sum(cnt_trio.values()) > 0:
                            target = max(trio, key=lambda c: cnt_trio[c])
                            p_win = cnt_trio[target] / sum(cnt_trio.values())
                        else:
                            # no signal; keep deterministic but record p_win=0.0
                            target, p_win = list(trio)[0], 0.0
                        reason = "3way_sharpen_vote_trio"

                # Assign the whole component to target; record unchanged too
                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    if old != target:
                        labels.iat[gidx] = target
                        changes.append({
                            'spot': adata.obs_names[gidx], 'old': old, 'new': target,
                            'reason': reason,
                            'frac': float(p_win), 'nN': int(union_local.size),
                            'anchors': int(n_anchor), 'cluster_size': len(comp_globals),
                        })
                    else:
                        no_changes.append({
                            'spot': adata.obs_names[gidx], 'old': old, 'new': old,
                            'reason': 'keep_already_consistent_sharpen',
                            'frac': float(p_win), 'nN': int(union_local.size),
                            'anchors': int(n_anchor), 'cluster_size': len(comp_globals),
                        })
                processed.update(comp_globals)

    changes_df = pd.DataFrame(changes, 
                              columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])
    no_changes_df = pd.DataFrame(no_changes, 
                                 columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])

    return labels, changes_df, no_changes_df

from collections import defaultdict, deque

# ---------------- 4-way joint passes ----------------
def _pass_4way_joint(
    adata, labels,
    *,  # keyword-only
    mode,                           # "smooth" or "sharpen"
    # voting / neighbors
    frac=None, proportional_mixed=True,
    sep="_", group_key="islet", coord_key="spatial",
    k=10, radius=None, min_neighbors=8, min_neighbors_lo=3,
    majority_thresh=0.50, majority_thresh_lo=2/3,
    use_wilson=False, z_wilson=1.64,
    cluster_radius_factor=0.5,
    laplace=0.5, weight_mixed=0.5,
    use_distance_weights=True,
    exclude_threeway=False, threeway_weight=0.2,   # <-- aligned names
    min_anchor_pure=3,
    sharpen_gamma_delta_priority=("δ","γ"),       # tie-break for sharpen
    candidates_idx=None
):
    """
    Jointly relabel connected components of 4-way mixed spots (labels like 'α_β_γ_δ').

    mode="smooth":
        - For each connected component of identical 4-way labels within an islet,
          collect neighbors (excluding the component), tally votes over the 4 singletons,
          and assign the winning singleton to all members (majority or Wilson lower-bound).

    mode="sharpen":
        - For each 4-way component, collapse specifically onto {γ, δ}:
          tally votes only over γ vs δ, pick the winner; if tie, use `sharpen_gamma_delta_priority`.
          (Rationale: α/β diffusion; prefer resolving the rarer endocrine identities.)

    Parameters
    ----------
    adata : AnnData
        Spatial data with obs[group_key] and obsm[coord_key].
    labels : pd.Series
        Current labels (mutable copy).
    mode : {"smooth","sharpen"}
        Strategy for 4-way handling.
    frac : pd.DataFrame or None
        Per-spot absolute fractions; rows=spot_id, cols=classes (e.g. α,β,γ,δ).
    proportional_mixed : bool, default True
        Use `frac` to split mixed neighbor votes proportionally (else equal split).
    sep : str, default "_"
        Separator used in labels.
    group_key : str, default "islet"
        Islet column in obs.
    coord_key : str, default "spatial"
        2D coordinates in obsm.
    k : int, default 10
        Neighbor cap per spot.
    radius : float or None
        Neighborhood radius; if None, auto per-islet (≈ 2.5× median 2NN).
    min_neighbors : int, default 8
        Typical neighbor count; affects fixed-threshold branch in smooth mode.
    min_neighbors_lo : int, default 3
        Minimal neighbors to proceed.
    majority_thresh : float, default 0.5
        Fixed threshold for smooth mode (normal N) when not using Wilson.
    majority_thresh_lo : float, default 2/3
        Fixed threshold for smooth mode (low N) when not using Wilson.
    use_wilson : bool, default False
        If True (smooth mode), require Wilson lower bound over base threshold.
    z_wilson : float, default 1.64
        Normal z for two-sided CI: ~1.28 (80%), 1.64 (90%), 1.96 (95%), 2.58 (99%).
    cluster_radius_factor : float, default 0.5
        Component adjacency radius as factor of r_use.
    laplace, weight_mixed, use_distance_weights, exclude_threeway, threeway_weight
        Voting and weighting knobs passed to _vote_counts.
    weight_mixed : float, default 0.5
        Total weight assigned to a mixed neighbor before splitting votes.
    use_distance_weights : bool, default True
        Whether to apply Gaussian distance weighting.
    exclude_threeway : bool, default False
        If True, neighbors with 3 or more parts are ignored entirely.
    threeway_weight : float, default 0.2
        If not excluding 3-way, multiply their contribution by this factor.
    min_anchor_pure : int, default 3
        Require at least this many anchor neighbors (pure or trusted 2-way) to act.
    sharpen_gamma_delta_priority : set (e.g. ("δ","γ"))
        Tie-breaker when γ and δ are equal under sharpen fallback.

    Returns
    -------
    labels : pd.Series
        Updated labels after 4-way pass.
    changes : pd.DataFrame
        Log of changes with metadata (reason, frac, nN, anchors, cluster_size).
    """
    if mode not in {"smooth", "sharpen"}:
        raise ValueError(f"Invalid mode={mode!r}. Choose 'smooth' or 'sharpen'.")

    idx = adata.obs_names.to_numpy()
    changes, no_changes = [], []
    processed = set()

    # --- bucket 4-mix spots by (islet, four signature) ---
    buckets = defaultdict(list)  # (islet, frozenset(parts)) -> list of (i_global, gi, Xi, i_local, r_use)

    iter_indices = _normalize_indices(candidates_idx, len(idx))
    for i_global in iter_indices:
        name = idx[i_global]
        lab = labels.iat[i_global]
        parts = [p for p in lab.split(sep) if p]
        if len(parts) != 4:
            no_changes.append({
                "spot": name, "old": lab, "new": lab,
                "reason": "skip_not_4way", "frac": None, "nN": None,
                "anchors": None, "cluster_size": None
            })
            continue

        gi, loc, Xi, i_local, r_use, g = _neighbors_within_islet(
            adata, i_global, group_key, coord_key, k, radius, min_neighbors
        )
        if Xi is None:
            no_changes.append({
                "spot": name, "old": lab, "new": lab,
                "reason": "skip_no_coords", "frac": None, "nN": None,
                "anchors": None, "cluster_size": None
            })
            continue

        four = frozenset(parts)
        buckets[(g, four)].append((i_global, gi, Xi, i_local, r_use))

    for (g, four), recs in buckets.items():
        gi = recs[0][1]; Xi = recs[0][2]; r_use = recs[0][4]
        members_global = [r[0] for r in recs]
        members_local  = [r[3] for r in recs]

        # --- connected components among this group's 4-way members ---
        cluster_r = (cluster_radius_factor * r_use) if (r_use and r_use > 0) else None
        if cluster_r is None or len(recs) == 1:
            components = [[j] for j in range(len(recs))]
        else:
            sub_X = Xi[members_local]
            nn_sub = NearestNeighbors(radius=cluster_r).fit(sub_X)
            nbrs = nn_sub.radius_neighbors(sub_X, return_distance=False)
            components, seen = [], set()
            for j in range(len(recs)):
                if j in seen: 
                    continue
                q = deque([j]); comp=[]
                seen.add(j)
                while q:
                    u = q.popleft(); comp.append(u)
                    for v in nbrs[u]:
                        if v not in seen:
                            seen.add(v); q.append(v)
                components.append(comp)

        for comp in components:
            comp_globals = [members_global[j] for j in comp]
            if any(m in processed for m in comp_globals):
                continue
            comp_locals  = [members_local[j]  for j in comp]
            centers = Xi[comp_locals]

            # union neighborhood around component (exclude component members)
            union_local = set()
            for gidx in comp_globals:
                gi2, loc2, Xi2, i_loc2, _r2, _ = _neighbors_within_islet(
                    adata, gidx, group_key, coord_key, k, radius, min_neighbors
                )
                if Xi2 is None:
                    continue
                union_local.update(loc2.tolist())
            union_local = np.array(sorted(set(union_local) - set(comp_locals)), dtype=int)

            # too few neighbors → record and skip
            if union_local.size < min_neighbors_lo:
                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    no_changes.append({
                        "spot": adata.obs_names[gidx], "old": old, "new": old,
                        "reason": "skip_component_neighbors_too_few",
                        "frac": None, "nN": int(union_local.size) if hasattr(union_local, "size") else None,
                        "anchors": None, "cluster_size": len(comp_globals),
                    })
                processed.update(comp_globals)
                continue

            neigh_globals = gi[union_local]
            neigh_labs = labels.iloc[neigh_globals].tolist()

            # require some anchor neighbors (pure or trusted 2-way)
            n_anchor = 0
            for nl in neigh_labs:
                if nl.count(sep) == 0 or nl.count(sep) == 1:
                    n_anchor += 1
            if n_anchor < min_anchor_pure:
                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    no_changes.append({
                        "spot": adata.obs_names[gidx], "old": old, "new": old,
                        "reason": "skip_not_enough_anchor_neighbors",
                        "frac": None, "nN": int(union_local.size),
                        "anchors": int(n_anchor), "cluster_size": len(comp_globals),
                    })
                processed.update(comp_globals)
                continue

            # distance weights (min dist to any member in component)
            if use_distance_weights and (r_use and r_use > 0):
                sigma = r_use / 2.5
                dmin = np.min(np.linalg.norm(Xi[union_local,None,:] - centers[None,:,:], axis=2), axis=1)
                dweights = np.exp(-(dmin**2) / (2.0 * sigma**2))
            else:
                dweights = np.ones(len(neigh_labs), dtype=float)

            four_list = sorted(list(four))  # stable order

            if mode == "smooth":
                # vote among all 4 singleton classes
                cnt = _vote_counts(
                    neigh_labs, four_list,
                    neigh_ids=adata.obs_names[neigh_globals].tolist(),
                    frac=frac, proportional_mixed=proportional_mixed,
                    sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                    exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                    dweights=dweights
                )
                total = sum(cnt.values())
                if total <= 0:
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        no_changes.append({
                            "spot": adata.obs_names[gidx], "old": old, "new": old,
                            "reason": "keep_no_votes_smooth",
                            "frac": None, "nN": int(union_local.size),
                            "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                        })
                    processed.update(comp_globals)
                    continue

                winner = max(four_list, key=lambda c: cnt[c])
                p_win = cnt[winner] / total

                if use_wilson:
                    n_eff = kish_effective_n(dweights) if dweights is not None else float(union_local.size)
                    lb = wilson_lower(p_win, max(n_eff, 1.0), z=z_wilson)
                    pass_cut = (lb - 0.50) > 1e-6
                else:
                    thr = majority_thresh if union_local.size >= min_neighbors else majority_thresh_lo
                    pass_cut = (p_win - thr) > 1e-6

                if pass_cut or (not use_wilson):  # set use_wilson=False to always resolve
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        if old != winner:
                            labels.iat[gidx] = winner
                            changes.append({
                                "spot": adata.obs_names[gidx], "old": old, "new": winner,
                                "reason": "4way_joint_smooth" + ("_wilson" if use_wilson else ""),
                                "frac": float(p_win), "nN": int(union_local.size),
                                "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                            })
                        else:
                            no_changes.append({
                                "spot": adata.obs_names[gidx], "old": old, "new": old,
                                "reason": "keep_already_consistent_smooth",
                                "frac": float(p_win), "nN": int(union_local.size),
                                "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                            })
                    processed.update(comp_globals)
                else:
                    reason_keep = "keep_smooth_wilson_not_significant" if use_wilson else "keep_smooth_frac_below_threshold"
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        no_changes.append({
                            "spot": adata.obs_names[gidx], "old": old, "new": old,
                            "reason": reason_keep,
                            "frac": float(p_win), "nN": int(union_local.size),
                            "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                        })
                    processed.update(comp_globals)

            else:  # mode == "sharpen"
                gd = [c for c in four_list if c in ("γ","δ")]
                if len(gd) != 2:
                    for gidx in comp_globals:
                        old = labels.iat[gidx]
                        no_changes.append({
                            "spot": adata.obs_names[gidx], "old": old, "new": old,
                            "reason": "skip_sharpen_no_gd_pair",
                            "frac": None, "nN": int(union_local.size),
                            "anchors": int(n_anchor), "cluster_size": len(comp_globals)
                        })
                    processed.update(comp_globals)
                    continue

                cnt = _vote_counts(
                    neigh_labs, gd,
                    neigh_ids=adata.obs_names[neigh_globals].tolist(),
                    frac=frac, proportional_mixed=proportional_mixed,
                    sep=sep, laplace=laplace, weight_mixed=weight_mixed,
                    exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
                    dweights=dweights
                )
                total_gd = max(cnt[gd[0]] + cnt[gd[1]], 1e-9)
                if cnt[gd[0]] > cnt[gd[1]]:
                    target, p_win = gd[0], cnt[gd[0]] / total_gd
                elif cnt[gd[1]] > cnt[gd[0]]:
                    target, p_win = gd[1], cnt[gd[1]] / total_gd
                else:
                    target = None
                    for pref in sharpen_gamma_delta_priority:
                        if pref in gd: 
                            target = pref; break
                    if target is None: 
                        target = gd[0]
                    p_win = 0.5

                for gidx in comp_globals:
                    old = labels.iat[gidx]
                    if old != target:
                        labels.iat[gidx] = target
                        changes.append({
                            'spot': adata.obs_names[gidx], 'old': old, 'new': target,
                            'reason': "4way_sharpen_vote_gd",
                            'frac': float(p_win), 'nN': int(union_local.size),
                            'anchors': int(n_anchor), 'cluster_size': len(comp_globals),
                        })
                    else:
                        no_changes.append({
                            'spot': adata.obs_names[gidx], 'old': old, 'new': old,
                            'reason': 'keep_already_consistent_sharpen',
                            'frac': float(p_win), 'nN': int(union_local.size),
                            'anchors': int(n_anchor), 'cluster_size': len(comp_globals),
                        })
                processed.update(comp_globals)

    changes_df = pd.DataFrame(changes, 
                              columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])
    no_changes_df = pd.DataFrame(no_changes,
                                 columns=["spot","old","new","reason","frac","nN","anchors","cluster_size"])

    return labels, changes_df, no_changes_df

# ---------------- Orchestrator ----------------
def staged_relabel_mode(
    adata,
    *,  # keyword-only from here
    mode,                           # "smooth" or "sharpen"
    pairs_to_r=(("α","β"),),        # defines 'r' in α_β_r trios (order-insensitive)
    frac=None,                      # per-spot fractions DataFrame (rows=spot_id, cols=classes), or None
    proportional_mixed=True,        # split mixed-neighbor votes by frac if provided
    sep='_',
    coord_key="spatial",
    group_key="islet",
    label_col="label",

    # neighborhood & thresholds (used by both passes)
    k=10,
    radius=None,
    min_neighbors=8,
    min_neighbors_lo=3,
    majority_thresh=0.5,            # base majority threshold (normal N)
    majority_thresh_lo=2/3,         # stricter threshold (low N)

    # optional Wilson gating (primarily for smooth mode)
    use_wilson=False,
    z_wilson=1.64,                  # ~90% CI; 1.96 for ~95%

    # robustness knobs for vote aggregation
    laplace=0.5,
    weight_mixed=0.5,
    use_distance_weights=True,

    # 3-way joint extras
    cluster_radius_factor=0.5,      # adjacency radius = factor * r_use
    exclude_threeway=True,          # when aggregating neighbor votes, drop 3-way neighbors
    threeway_weight=0.2,            # (if not excluded) downweight 3-way neighbors
    min_anchor_pure=3,              # require ≥ this many pure/2-way anchors before acting

    # sharpen fallback when γ & δ tie: choose first in this tuple
    sharpen_gamma_delta_priority=("δ","γ"),
):
    """
    Run a two-stage relabeling pipeline on spatial labels.

    Stages
    ------
    1) 2-way pass:
       - mode="smooth": encourage homogeneity → reassign 2-way mixes to local majority
       - mode="sharpen": preserve complexity → reassign 2-way mixes to local minority
       Uses per-islet neighbors (radius/k), distance weights (optional), and mixed-label handling.
       Optional Wilson lower-bound gating (use_wilson=True) replaces fixed thresholds.

    2) 3-way joint pass:
       - mode="smooth": group adjacent 3-mix spots (same trio) and assign the component to the
         trio majority (optionally Wilson-gated). This always resolves 3-mixes.
       - mode="sharpen": if trio matches α_β_r (as defined by pairs_to_r), assign to r;
         else vote between {γ, δ} and pick the winner (tie broken by `sharpen_gamma_delta_priority`).
        
    3) 4-way joint pass:
        - mode="smooth": tally votes over the 4 singletons, and assign the winning singleton to all 
          members (majority or Wilson lower-bound).
        - mode="sharpen": tally votes only over γ vs δ, pick the winner; 
          if tie, use `sharpen_gamma_delta_priority`.(Rationale: α/β diffusion; 
          prefer resolving the rarer endocrine identities.)

    Parameters
    ----------
    adata : AnnData
        Spatial AnnData with obs[group_key], obsm[coord_key], and obs[label_col].
    mode : {"smooth","sharpen"}
        Strategy controlling both stages (see above).
    pairs_to_r : iterable[tuple[str,str]]
        Pairs that define “r” in α_β_r trios (order-insensitive).
    frac : pd.DataFrame or None
        Per-spot absolute fractions for classes (e.g., α,β,γ,δ). If provided and
        proportional_mixed=True, mixed neighbors split votes proportionally.
    proportional_mixed : bool
        If True and `frac` provided, use proportional splitting; else equal split.
    sep, coord_key, group_key, label_col : str
        Column/keys in AnnData for labels, coordinates, islets.
    k, radius : int or float
        Neighbor cap and/or search radius (auto radius if None).
    min_neighbors, min_neighbors_lo : int
        Neighbor count thresholds for normal vs “low-N” decisions.
    majority_thresh, majority_thresh_lo : float
        Fixed pass thresholds (used if `use_wilson=False`).
    use_wilson : bool
        If True, apply Wilson lower bound instead of fixed thresholds (mostly smooth mode).
    z_wilson : float
        Z-score for Wilson bound (1.64≈90%, 1.96≈95%).
    laplace, weight_mixed : float
        Voting smoothing and mixed-neighbor weight budget.
    use_distance_weights : bool
        Apply Gaussian distance weights per islet (closer neighbors count more).
    cluster_radius_factor : float
        For 3-way: adjacency radius = factor * r_use to cluster connected 3-mix spots.
    exclude_threeway : bool
        If True, drop 3-way neighbors from voting (as they are noisy).
    threeway_weight : float
        If not excluded, downweight 3-way neighbors by this factor.
    min_anchor_pure : int
        Require at least this many anchor neighbors (pure/2-way) to act in 3-way.
    sharpen_gamma_delta_priority : tuple[str,str]
        Deterministic tie-break order when γ vs δ votes tie in sharpen mode.

    Returns
    -------
    labels : pd.Series
        Updated labels.
    summary : pd.DataFrame
        Concatenated logs from both stages: columns like ["spot","old","new","reason",...].

    Notes
    -----
    - The function does not write back to `adata.obs[label_col]`. Assign manually if desired.
    - Ensure `_pass_2way`, `_pass_3way_joint` and `_pass_4way_joint` accept the parameters you pass
      (especially `z_wilson` if `use_wilson=True`).
    """
    # Validate mode early
    if mode not in {"smooth", "sharpen"}:
        raise ValueError(f"Invalid mode={mode!r}. Choose 'smooth' or 'sharpen'.")

    # Work on a mutable copy of labels
    labels = adata.obs[label_col].astype(str).copy()

    # ---- compute candidates once ----
    # count of separators = (#parts - 1)
    sep_counts = labels.str.count(repr(sep)[1:-1] if sep in r"\.^$*+?{}[]|()" else sep)
    # If sep is a literal (most cases), the above reduces to labels.str.count(sep)

    idx_all = np.arange(len(labels))
    cand2 = idx_all[sep_counts == 1]  # 2-mix
    cand3 = idx_all[sep_counts == 2]  # 3-mix
    cand4 = idx_all[sep_counts == 3]  # 4-mix

    changes_logs = []
    no_changes_logs = []

    # ---- Stage 1: 2-way (only 2-mix) ----
    labels, ch1, nch1 = _pass_2way(
        adata, labels, mode=mode,
        frac=frac, proportional_mixed=proportional_mixed,
        sep=sep, group_key=group_key, coord_key=coord_key,
        k=k, radius=radius,
        min_neighbors=min_neighbors, min_neighbors_lo=min_neighbors_lo,
        majority_thresh=majority_thresh, majority_thresh_lo=majority_thresh_lo,
        use_wilson=use_wilson, z_wilson=z_wilson,
        laplace=laplace, weight_mixed=weight_mixed, use_distance_weights=use_distance_weights,
        candidates_idx=cand2,
    )
    if not ch1.empty: ch1 = ch1.assign(stage="2way"); changes_logs.append(ch1)
    if not nch1.empty: nch1 = nch1.assign(stage="2way"); no_changes_logs.append(nch1)

    # ---- Stage 2: 3-way joint (only 3-mix) ----
    labels, ch2, nch2 = _pass_3way_joint(
        adata, labels, mode=mode,
        pairs_to_r=pairs_to_r,
        frac=frac, proportional_mixed=proportional_mixed,
        sep=sep, group_key=group_key, coord_key=coord_key,
        k=k, radius=radius,
        min_neighbors=min_neighbors, min_neighbors_lo=min_neighbors_lo,
        majority_thresh=majority_thresh, majority_thresh_lo=majority_thresh_lo,
        use_wilson=use_wilson, z_wilson=z_wilson,
        cluster_radius_factor=cluster_radius_factor,
        laplace=laplace, weight_mixed=weight_mixed, use_distance_weights=use_distance_weights,
        exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
        min_anchor_pure=min_anchor_pure,
        sharpen_gamma_delta_priority=sharpen_gamma_delta_priority,
        candidates_idx=cand3,
    )
    if not ch2.empty: ch2 = ch2.assign(stage="3way"); changes_logs.append(ch2)
    if not nch2.empty: nch2 = nch2.assign(stage="3way"); no_changes_logs.append(nch2)

    # ---- Stage 3: 4-way joint (only 4-mix) ----
    labels, ch3, nch3 = _pass_4way_joint(
        adata, labels, mode=mode,
        frac=frac, proportional_mixed=proportional_mixed,
        sep=sep, group_key=group_key, coord_key=coord_key,
        k=k, radius=radius,
        min_neighbors=min_neighbors, min_neighbors_lo=min_neighbors_lo,
        majority_thresh=majority_thresh, majority_thresh_lo=majority_thresh_lo,
        use_wilson=use_wilson, z_wilson=z_wilson,
        cluster_radius_factor=cluster_radius_factor,
        laplace=laplace, weight_mixed=weight_mixed, use_distance_weights=use_distance_weights,
        exclude_threeway=exclude_threeway, threeway_weight=threeway_weight,
        min_anchor_pure=min_anchor_pure,
        sharpen_gamma_delta_priority=sharpen_gamma_delta_priority,
        candidates_idx=cand4,
    )
    if not ch3.empty: ch3 = ch3.assign(stage="4way"); changes_logs.append(ch3)
    if not nch3.empty: nch3 = nch3.assign(stage="4way"); no_changes_logs.append(nch3)

    std_cols = ["spot","old","new","reason","frac","nN","anchors","cluster_size","stage"]
    changes_summary = (pd.concat(changes_logs, ignore_index=True)[std_cols]
                       if changes_logs else pd.DataFrame(columns=std_cols))
    no_changes_summary = (pd.concat(no_changes_logs, ignore_index=True)[std_cols]
                          if no_changes_logs else pd.DataFrame(columns=std_cols))

    return labels, changes_summary, no_changes_summary

'''
spot → the spot/barcode ID

old / new → labels before and after the pass (same here since no relabel happened)

reason → why the relabel didn’t happen

frac → the majority fraction (strength of majority vote among neighbors)

nN → number of neighbors considered

anchors → number of “trusted” anchor neighbors (pure or 2-way); NaN if not evaluated at that stage

cluster_size → number of same-mix neighbors in the local cluster

stage → which pass (2way / 3way / 4way) was running
'''

# -------- Run in smooth mode
# Encourages local homogeneity (majority for 2-way, majority for 3-way)
new_labels_smooth, changes_summary_smooth, no_changes_summary_smooth = staged_relabel_mode(
    spat_low,
    mode="smooth",
    pairs_to_r=(("α","β"),),   # defines α_β_r → r (used only in sharpen mode)
    frac=frac,              # optional, to split mixed votes proportionally
    proportional_mixed=True,
    label_col="label",
    group_key="islet",
    coord_key="spatial",
    k=10,
    radius=None,               # auto radius ≈ 2.5× spacing
    min_neighbors=8,
    min_neighbors_lo=3,
    majority_thresh=0.50,
    majority_thresh_lo=2/3,
    use_wilson=True,           # optional: turn on Wilson gating
    z_wilson=1.64,             # 90% confidence
)

# # ------- Run in sharpen mode
# # Preserves local complexity (minority for 2-way, δ/γ or r for 3-way):
new_labels_sharpen, changes_summary_sharpen, no_changes_summary_sharpen = staged_relabel_mode(
    spat_low,
    mode="sharpen",
    pairs_to_r=(("α","β"),),   # αβr trios → assign to r
    frac=frac,
    proportional_mixed=True,
    label_col="label",
    group_key="islet",
    coord_key="spatial",
    k=10,
    radius=None,
    min_neighbors=8,
    min_neighbors_lo=3,
    majority_thresh=0.50,
    majority_thresh_lo=0.5,
    use_wilson=True,           # optional: turn on Wilson gating
    z_wilson=1.64,             # 90% confidence
)

# Save back to AnnData if you like
spat_low.obs["dominant_smooth"]  = new_labels_smooth
spat_low.obs["dominant_sharpen"] = new_labels_sharpen


sample = 'U6-slice'

fig, ax = plt.subplots()
sc.pl.spatial(
    spat_low[spat_low.obs['sample'] == sample].copy(),
    color=["dominant_smooth"],
    spot_size=10,
    cmap="Reds",
    show=False,
    ax=ax,
    crop_coord=(1100,1500,2400,2900)   # your crop if needed
)
ax.invert_xaxis()
plt.show()


sample = 'U6-slice'

fig, ax = plt.subplots()
sc.pl.spatial(
    spat_low[spat_low.obs['sample'] == sample].copy(),
    color=["dominant_sharpen"],
    spot_size=10,
    cmap="Reds",
    show=False,
    ax=ax,
    crop_coord=(1100,1500,2400,2900)   # your crop if needed
)
ax.invert_xaxis()
plt.show()




