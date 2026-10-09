"""
Writes paper/mst/numbers.tex: one LaTeX macro per result quoted in the Measurement Science and Technology manuscript,
each computed from data/results/*.json at build time. The manuscript types no result by hand; check_mst.py verifies
that every decimal left in the text is a declared design constant.
"""
import os, json, math

HERE = os.path.dirname(__file__)
RES = os.path.join(HERE, "..", "data", "results")
OUT = os.path.join(HERE, "..", "paper", "mst", "numbers.tex")
J = lambda n: json.load(open(os.path.join(RES, n)))


def f(x, dp):
    s = f"{x:.{dp}f}"
    return ("\\ensuremath{-}" + s[1:]) if s.startswith("-") else s


def g4(x):
    """Grid values to at most four decimals, trailing zeros dropped (e.g. 1.7375, 1.785)."""
    return f"{round(x, 4):.4f}".rstrip("0").rstrip(".")


def sci(x):
    """Upper bound in the form m x 10^e with one significant digit (rounded up), for statements of the kind 'below'."""
    e = int(math.floor(math.log10(abs(x))))
    m = math.ceil(x / 10 ** e - 1e-9)
    if m == 10:
        m, e = 1, e + 1
    return f"{m:d}\\times10^{{{e}}}"


def mant(x, dp):
    """x in the form m x 10^e with dp decimals in the mantissa (for math mode)."""
    e = int(math.floor(math.log10(abs(x))))
    return f"{x / 10 ** e:.{dp}f}\\times10^{{{e}}}"


def ci(c, dp):
    return f"[{f(c[0], dp)}, {f(c[1], dp)}]"


def pct(x, dp):
    return f(100 * x, dp)


def build():
    V = {}
    S, M, W, F, D, Mo = J("rev_structural.json"), J("rev_matched.json"), J("rev_sweep.json"), J("rev_fisher.json"), J("rev_dispersion.json"), J("rev_modulation.json")

    # ---- structure
    sy = S["symmetry"]
    V["SymMaxDR"] = sci(sy["max_abs_reflectance_difference"]); V["SymCases"] = str(sy["n_random_cases"])
    V["TwinN"] = f(sy["example_ns_2p30"]["n_mirror"], 4); V["TwinD"] = f(sy["example_ns_2p30"]["d_mirror_nm"], 1)
    P = S["index_profiles"]
    for key, tag in (("main_regime_ns1p5", "Main"), ("low_index_ns1p5", "Low"), ("reviewer_example_ns2p30", "Rev"), ("high_index_ns2p30", "High")):
        p = P[key]
        V[f"SetKnown{tag}W"] = f(p["known_gain_set_width"], 3); V[f"SetUnknown{tag}W"] = f(p["unknown_gain_set_width"], 3)
        V[f"SetKnown{tag}"] = f"[{g4(p['known_gain_set'][0])}, {g4(p['known_gain_set'][1])}]"
        V[f"SetUnknown{tag}"] = f"[{g4(p['unknown_gain_set'][0])}, {g4(p['unknown_gain_set'][1])}]"
        V[f"FarKnown{tag}"] = f(p["min_far_rss_over_sigma2_known"], 1); V[f"FarUnknown{tag}"] = f(p["min_far_rss_over_sigma2_unknown"], 1)
    ge = S["gain_error_bias"]["by_gain_error"]
    V["GEFilms"] = str(S["gain_error_bias"]["n_films"])
    V["GEonePct"] = f(0.5 * (ge["+0.01"]["median_abs_bias_n"] + ge["-0.01"]["median_abs_bias_n"]), 4)
    assert f(ge["+0.01"]["median_abs_bias_n"], 4) == f(ge["-0.01"]["median_abs_bias_n"], 4) == V["GEonePct"]
    V["GEtenPos"] = f(ge["+0.10"]["median_abs_bias_n"], 3); V["GEtenNeg"] = f(ge["-0.10"]["median_abs_bias_n"], 3)
    V["GEtwentyPos"] = f(ge["+0.20"]["median_abs_bias_n"], 3)
    V["GEtauPerPct"] = f(max(ge["+0.01"]["median_abs_bias_tau_nm"], ge["-0.01"]["median_abs_bias_tau_nm"]), 3)
    V["GEmatched"] = str(ge["+0.01"]["n_index_matched_films"])
    V["GEtauMaxExclTen"] = f(max(ge[k]["max_abs_bias_tau_nm_excluding_index_matched"] for k in ge if abs(float(k)) <= 0.10), 1)
    V["GEtauMaxExclTwenty"] = f(ge["+0.20"]["max_abs_bias_tau_nm_excluding_index_matched"], 0)
    dep = S["departures"]["cases"]
    V["DepMatched"] = str(dep["scalar_gain_1p10"]["gain_fixed_at_1"]["n_index_matched_films"])
    V["DepScaleFixed"] = f(dep["scalar_gain_1p10"]["gain_fixed_at_1"]["median_abs_bias_n"], 3)
    V["DepScaleProfMax"] = sci(dep["scalar_gain_1p10"]["scalar_gain_profiled"]["max_abs_bias_tau_nm"])
    o = dep["additive_offset_0p02"]
    V["DepOffsetFixed"] = f(o["gain_fixed_at_1"]["median_abs_bias_n"], 3); V["DepOffsetProf"] = f(o["scalar_gain_profiled"]["median_abs_bias_n"], 3)
    V["DepOffsetFixedTau"] = f(o["gain_fixed_at_1"]["median_abs_bias_tau_nm"], 1); V["DepOffsetProfTau"] = f(o["scalar_gain_profiled"]["median_abs_bias_tau_nm"], 3)
    V["DepOffsetFixedTauFrac"] = pct(o["gain_fixed_at_1"]["frac_abs_bias_tau_gt_10nm"], 0)
    t = dep["linear_spectral_gain_+10pct_450nm_-5pct_700nm"]
    V["DepTiltFixed"] = f(t["gain_fixed_at_1"]["median_abs_bias_n"], 4); V["DepTiltProf"] = f(t["scalar_gain_profiled"]["median_abs_bias_n"], 4)
    V["DepTiltFixedTau"] = f(t["gain_fixed_at_1"]["median_abs_bias_tau_nm"], 2); V["DepTiltProfTau"] = f(t["scalar_gain_profiled"]["median_abs_bias_tau_nm"], 2)

    # ---- matched-model Monte Carlo
    e1 = M["E1"]; k, p, m = e1["calibrated_known_gain"], e1["shape_only_profiled_gain"], e1["unknown_gain_marginalized"]
    V["EoneKnownRel"] = f(k["n"]["rel_rmse"], 4); V["EoneProfRel"] = f(p["n"]["rel_rmse"], 4); V["EoneMargRel"] = f(m["n"]["rel_rmse"], 4)
    V["EoneKnownAbs"] = f(k["n"]["rmse"], 4); V["EoneProfAbs"] = f(p["n"]["rmse"], 4); V["EoneMargAbs"] = f(m["n"]["rmse"], 4)
    V["EoneKnownD"] = f(k["d"]["rmse"], 1); V["EoneProfD"] = f(p["d"]["rmse"], 1); V["EoneMargD"] = f(m["d"]["rmse"], 1)
    V["EoneKnownTau"] = f(k["tau"]["rmse"], 1); V["EoneProfTau"] = f(p["tau"]["rmse"], 1); V["EoneMargTau"] = f(m["tau"]["rmse"], 1)
    V["EoneKnownBias"] = f(k["n"]["bias"], 4); V["EoneProfBias"] = f(p["n"]["bias"], 4); V["EoneMargBias"] = f(m["n"]["bias"], 4)
    R = e1["ratios"]
    for est, tag in (("mean", "Mean"), ("median", "Median"), ("map", "Map")):
        V[f"RatioProf{tag}"] = f(R[f"profile_over_known_n_{est}"]["ratio"], 2); V[f"RatioProf{tag}CI"] = ci(R[f"profile_over_known_n_{est}"]["ci95"], 2)
        V[f"RatioMarg{tag}"] = f(R[f"marginal_over_known_n_{est}"]["ratio"], 2); V[f"RatioMarg{tag}CI"] = ci(R[f"marginal_over_known_n_{est}"]["ci95"], 2)
    V["RatioTau"] = f(R["profile_over_known_tau_mean"]["ratio"], 2); V["RatioTauCI"] = ci(R["profile_over_known_tau_mean"]["ci95"], 2)
    V["RatioD"] = f(R["profile_over_known_d_mean"]["ratio"], 2); V["RatioDCI"] = ci(R["profile_over_known_d_mean"]["ci95"], 2)
    gh = e1["profiled_gain_at_map_cell"]; V["GhatMin"] = f(gh["min"], 2); V["GhatMax"] = f(gh["max"], 2); V["GhatNonpos"] = str(gh["n_nonpositive"])
    V["ModMedian"] = f(Mo["modulation_over_sigma_median"], 1); V["ModQ"] = ci(Mo["modulation_over_sigma_q25_q75"], 1)
    V["ModBelowPct"] = pct(Mo["frac_modulation_below_sigma"], 1)
    mb = Mo["by_modulation"]
    V["ModLowCount"] = str(mb[0]["count"]); V["ModLowNlo"] = f(mb[0]["true_index_range"][0], 2); V["ModLowNhi"] = f(mb[0]["true_index_range"][1], 2)
    V["ModLowTau"] = f(mb[0]["known_rmse_tau_nm"], 0); V["ModHighTau"] = f(mb[3]["known_rmse_tau_nm"], 1); V["ModTopTau"] = f(mb[4]["known_rmse_tau_nm"], 1)
    V["ModMidTau"] = f(mb[2]["known_rmse_tau_nm"], 1)
    V["ModHighTauProf"] = f(mb[3]["profile_rmse_tau_nm"], 1); V["ModTopTauProf"] = f(mb[4]["profile_rmse_tau_nm"], 1)
    V["ModHighRatio"] = f(mb[3]["profile_over_known_rmse_n"], 1); V["ModTopRatio"] = f(mb[4]["profile_over_known_rmse_n"], 1)
    V["TauMedianErr"] = f(Mo["known_median_abs_tau_err_nm_all"], 1)
    e2 = M["E2"]
    for key, tag in (("known", "Cor"), ("half", "Half"), ("double", "Dbl")):
        for q in ("n", "d", "tau"):
            Q = q.capitalize() if q != "tau" else "Tau"
            V[f"Cov{tag}{Q}Six"] = f(e2[key][q]["cov68"], 3); V[f"Cov{tag}{Q}Nine"] = f(e2[key][q]["cov90"], 3)
        V[f"Wid{tag}N"] = f(e2[key]["n"]["mean_width90"], 3)
    V["CovMargN"] = f(m["n"]["cov90"], 3); V["WidMargN"] = f(m["n"]["mean_width90"], 3)
    terc = e2["coverage_by_true_parameter_tercile_correct_noise"]
    vals = [terc[a][f"tercile_{j}"][f"{b}_cov90"] for a in ("n", "d") for j in (1, 2, 3) for b in ("n", "d", "tau")]
    V["TercMin"] = f(min(vals), 3); V["TercMax"] = f(max(vals), 3)
    gc = M["grid_convergence"]
    V["GCSix"] = f(gc["E2_correct"]["n"]["cov68"], 3); V["GCNine"] = f(gc["E2_correct"]["n"]["cov90"], 3)
    V["GCKnownChange"] = f(100 * abs(gc["E1_calibrated"]["n"]["rel_rmse"] / k["n"]["rel_rmse"] - 1), 2)
    V["GCProfChange"] = f(100 * abs(gc["E1_shape_only"]["n"]["rel_rmse"] / p["n"]["rel_rmse"] - 1), 2)
    V["CovSixSE"] = f(e2["known"]["n"]["cov68_se"], 3)
    for tag, E in (("", M["E3_fixed_noise"]), ("b", M["E3b_noise_scales_with_gain"])):
        T = "Three" + tag
        for kk, kt in (("true_gain", "True"), ("gain_one", "One"), ("marginal", "Marg")):
            V[f"{T}{kt}CovN"] = f(E[kk]["n"]["cov90"], 3); V[f"{T}{kt}RmseN"] = f(E[kk]["n"]["rmse"], 4); V[f"{T}{kt}WidN"] = f(E[kk]["n"]["mean_width90"], 3)
            V[f"{T}{kt}CovD"] = f(E[kk]["d"]["cov90"], 3); V[f"{T}{kt}RmseD"] = f(E[kk]["d"]["rmse"], 1); V[f"{T}{kt}WidD"] = f(E[kk]["d"]["mean_width90"], 1)
            V[f"{T}{kt}CovTau"] = f(E[kk]["tau"]["cov90"], 3)
        V[f"{T}ProfRmseN"] = f(E["profile"]["n"]["rmse"], 4)
        r = E["ratios"]
        V[f"{T}RatioMarg"] = f(r["marginal_over_true_gain_n"]["ratio"], 2); V[f"{T}RatioMargCI"] = ci(r["marginal_over_true_gain_n"]["ci95"], 2)
        V[f"{T}RatioOne"] = f(r["gain_one_over_true_gain_n"]["ratio"], 2); V[f"{T}RatioOneCI"] = ci(r["gain_one_over_true_gain_n"]["ci95"], 2)
        V[f"{T}RatioProf"] = f(r["profile_over_true_gain_n"]["ratio"], 2); V[f"{T}RatioProfCI"] = ci(r["profile_over_true_gain_n"]["ci95"], 2)
        V[f"{T}RatioMargD"] = f(r["marginal_over_true_gain_d"]["ratio"], 2); V[f"{T}RatioMargDCI"] = ci(r["marginal_over_true_gain_d"]["ci95"], 2)
        bins = E["by_gain_error"]
        for bk, bt in (("|g-1|<0.05", "Lo"), ("0.05<=|g-1|<0.10", "Mid"), ("|g-1|>=0.10", "Hi")):
            V[f"{T}Bin{bt}Count"] = str(bins[bk]["count"])
            for kk, kt in (("true_gain", "True"), ("gain_one", "One"), ("marginal", "Marg")):
                V[f"{T}Bin{bt}{kt}Cov"] = f(bins[bk][kk]["n_cov90"], 3); V[f"{T}Bin{bt}{kt}Rmse"] = f(bins[bk][kk]["n_rmse"], 3)
        sg = E["gain_one_bias_by_sign"]
        V[f"{T}BiasBelow"] = f(sg["g<1"]["n_bias"], 3); V[f"{T}BiasAbove"] = f(sg["g>1"]["n_bias"], 3)
        V[f"{T}OneBias"] = f(E["gain_one"]["n"]["bias"], 4)

    # ---- sweep
    sb = W["axes"]["substrate_index"]
    names = {1.35: "A", 1.5: "B", 1.75: "C", 2.0: "D", 2.3: "E"}
    for r in sb:
        t = names[r["n_s"]]
        V[f"Sub{t}Full"] = f(r["rho_sep_full"]["ratio"], 2); V[f"Sub{t}FullCI"] = ci(r["rho_sep_full"]["ci95"], 2)
        V[f"Sub{t}Br"] = f(r["rho_sep_branch"]["ratio"], 2); V[f"Sub{t}BrCI"] = ci(r["rho_sep_branch"]["ci95"], 2)
        V[f"Sub{t}RelFull"] = f(r["rel_n_known_full"], 4); V[f"Sub{t}RelBr"] = f(r["rel_n_known_branch"], 4)
        V[f"Sub{t}Mirror"] = pct(r["frac_mirror_in_grid"], 0); V[f"Sub{t}Mass"] = f(r["mean_mirror_mass_known"], 3)
    fr = W["axes"]["fringe_count"]
    V["FrMin"] = f(min(r["rho_sep_full"]["ratio"] for r in fr), 2); V["FrMax"] = f(max(r["rho_sep_full"]["ratio"] for r in fr), 2)
    V["FrFringesLo"] = f(fr[0]["fringes_mid"], 1); V["FrFringesHi"] = f(fr[-1]["fringes_mid"], 1)
    nz = W["axes"]["noise"]
    for r, t in zip(nz, ("A", "B", "C", "D", "E")):
        V[f"Nz{t}"] = f(r["rho_sep_full"]["ratio"], 2); V[f"Nz{t}CI"] = ci(r["rho_sep_full"]["ci95"], 2)
    gcs = W["grid_convergence"]
    V["SwGCAFull"] = f(gcs[0]["rho_sep_full"]["ratio"], 2); V["SwGCEFull"] = f(gcs[1]["rho_sep_full"]["ratio"], 2); V["SwGCEBr"] = f(gcs[1]["rho_sep_branch"]["ratio"], 2)
    V["SwFilms"] = str(W["config"]["n_sims"])
    V["SwProfHighNoise"] = f(nz[-1]["rel_n_profile_full"], 3); assert nz[-1]["sigma"] == 0.08
    lo_, hi_ = W["config"]["n_range"]
    V["NoInfo"] = f((hi_ - lo_) / math.sqrt(12) / (0.5 * (lo_ + hi_)), 3)

    # ---- local analysis
    dc = F["derivative_checks"]
    row6 = [r for r in dc["central_difference_table"] if r["step_n"] == 1e-6][0]
    assert math.isclose(row6["step_d_nm"], 1e-4)
    V["DerivErr"] = sci(row6["max_rel_err_dRdn"]); V["DerivErrD"] = sci(row6["max_rel_err_dRdd"])
    V["OldStepChange"] = sci(dc["fisher_theory_steps_max_rel_change_in_sd_ratio"])
    V["ModelErr"] = sci(dc["max_abs_R_difference_vs_optics_reflectance"])
    V["IdentityErr"] = sci(max(F["identity_checks"].values()))
    V["FixedPointErr"] = sci(max(v["max_abs_fresnel_vector_at_n_star"] for v in F["fixed_point"].values()))
    rf = F["reference_films"]
    for key, tag in (("n1p80_d650_ns1p5", "A"), ("n1p45_d650_ns1p5", "B"), ("n2p00_d650_ns2p30", "C")):
        V[f"Ref{tag}Ratio"] = f(rf[key]["sd_ratio"], 2); V[f"Ref{tag}Rho"] = f(rf[key]["rho_ng_given_tau"], 3)
        V[f"Ref{tag}CorrND"] = f(rf[key]["corr_n_d"], 3); V[f"Ref{tag}CrbK"] = f(rf[key]["crb_sd_n_known"], 4); V[f"Ref{tag}CrbU"] = f(rf[key]["crb_sd_n_unknown"], 4)
        V[f"Ref{tag}Cond"] = f(rf[key]["cond_corr_ndg"], 0)
    for r, t in zip(F["axes"]["substrate_index"], ("A", "B", "C", "D", "E")):
        V[f"Crb{t}"] = f(r["sd_ratio_median"], 2); V[f"Crb{t}IQR"] = ci([r["sd_ratio_p25"], r["sd_ratio_p75"]], 2)
        V[f"Cond{t}"] = f(r["median_cond_corr_ndg"], 0); V[f"CondMax{t}"] = f(r["max_cond_corr_ndg"], 0)
    frc = F["axes"]["fringe_count"]
    V["CrbFrMin"] = f(min(r["sd_ratio_median"] for r in frc), 2); V["CrbFrMax"] = f(max(r["sd_ratio_median"] for r in frc), 2)
    e1f = F["e1_films"]
    V["FilmsRatioMed"] = f(e1f["sd_ratio_median"], 2); V["FilmsRatioNinety"] = f(e1f["sd_ratio_p90"], 2)
    V["FilmsCrbK"] = f(e1f["rms_crb_known"], 4); V["FilmsCrbU"] = f(e1f["rms_crb_unknown"], 4)
    V["FilmsMcK"] = f(e1f["mc_rmse_known"], 4); V["FilmsMcU"] = f(e1f["mc_rmse_profile"], 4)
    for (bk, b), t in zip(e1f["by_true_index"].items(), ("A", "B", "C", "D", "E")):
        V[f"Band{t}Count"] = str(b["count"]); V[f"Band{t}Crb"] = f(b["sd_ratio_median"], 2); V[f"Band{t}Mc"] = f(b["mc_rmse_ratio"], 2)
        V[f"Band{t}Far"] = pct(b["frac_abs_profile_err_gt_0p1"], 0)
    fm = e1f["films_below_1p45_profile_map_above_ns"]; V["FarModeCount"] = str(fm["count"]); V["FarModeOf"] = str(fm["of"])
    gerr = F["gain_error"]
    V["LinPerPct"] = f(gerr["linear_median_abs_bias_n_per_1pct"], 4)
    assert V["LinPerPct"] == V["GEonePct"], (V["LinPerPct"], V["GEonePct"])
    V["PosCoefShare"] = pct(gerr["e3_share_positive_index_coefficient"], 0); V["PosCoefFilms"] = str(gerr["e3_films"])
    V["LinDPerPct"] = f(gerr["linear_median_abs_bias_d_nm_per_1pct"], 2)
    V["RmsBoundB"] = f(F["axes"]["substrate_index"][1]["sd_ratio_rms_bound_ratio"], 2)
    assert F["axes"]["substrate_index"][1]["n_s"] == 1.5 and F["axes"]["substrate_index"][1]["d_band"] == [400, 900]
    V["TolAtFu"] = pct(gerr["tolerance_median"] * 0.0008 / 0.02, 2)      # rescaled to the noise level of Fu et al (2003)
    # the Cramer-Rao half-widths of the reference film agree with its noiseless profile sets to within the 0.0025 index grid step
    pm = S["index_profiles"]["main_regime_ns1p5"]
    assert abs(1.96 * rf["n1p80_d650_ns1p5"]["crb_sd_n_known"] - pm["known_gain_set_width"] / 2) < 0.0025
    assert abs(1.96 * rf["n1p80_d650_ns1p5"]["crb_sd_n_unknown"] - pm["unknown_gain_set_width"] / 2) < 0.0025
    V["HalfKnownA"] = f(1.96 * rf["n1p80_d650_ns1p5"]["crb_sd_n_known"], 4); V["HalfUnknownA"] = f(1.96 * rf["n1p80_d650_ns1p5"]["crb_sd_n_unknown"], 4)
    V["TolMed"] = pct(gerr["tolerance_median"], 1); V["TolTen"] = pct(gerr["tolerance_p10"], 1); V["TolNinety"] = pct(gerr["tolerance_p90"], 1)

    # ---- dispersion
    d1a, d1b = D["D1"]["band_400_900"], D["D1"]["band_1600_2400"]
    q = lambda v, dp: f"{f(100 * v[1], dp)}\\,\\% [{f(100 * v[0], dp)}, {f(100 * v[2], dp)}]"
    V["DoneThinPhase"] = q(d1a["known"]["rel_diff_tau_vs_phase_OT_q25_med_q75"], 1); V["DoneThinGroup"] = q(d1a["known"]["rel_diff_tau_vs_group_OT_q25_med_q75"], 1)
    V["DoneThinCloser"] = pct(d1a["known"]["frac_tau_closer_to_phase_OT"], 0)
    V["DoneThinDn"] = f(d1a["known"]["n_hat_minus_n_ref_q25_med_q75"][1], 4); V["DoneThinDnQ"] = ci([d1a["known"]["n_hat_minus_n_ref_q25_med_q75"][0], d1a["known"]["n_hat_minus_n_ref_q25_med_q75"][2]], 3)
    V["DoneThinLam"] = f(d1a["known"]["lam_star_nm_q25_med_q75"][1], 0); V["DoneThinLamQ"] = ci([d1a["known"]["lam_star_nm_q25_med_q75"][0], d1a["known"]["lam_star_nm_q25_med_q75"][2]], 0)
    V["DoneThinRss"] = f(d1a["known"]["rss_over_L_sigma2_q25_med_q75"][1], 2)
    V["DoneThickPhase"] = q(d1b["known"]["rel_diff_tau_vs_phase_OT_q25_med_q75"], 1); V["DoneThickGroup"] = q(d1b["known"]["rel_diff_tau_vs_group_OT_q25_med_q75"], 1)
    V["DoneThickCloser"] = pct(1 - d1b["known"]["frac_tau_closer_to_phase_OT"], 1)
    V["DoneThickDn"] = f(d1b["known"]["n_hat_minus_n_ref_q25_med_q75"][1], 4)
    V["DoneThickD"] = q(d1b["known"]["rel_diff_d_hat_q25_med_q75"], 1)
    V["DoneThickRss"] = f(d1b["known"]["rss_over_L_sigma2_q25_med_q75"][1], 2)
    V["DoneThinProfDn"] = f(d1a["profile"]["n_hat_minus_n_ref_q25_med_q75"][1], 3); V["DoneThinProfDnQ"] = ci([d1a["profile"]["n_hat_minus_n_ref_q25_med_q75"][0], d1a["profile"]["n_hat_minus_n_ref_q25_med_q75"][2]], 3)
    V["DoneGroupMinusPhase"] = f(d1a["group_minus_phase_index_median"], 3); V["DoneSpan"] = f(d1a["dispersion_span_over_band_pct_median"], 1)
    V["DoneFilms"] = str(D["config"]["D1_films"])
    lam_lo, lam_hi = D["config"]["lam_nm"][0], D["config"]["lam_nm"][1]
    fringes = lambda d: 2.0 * 1.75 * d * (1.0 / lam_lo - 1.0 / lam_hi)       # fringes across the band at the mean index
    V["FringesThinLo"], V["FringesThinHi"] = f(fringes(d1a["d_band"][0]), 1), f(fringes(d1a["d_band"][1]), 1)
    V["FringesThickLo"], V["FringesThickHi"] = f(fringes(d1b["d_band"][0]), 1), f(fringes(d1b["d_band"][1]), 1)
    V["DoneThinLamCount"] = str(D["config"]["D1_films"] - d1a["known"]["n_lam_star_undefined"]); V["DtwoFilms"] = str(D["config"]["D2_films"])
    d2, d2c = D["D2"]["main"], D["D2"]["coarse"]
    V["DtwoKnownRel"] = f(d2["e1_known"]["n"]["rel_rmse"], 4); V["DtwoProfRel"] = f(d2["e1_profile"]["n"]["rel_rmse"], 4); V["DtwoMargRel"] = f(d2["e1_marginal"]["n"]["rel_rmse"], 4)
    rr = d2["ratios"]
    for key, tag in (("e1_profile_over_known_n", "Prof"), ("e1_marginal_over_known_n", "Marg"), ("e1_profile_over_known_d", "ProfD"), ("e1_profile_over_known_tau_g", "ProfTauG"),
                     ("e3_marginal_over_true_gain_n", "ThreeMarg"), ("e3_gain_one_over_true_gain_n", "ThreeOne"), ("e3_marginal_over_true_gain_B", "ThreeMargB")):
        V[f"Dtwo{tag}"] = f(rr[key]["ratio"], 2); V[f"Dtwo{tag}CI"] = ci(rr[key]["ci95"], 2)
    V["DtwoCoarseProf"] = f(d2c["ratios"]["e1_profile_over_known_n"]["ratio"], 2); V["DtwoCoarseMarg"] = f(d2c["ratios"]["e1_marginal_over_known_n"]["ratio"], 2)
    for key, tag in (("e1_known", "Known"), ("e3_true_gain", "True"), ("e3_gain_one", "One"), ("e3_marginal", "Marg")):
        V[f"Dtwo{tag}CovN"] = f(d2[key]["n"]["cov90"], 3); V[f"Dtwo{tag}WidN"] = f(d2[key]["n"]["mean_width90"], 3); V[f"Dtwo{tag}RmseN"] = f(d2[key]["n"]["rmse"], 4)
        V[f"Dtwo{tag}CovB"] = f(d2[key]["B"]["cov90"], 3)
    V["DtwoKnownRmseB"] = f(d2["e1_known"]["B"]["rmse"], 0)
    V["BTrueSD"] = f((D["config"]["B_range_nm2"][1] - D["config"]["B_range_nm2"][0]) / math.sqrt(12), 0)
    V["DtwoFloatN"] = sci(D["float32_check_max_abs_change_in_posterior_mean"]["n"])
    V["DtwoModelErr"] = sci(D["model_check_max_abs_dR"])
    # ---- low noise: direct least-squares fits (rev_lownoise.py)
    L = J("rev_lownoise.json"); Lr = L["by_sigma"]
    assert [r["sigma"] for r in Lr] == [0.001, 0.002, 0.005, 0.01, 0.02] and all(r["fits_converged"] == r["films"] for r in Lr)
    assert all(r["ratio_profile_over_known"]["ci95"][0] <= L["predicted_ratio_rms_crb"] <= r["ratio_profile_over_known"]["ci95"][1] for r in Lr)
    V["LnFilms"] = str(Lr[0]["films"]); V["LnPred"] = f(L["predicted_ratio_rms_crb"], 2)
    rs = [r["ratio_profile_over_known"]["ratio"] for r in Lr]
    V["LnRatioLo"], V["LnRatioHi"] = f(min(rs), 2), f(max(rs), 2)
    dev = max(abs(r[k] - 1) for r in Lr for k in ("rmse_over_rms_crb_known", "rmse_over_rms_crb_unknown"))
    V["LnEffMaxPct"] = f(math.ceil(1000 * dev) / 10, 1)
    dev_low = max(abs(r[k] - 1) for r in Lr if r["sigma"] <= 0.002 for k in ("rmse_over_rms_crb_known", "rmse_over_rms_crb_unknown"))
    V["LnEffLowPct"] = f(math.ceil(1000 * dev_low) / 10, 1)
    for tag, r in zip("ABCDE", Lr):
        V["LnSig" + tag] = f"{r['sigma']:g}"
        V["LnKnown" + tag] = f(r["rmse_n_known"], 5); V["LnProf" + tag] = f(r["rmse_n_profile"], 5)
        V["LnRatio" + tag] = f(r["ratio_profile_over_known"]["ratio"], 2); V["LnRatioCI" + tag] = ci(r["ratio_profile_over_known"]["ci95"], 2)
        V["LnTol" + tag] = pct(r["tolerance_median"], 2)
    V["LnCrbKnownE"], V["LnCrbProfE"] = f(Lr[-1]["rms_crb_known"], 4), f(Lr[-1]["rms_crb_unknown"], 4)
    # the twin with a dispersive film, and what the swap preserves (twin_dispersion.py)
    T = J("twin_dispersion.json"); tf = T["twin_fits"]
    assert T["config"]["n_ref"] == 1.4 and T["config"]["d_nm"] == 700.0 and T["config"]["n_s"] == 2.3 and T["config"]["lam_c_nm"] == D["config"]["lam_c_nm"]
    assert all(t["twin_rss"] < 1e-20 for t in tf if t["B_nm2"] == 0) and all(t["true_side_refit_rss"] < 1e-20 for t in tf)
    base = [t for t in tf if t["gain"] == "known" and t["B_nm2"] > 0]
    assert [t["B_nm2"] for t in base] == [1.2e4, 1.8e4, 2.4e4]
    for t, tag in zip(base, "ABC"):
        V[f"TwinDispEx{tag}"] = f(t["chi2_excess"]["0.02"], 2); V[f"TwinDispExLow{tag}"] = f(t["chi2_excess"]["0.002"], 0)
    rms = [t["twin_rms_dR"] for t in base]
    V["TwinDispRmsLo"], V["TwinDispRmsHi"] = mant(min(rms), 2), mant(max(rms), 2)
    alt = [t["chi2_excess"]["0.002"] for t in tf if t["B_nm2"] > 0] + [t["three_term_twin"]["chi2_excess"]["0.002"] for t in tf if t["B_nm2"] > 0]
    V["TwinDispMinAlt"] = str(math.floor(min(alt)))
    sc = [t["sigma_at_chi2_excess_3p84"] for t in base]
    V["TwinDispSigLo"], V["TwinDispSigHi"] = f(min(sc), 3), f(max(sc), 3)
    sw = T["swap"]
    assert max(sw["max_abs_diff_R_normal"], sw["max_abs_identity_err"], sw["max_abs_R_airy_vs_matrix"], sw["max_abs_complex_r_airy_vs_matrix_normal"]) < 1e-12
    V["TwinPhaseMax"] = f(sw["max_abs_phase_diff_rad_normal"], 2)
    V["TwinPsiMax"], V["TwinDeltaMax"] = f(sw["max_abs_diff_psi_deg"], 0), f(sw["max_abs_diff_delta_deg"], 0)
    # measured spectra of a silicon nitride film on BOROFLOAT 33 (real_data_check.py)
    RD = J("real_data_check.json"); A = RD["A_nitride_on_bf33"]; F = A["fits"]; cfg = RD["config"]
    objs = [o["name"] for o in cfg["objectives"]]
    assert objs == ["10x", "50xUL", "100xUL", "100x"] and [o["NA"] for o in cfg["objectives"]] == [0.25, 0.35, 0.6, 0.9]
    assert [o["backside"] for o in cfg["objectives"]] == [True, False, False, False] and cfg["weightings"][0] == "sin2"
    low = objs[:3]                                              # the objectives up to NA 0.60, where the aperture model is reliable
    V["RDStem"] = f(cfg["d_stem_nm"], 1)
    V["RDLamLo"], V["RDLamHi"] = f(A["data"]["grid_nm"][0], 0), f(A["data"]["grid_nm"][1], 0)
    V["RDLamC"] = f(A["data"]["lam_c_nm"], 0)
    for o, tag in zip(cfg["objectives"], "ABCD"):
        V["RDNa" + tag] = f(o["NA"], 2)
    V["RDBfDev"] = f(cfg["bf33"]["max_abs_dev_from_table"], 5)
    V["RDMatrix"] = sci(A["matrix_check_max_dR"])
    V["RDAngConv"] = sci(max(F[o]["angle_convergence_max_dR"] for o in objs))
    idn = A["identification"]
    V["RDIdDA"], V["RDIdDB"] = f(idn["7a_50xUL_authors_sim"]["d_nm"], 1), f(idn["7a_100xUL_authors_sim"]["d_nm"], 1)
    V["RDIdRms"] = sci(max(idn["7a_50xUL_authors_sim"]["rms"], idn["7a_100xUL_authors_sim"]["rms"]))
    V["RDIdOther"] = f(min(idn["7b_50xUL_authors_sim"]["rms"], idn["7b_100xUL_authors_sim"]["rms"]), 3)
    rms = [F[o]["sin2"][k]["rms"] for o in objs for k in ("fixed", "profiled")]
    V["RDRmsLo"], V["RDRmsHi"] = f(min(rms), 4), f(max(rms), 4)
    nz = list(A["data"]["second_difference_sd"].values())
    V["RDNoiseLo"], V["RDNoiseHi"] = mant(min(nz), 1), mant(max(nz), 1)
    VARS = ("sin2", "sin", "theta", "normal", "sin2_absorbing")
    dtau = lambda o, v: 100 * abs(F[o][v]["profiled"]["tau_c_nm"] - F[o][v]["fixed"]["tau_c_nm"]) / F[o][v]["fixed"]["tau_c_nm"]
    V["RDTauMaxLow"] = f(max(dtau(o, v) for o in low for v in VARS), 2)
    V["RDTauMaxHigh"] = f(max(dtau("100x", v) for v in VARS), 2)
    dn = [abs(F[o]["sin2"]["fixed"]["n_c"] - F[o]["sin2"]["profiled"]["n_c"]) for o in low]
    dd = [abs(F[o]["sin2"]["fixed"]["d_nm"] - F[o]["sin2"]["profiled"]["d_nm"]) for o in low]
    V["RDDnLo"], V["RDDnHi"] = f(min(dn), 3), f(max(dn), 3)
    V["RDDdLo"], V["RDDdHi"] = f(min(dd), 1), f(max(dd), 1)
    V["RDDnFifty"] = f(dn[1], 3)
    fo = [F[o]["first_order"] for o in objs]
    V["RDFoNErr"] = f(max(100 * abs(x["pred_shift_n"] - x["obs_shift_n"]) / abs(x["obs_shift_n"]) for x in fo), 1)
    V["RDFoDErr"] = f(max(100 * abs(x["pred_shift_d_nm"] - x["obs_shift_d_nm"]) / abs(x["obs_shift_d_nm"]) for x in fo), 1)
    kap = [F[o]["kappa"] for o in objs]
    V["RDKapLo"], V["RDKapHi"] = f(min(kap), 1), f(max(kap), 1)
    kp = A["kappa_nondispersive_normal"]; assert list(kp) == ["1.8", "1.9", "2.0"]
    V["RDKapNLo"], V["RDKapNHi"] = "1.8", "2.0"
    V["RDKapPredLo"], V["RDKapPredHi"] = f(min(kp.values()), 1), f(max(kp.values()), 1)
    fr, bk = F["10x"]["sin2_front_only"]["profiled"], F["10x"]["sin2"]["profiled"]
    V["RDFrontG"], V["RDFrontN"] = f(fr["gain"], 2), f(fr["n_c"], 3)
    V["RDFrontDn"] = f(bk["n_c"] - fr["n_c"], 2)
    V["RDFrontTau"] = f(100 * abs(fr["tau_c_nm"] - bk["tau_c_nm"]) / bk["tau_c_nm"], 1)
    AV = ("sin2", "sin", "theta", "sin2_absorbing")              # every aperture-modelled variant, with or without absorption
    fixd = [F[o][v]["fixed"]["d_nm"] for o in low for v in AV]
    prod = [F[o][v]["profiled"]["d_nm"] for o in low for v in AV]
    prog = [F[o][v]["profiled"]["gain"] for o in low for v in AV]
    V["RDFixDLo"], V["RDFixDHi"] = f(min(fixd), 1), f(max(fixd), 1)
    V["RDFixDErr"] = f(max(100 * abs(x - cfg["d_stem_nm"]) / cfg["d_stem_nm"] for x in fixd), 1)
    V["RDProfDLo"], V["RDProfDHi"] = f(min(prod), 1), f(max(prod), 1)
    V["RDProfGLo"], V["RDProfGHi"] = f(min(prog), 3), f(max(prog), 3)
    V["RDGNoAbs"], V["RDGAbs"] = f(F["50xUL"]["sin2"]["profiled"]["gain"], 3), f(F["50xUL"]["sin2_absorbing"]["profiled"]["gain"], 3)
    V["RDKAbs"] = f(F["50xUL"]["sin2_absorbing"]["profiled"]["k"], 4)
    sl = [F[o]["first_order"]["slope_n_per_pct"] for o in low]
    V["RDSlopeLo"], V["RDSlopeHi"] = f(min(sl), 4), f(max(sl), 4)
    hi = [F["100x"][v]["fixed"]["d_nm"] for v in ("sin2", "sin", "theta")]
    V["RDHiNaDLo"], V["RDHiNaDHi"] = f(min(hi), 1), f(max(hi), 1)
    for o, tag in zip(objs, "ABCD"):                            # table: primary model
        a, b = F[o]["sin2"]["fixed"], F[o]["sin2"]["profiled"]
        V["RDTabFixN" + tag], V["RDTabFixD" + tag] = f(a["n_c"], 4), f(a["d_nm"], 1)
        V["RDTabProfG" + tag], V["RDTabProfN" + tag], V["RDTabProfD" + tag] = f(b["gain"], 3), f(b["n_c"], 4), f(b["d_nm"], 1)
        V["RDTabKap" + tag] = f(F[o]["kappa"], 1)
    return V


def main():
    V = build()
    lines = ["% generated by code/make_mst_numbers.py from data/results/*.json; do not edit"]
    for k in sorted(V):
        assert k.isalpha(), k
        lines.append(f"\\newcommand{{\\V{k}}}{{{V[k]}}}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"{len(V)} macros -> {os.path.relpath(OUT)}")


if __name__ == "__main__":
    main()
