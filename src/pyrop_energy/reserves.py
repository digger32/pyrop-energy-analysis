"""Three orthogonal energy-saving reserves on the supply-side balance boundary.

The balance boundary is the 6 kV switchboard: the annual active energy
W_year is the sum of the two grid incomers (bus 1, bus 2) and the two
cogeneration cells (gen cell 2, gen cell 23), annualised from the
near-complete 2023 hourly panel (8,712 of 8,760 hours). Internal outgoing
feeders (cell 307, cell 402) are a subset of this consumption and are NOT
added to the base — summing all six flows would double-count energy.

The three reserves (manuscript headline: 1.22 GWh/year, ~1.0% of the
annual consumption of 126.7 GWh, 487 t CO2/year at 0.4 kg CO2/kWh):

1. Reactive compensation on the grid incomers, raising their operating
   power factor to the regulatory target 0.95. The loss-reduction factor
   is computed PER INCOMER from the hourly joint distribution of P and Q:

       factor = 1 - sum_t(P_t^2 + Q95_t^2) / sum_t(P_t^2 + Q_t^2),
       Q95_t  = P_t * tan(arccos 0.95),

   i.e. the reduction of the current-dependent loss integral
   (proportional to sum_t S_t^2) at constant active power. The factor is
   applied to a baseline loss share (default 5%) of the annual energy
   imported through that incomer only. Power-factor correction does not
   reduce the useful active energy of the process; no-load transformer
   losses, released capacity, tariff effects, and the losses of the
   compensation equipment itself are excluded from the screening estimate.
2. Elimination of the parasitic 200 kW load drawn by the SAG-mill VFD
   during mill stoppages (~1,500 h/year).
3. Modernisation of the 6/0.4 kV transformer fleet: a 20% loss reduction
   on a fleet carrying a 30% nominal share of the plant load, with the
   same 5% loss share of the annual supply.

The reserves affect different subsystems (grid incomers, mill drive,
transformer fleet), so their effects sum without double counting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HOURS_PER_YEAR = 8760


@dataclass
class ReserveAssumptions:
    """Engineering assumptions for the three reserves.

    Defaults reproduce the headline figures of the manuscript
    (1.22 GWh/year, 487 t CO2/year). All values can be overridden per-site.
    """
    # Supply-side balance boundary
    incomers:       tuple = ("bus_1", "bus_2")
    supply_sources: tuple = ("bus_1", "bus_2", "gen_cell_2", "gen_cell_23")
    # Power-factor reserve
    cos_phi_new:    float = 0.95
    loss_share:     float = 0.05   # baseline current-dependent loss share
    # VFD idle reserve
    vfd_idle_kw:         float = 200.0
    vfd_idle_h_per_year: float = 1500.0
    # Transformer-fleet reserve
    transformer_load_share:     float = 0.30
    transformer_loss_reduction: float = 0.20
    # Reference values
    co2_kg_per_kwh:        float = 0.40
    electricity_price_rub: float = 6.0


def _merge_blocks(panel: pd.DataFrame, source: str) -> pd.Series:
    """Merge the two archival block columns of one source into one series."""
    cols = [c for c in panel.columns if c.startswith(source)]
    if not cols:
        raise KeyError(f"source {source!r} not found in panel")
    out = panel[cols[0]]
    for c in cols[1:]:
        out = out.combine_first(panel[c])
    return out


def annual_energy_kwh(panel_p: pd.DataFrame, source: str) -> float:
    """Annualised active energy of one source from the hourly panel, kWh."""
    p = _merge_blocks(panel_p, source).dropna()
    return float(p.sum() * HOURS_PER_YEAR / len(p))


def pf_loss_reduction_factor(p: pd.Series, q: pd.Series,
                             cos_phi_new: float = 0.95) -> float:
    """Loss-reduction factor 1 - sum(P^2+Q95^2)/sum(P^2+Q^2)."""
    d = pd.DataFrame({"P": p, "Q": q}).dropna()
    q95 = d["P"] * np.tan(np.arccos(cos_phi_new))
    return float(1.0 - ((d["P"] ** 2 + q95 ** 2).sum()
                        / (d["P"] ** 2 + d["Q"] ** 2).sum()))


def annual_ratio_power_factor(p: pd.Series, q: pd.Series) -> float:
    """Annual ratio-based power factor sum(P_t) / sum(S_t)."""
    d = pd.DataFrame({"P": p, "Q": q}).dropna()
    return float(d["P"].sum() / np.sqrt(d["P"] ** 2 + d["Q"] ** 2).sum())


def compute_reserves(panel_p: pd.DataFrame,
                     panel_q: pd.DataFrame,
                     a: ReserveAssumptions | None = None) -> pd.DataFrame:
    """Compute the three reserves from the hourly P and Q panels.

    Parameters
    ----------
    panel_p, panel_q : pd.DataFrame
        Hourly panels with one column per source-block (e.g. "bus_2_july").
    a : ReserveAssumptions, optional

    Returns
    -------
    pd.DataFrame
        Rows: one per reserve plus a TOTAL row.
        Columns: reserve, saving_kwh_per_year, saving_pct_of_total,
                 saving_rub_per_year, co2_avoided_t_per_year.
    """
    if a is None:
        a = ReserveAssumptions()

    annual_kwh = sum(annual_energy_kwh(panel_p, s) for s in a.supply_sources)

    # Reserve 1: per-incomer power-factor loss reduction
    pf_saving = 0.0
    pf_details = []
    for inc in a.incomers:
        p = _merge_blocks(panel_p, inc)
        q = _merge_blocks(panel_q, inc)
        factor = pf_loss_reduction_factor(p, q, a.cos_phi_new)
        e_inc = annual_energy_kwh(panel_p, inc)
        pf_saving += a.loss_share * e_inc * factor
        pf_details.append(f"{inc}: pf={annual_ratio_power_factor(p, q):.2f}, "
                          f"factor={factor:.3f}")

    vfd_saving = a.vfd_idle_kw * a.vfd_idle_h_per_year
    transformer_saving = (a.loss_share * annual_kwh
                          * a.transformer_load_share
                          * a.transformer_loss_reduction)

    rows = [
        (f"Power factor to {a.cos_phi_new:.2f} on incomers "
         f"({'; '.join(pf_details)})", pf_saving),
        (f"Idle VFD shutdown ({a.vfd_idle_kw:.0f} kW x "
         f"{a.vfd_idle_h_per_year:.0f} h)", vfd_saving),
        ("Transformer modernization (-20% losses, 30% load share)",
         transformer_saving),
    ]

    out_rows = []
    for name, kwh in rows:
        out_rows.append({
            "reserve":                name,
            "saving_kwh_per_year":    kwh,
            "saving_pct_of_total":    100 * kwh / annual_kwh,
            "saving_rub_per_year":    kwh * a.electricity_price_rub,
            "co2_avoided_t_per_year": kwh * a.co2_kg_per_kwh / 1000.0,
        })
    total_kwh = sum(r["saving_kwh_per_year"] for r in out_rows)
    out_rows.append({
        "reserve":                "TOTAL",
        "saving_kwh_per_year":    total_kwh,
        "saving_pct_of_total":    100 * total_kwh / annual_kwh,
        "saving_rub_per_year":    total_kwh * a.electricity_price_rub,
        "co2_avoided_t_per_year": total_kwh * a.co2_kg_per_kwh / 1000.0,
    })
    df = pd.DataFrame(out_rows)
    df.attrs["annual_consumption_kwh"] = annual_kwh
    return df
