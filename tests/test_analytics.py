import pandas as pd

from src.analytics import (
    bottleneck_score,
    build_recommendation_actions,
    build_route_summary,
    minmax,
)


def sample_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["1", "2", "3", "4"],
            "route_label": ["A -> X", "A -> X", "B -> Y", "C -> Z"],
            "factory": ["A", "A", "B", "C"],
            "state": ["X", "X", "Y", "Z"],
            "region": ["North", "North", "South", "West"],
            "lead_time_days": [4, 6, 12, 8],
            "delay_flag": [0, 1, 1, 0],
            "sales": [100, 150, 220, 180],
            "cost": [40, 55, 90, 70],
            "gross_profit": [60, 95, 130, 110],
            "units": [2, 3, 4, 2],
        }
    )


def test_minmax_returns_ones_for_constant_series():
    result = minmax(pd.Series([5, 5, 5]))
    assert result.tolist() == [1.0, 1.0, 1.0]


def test_bottleneck_score_prefers_higher_pressure():
    score = bottleneck_score(pd.Series([2, 10]), pd.Series([10, 100]), pd.Series([0.1, 0.9]))
    assert score.iloc[1] > score.iloc[0]


def test_build_route_summary_has_expected_columns():
    summary = build_route_summary(sample_orders())
    expected = {"route_label", "shipments", "avg_lead_time", "delay_rate", "efficiency_score", "revenue_at_risk"}
    assert expected.issubset(summary.columns)
    assert summary.loc[summary["route_label"] == "A -> X", "shipments"].iloc[0] == 2


def test_build_recommendation_actions_returns_four_actions():
    orders = sample_orders()
    route_summary = build_route_summary(orders)
    state_bottlenecks = pd.DataFrame(
        {"state": ["Y"], "avg_lead_time": [12.0], "shipments": [1], "delay_rate": [1.0]}
    )
    factory_summary = pd.DataFrame(
        {"factory": ["A"], "performance_score": [99.0]}
    )
    ship_mode_summary = pd.DataFrame(
        {"ship_mode": ["First Class"], "avg_lead_time": [3.0]}
    )
    actions = build_recommendation_actions(route_summary, state_bottlenecks, factory_summary, ship_mode_summary)
    assert len(actions) == 4
    assert all(isinstance(action, str) and action for action in actions)
