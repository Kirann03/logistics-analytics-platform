from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from functools import lru_cache

from src.lookup import LOCATION_COORDINATES, PRODUCT_FACTORY_LOOKUP, STATE_CODE_LOOKUP


REQUIRED_ORDER_COLUMNS = {
    "Order ID": "order_id",
    "Order Date": "order_date",
    "Ship Date": "ship_date",
    "Ship Mode": "ship_mode",
    "Country/Region": "country_region",
    "City": "city",
    "State/Province": "state",
    "Region": "region",
    "Product Name": "product_name",
    "Sales": "sales",
    "Units": "units",
    "Gross Profit": "gross_profit",
    "Cost": "cost",
}

OPTIONAL_ORDER_COLUMNS = {
    "Customer ID": "customer_id",
    "Division": "division",
    "Postal Code": "postal_code",
    "Product ID": "product_id",
    "Factory": "factory",
}

FIELD_ALIASES = {
    "order_id": ["order id", "order_id", "id", "shipment_id"],
    "order_date": ["order date", "order_date", "booking date", "shipment date", "created_at"],
    "ship_date": ["ship date", "ship_date", "delivery date", "dispatch date", "delivered_at"],
    "lead_time_days": ["lead_time", "lead time", "lead_time_days", "lead days", "days", "delivery_time"],
    "ship_mode": ["ship mode", "ship_mode", "shipping mode", "mode", "vehicle"],
    "country_region": ["country/region", "country region", "country", "destination_country"],
    "city": ["city", "destination city", "destination"],
    "state": ["state/province", "state province", "state", "province", "destination_state"],
    "region": ["region", "zone", "corridor"],
    "product_name": ["product name", "product_name", "product", "item", "sku_name"],
    "sales": ["sales", "revenue", "amount"],
    "units": ["units", "quantity", "qty", "volume"],
    "gross_profit": ["gross profit", "gross_profit", "profit", "margin_value"],
    "cost": ["cost", "shipping_cost", "expense"],
    "factory": ["factory", "warehouse", "warehouse or factory", "origin", "origin_factory"],
    "factory_lat": ["factory_lat", "factory latitude", "origin_lat", "origin latitude"],
    "factory_lon": ["factory_lon", "factory longitude", "origin_lon", "origin longitude"],
    "route_distance_km": ["distance", "route_distance_km", "distance_km", "miles", "kilometers"],
    "traffic_condition": ["traffic", "traffic_condition", "congestion", "congestion_level"],
    "weather_condition": ["weather", "weather_condition"],
    "customer_id": ["customer id", "customer_id", "customer"],
}

MINIMAL_IMPORT_FIELDS = {"order_date", "region", "state", "ship_mode", "units"}
ALTERNATE_MINIMAL_FIELDS = {"factory", "city", "ship_mode"}
SUPERVISED_GENERIC_LOGISTICS_FIELDS = {"lead_time_days", "ship_mode", "route_distance_km"}


@dataclass(frozen=True)
class DataBundle:
    orders: pd.DataFrame
    factories: pd.DataFrame
    products: pd.DataFrame
    quality_report: dict[str, int | str]
    data_source: str
    validation_messages: tuple[str, ...]


def normalize_text(series: pd.Series) -> pd.Series:
    text = series.copy()
    non_null = text.notna()
    text.loc[non_null] = (
        text.loc[non_null]
        .astype(str)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return text


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = sin(d_lat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    return 2 * radius * asin(sqrt(a))


def _load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path.name}")
    return pd.read_excel(path)


def parse_flexible_dates(series: pd.Series) -> pd.Series:
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    if parsed.isna().all():
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=True)
    return parsed


def _read_uploaded_frame(file_bytes: bytes, filename: str) -> pd.DataFrame:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(BytesIO(file_bytes))
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(BytesIO(file_bytes))
    raise ValueError("Uploaded dataset must be a CSV or Excel file.")


def _normalize_name(name: str) -> str:
    return str(name).strip().lower().replace("_", " ").replace("/", " ").replace("-", " ")


def _map_unknown_columns(raw_orders: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    mapped = raw_orders.copy()
    normalized = {_normalize_name(column): column for column in raw_orders.columns}
    applied = []
    rename_map: dict[str, str] = {}
    for target, aliases in FIELD_ALIASES.items():
        for alias in aliases:
            original = normalized.get(_normalize_name(alias))
            if original and original not in rename_map:
                rename_map[original] = target
                applied.append(f"Mapped {original} -> {target}")
                break
    if rename_map:
        mapped = mapped.rename(columns=rename_map)
    return mapped, applied


def validate_raw_orders(raw_orders: pd.DataFrame) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    canonical = set(raw_orders.columns)
    has_primary = MINIMAL_IMPORT_FIELDS.issubset(canonical)
    has_alternate = ALTERNATE_MINIMAL_FIELDS.issubset(canonical)
    has_generic_supervised = SUPERVISED_GENERIC_LOGISTICS_FIELDS.issubset(canonical)
    if not has_primary and not has_alternate and not has_generic_supervised:
        errors.append(
            "Uploaded dataset must include either order_date, region, state, ship_mode, units; "
            "an alternate logistics schema with factory/origin, city/destination, and ship_mode/vehicle; "
            "or a supervised delivery schema with distance, ship_mode/vehicle, and lead_time/delivery_time."
        )
    duplicates = int(raw_orders.duplicated(subset=["order_id"]).sum()) if "order_id" in raw_orders.columns else 0
    if duplicates:
        warnings.append(f"{duplicates} duplicate order IDs detected in the uploaded dataset.")
    return errors, warnings


def _reference_defaults(reference_orders: pd.DataFrame) -> dict[str, object]:
    ref = reference_orders.copy()
    route_stats = ref.groupby("route_label")["lead_time_days"].mean().to_dict()
    state_mode_stats = ref.groupby(["state", "ship_mode"])["lead_time_days"].mean().to_dict()
    state_stats = ref.groupby("state")["lead_time_days"].mean().to_dict()
    mode_stats = ref.groupby("ship_mode")["lead_time_days"].mean().to_dict()
    factory_cost = ref.groupby("factory")["cost"].mean().to_dict()
    mode_sales = ref.groupby("ship_mode")["sales"].mean().to_dict()
    mode_profit = ref.groupby("ship_mode")["gross_profit"].mean().to_dict()
    global_defaults = {
        "lead_time_days": float(ref["lead_time_days"].mean()),
        "sales": float(ref["sales"].mean()),
        "cost": float(ref["cost"].mean()),
        "gross_profit": float(ref["gross_profit"].mean()),
        "route_stats": route_stats,
        "state_mode_stats": state_mode_stats,
        "state_stats": state_stats,
        "mode_stats": mode_stats,
        "factory_cost": factory_cost,
        "mode_sales": mode_sales,
        "mode_profit": mode_profit,
    }
    return global_defaults


def _infer_missing_fields(orders: pd.DataFrame, reference_orders: pd.DataFrame | None) -> tuple[pd.DataFrame, list[str]]:
    inferred_messages: list[str] = []
    defaults = _reference_defaults(reference_orders) if reference_orders is not None and not reference_orders.empty else None

    if "order_id" not in orders.columns:
        orders["order_id"] = [f"UP-{index+1:06d}" for index in range(len(orders))]
        inferred_messages.append("Generated synthetic order IDs for uploaded rows.")

    if "factory" not in orders.columns:
        if "product_name" in orders.columns:
            orders["factory"] = orders["product_name"].map(PRODUCT_FACTORY_LOOKUP)
        elif {"traffic_condition", "weather_condition"}.intersection(orders.columns):
            traffic_part = orders.get("traffic_condition", pd.Series(["Unknown"] * len(orders), index=orders.index)).fillna("Unknown").astype(str)
            weather_part = orders.get("weather_condition", pd.Series(["Unknown"] * len(orders), index=orders.index)).fillna("Unknown").astype(str)
            orders["factory"] = "Imported Hub | " + traffic_part.str.title() + " | " + weather_part.str.title()
            inferred_messages.append("Synthesized factory labels from traffic and weather fields for generic delivery data.")
        else:
            orders["factory"] = pd.NA
    orders["factory"] = orders["factory"].fillna("Imported Logistics Hub")

    if "city" not in orders.columns:
        if "state" in orders.columns:
            orders["city"] = orders["state"]
        else:
            orders["city"] = "Imported Destination"
            inferred_messages.append("Created a placeholder destination city for generic delivery data.")

    if "state" not in orders.columns:
        if "city" in orders.columns and orders["city"].notna().any():
            orders["state"] = normalize_text(orders["city"]).str.upper()
            inferred_messages.append("Mapped destination labels into state because no state column was provided.")
        else:
            orders["state"] = "IMPORTED DESTINATION"
            inferred_messages.append("Created a placeholder state because no destination geography was provided.")

    if "region" not in orders.columns:
        orders["region"] = "Imported"
        inferred_messages.append("Filled missing region values as Imported.")

    if "order_date" not in orders.columns:
        synthetic_dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=max(len(orders), 1), freq="D")
        orders["order_date"] = pd.Series(synthetic_dates, index=orders.index)
        inferred_messages.append("Generated synthetic order dates because none were provided in the uploaded dataset.")

    if "product_name" not in orders.columns:
        orders["product_name"] = orders["factory"].replace("Unknown Factory", "Unknown Product")
        inferred_messages.append("Filled missing product names using factory labels.")

    if "country_region" not in orders.columns:
        orders["country_region"] = orders["state"].map(lambda value: LOCATION_COORDINATES.get(str(value).upper(), {}).get("country", "Unknown"))
        orders["country_region"] = orders["country_region"].fillna("Unknown")
        inferred_messages.append("Inferred destination country from state or province where possible.")

    if "units" not in orders.columns:
        orders["units"] = 1
        inferred_messages.append("Defaulted missing units to 1 for uploaded logistics records.")

    if "sales" not in orders.columns:
        if defaults:
            orders["sales"] = orders["ship_mode"].map(defaults["mode_sales"]).fillna(defaults["sales"])
        else:
            orders["sales"] = 0.0
        inferred_messages.append("Estimated missing sales values from historical ship-mode averages.")

    if "cost" not in orders.columns:
        if defaults:
            orders["cost"] = orders["factory"].map(defaults["factory_cost"]).fillna(defaults["cost"])
        else:
            orders["cost"] = 0.0
        inferred_messages.append("Estimated missing cost values from historical factory averages.")

    if "gross_profit" not in orders.columns:
        if defaults:
            estimated_profit = orders["ship_mode"].map(defaults["mode_profit"]).fillna(defaults["gross_profit"])
            orders["gross_profit"] = estimated_profit
        else:
            orders["gross_profit"] = orders["sales"] - orders["cost"]
        inferred_messages.append("Estimated missing gross profit values from historical patterns.")

    if "lead_time_days" not in orders.columns and "ship_date" not in orders.columns:
        if defaults:
            route_labels = orders["factory"].astype(str) + " -> " + orders["state"].astype(str)
            lead_estimate = route_labels.map(defaults["route_stats"])
            state_mode_estimate = pd.Series(
                [defaults["state_mode_stats"].get((state, mode), np.nan) for state, mode in zip(orders["state"], orders["ship_mode"])],
                index=orders.index,
            )
            lead_estimate = lead_estimate.fillna(state_mode_estimate)
            lead_estimate = lead_estimate.fillna(orders["state"].map(defaults["state_stats"]))
            lead_estimate = lead_estimate.fillna(orders["ship_mode"].map(defaults["mode_stats"]))
            lead_estimate = lead_estimate.fillna(defaults["lead_time_days"])
            orders["lead_time_days"] = lead_estimate.round().clip(lower=0)
            inferred_messages.append("Estimated lead times from historical route, state, and ship-mode patterns.")
        else:
            orders["lead_time_days"] = 7
            inferred_messages.append("No historical baseline found; defaulted lead time to 7 days.")

    if "ship_date" not in orders.columns:
        orders["ship_date"] = parse_flexible_dates(orders["order_date"]) + pd.to_timedelta(pd.to_numeric(orders["lead_time_days"], errors="coerce").fillna(0), unit="D")
        inferred_messages.append("Generated ship dates using inferred or provided lead times.")

    return orders, inferred_messages


def _build_bundle_from_frames(
    raw_orders: pd.DataFrame,
    factories: pd.DataFrame,
    products: pd.DataFrame,
    *,
    data_source: str,
    reference_orders: pd.DataFrame | None = None,
) -> DataBundle:
    mapped_orders, mapping_messages = _map_unknown_columns(raw_orders)
    errors, warnings = validate_raw_orders(mapped_orders)
    if errors:
        raise ValueError(" ; ".join(errors))

    orders = mapped_orders.copy()
    orders, inferred_messages = _infer_missing_fields(orders, reference_orders)

    orders["product_name"] = normalize_text(orders["product_name"])
    orders["state"] = normalize_text(orders["state"]).str.upper()
    orders["region"] = normalize_text(orders["region"]).str.title()
    orders["city"] = normalize_text(orders["city"]).str.title()
    orders["ship_mode"] = normalize_text(orders["ship_mode"]).str.title()
    orders["country_region"] = normalize_text(orders["country_region"]).str.title()
    orders["factory"] = normalize_text(orders["factory"])

    for column in ["sales", "units", "gross_profit", "cost", "lead_time_days", "route_distance_km", "factory_lat", "factory_lon"]:
        if column in orders.columns:
            orders[column] = pd.to_numeric(orders[column], errors="coerce")

    orders["order_date"] = parse_flexible_dates(orders["order_date"])
    orders["ship_date"] = parse_flexible_dates(orders["ship_date"])
    if "lead_time_days" not in orders.columns or orders["lead_time_days"].isna().all():
        orders["lead_time_days"] = (orders["ship_date"] - orders["order_date"]).dt.days
    else:
        computed_lead = (orders["ship_date"] - orders["order_date"]).dt.days
        orders["lead_time_days"] = orders["lead_time_days"].fillna(computed_lead)

    missing_dates = int(orders["order_date"].isna().sum() + orders["ship_date"].isna().sum())
    negative_lead_times = int((orders["lead_time_days"] < 0).fillna(False).sum())

    cleaned_orders = orders.dropna(
        subset=["order_id", "order_date", "ship_date", "state", "region", "ship_mode", "factory", "lead_time_days", "units"]
    ).copy()
    cleaned_orders = cleaned_orders[cleaned_orders["lead_time_days"] >= 0].copy()
    cleaned_orders = cleaned_orders.drop_duplicates(subset=["order_id"]).copy()

    if "sales" not in cleaned_orders.columns:
        cleaned_orders["sales"] = 0.0
    if "cost" not in cleaned_orders.columns:
        cleaned_orders["cost"] = 0.0
    if "gross_profit" not in cleaned_orders.columns:
        cleaned_orders["gross_profit"] = cleaned_orders["sales"] - cleaned_orders["cost"]

    cleaned_orders["delay_flag"] = False
    cleaned_orders["route_label"] = cleaned_orders["factory"] + " -> " + cleaned_orders["state"]
    cleaned_orders["factory_region_route"] = cleaned_orders["factory"] + " -> " + cleaned_orders["region"]
    cleaned_orders["order_month"] = cleaned_orders["order_date"].dt.to_period("M").dt.to_timestamp()
    cleaned_orders["state_code"] = cleaned_orders["state"].map(STATE_CODE_LOOKUP)
    cleaned_orders["dest_lat"] = cleaned_orders["state"].map(lambda value: LOCATION_COORDINATES.get(value, {}).get("lat"))
    cleaned_orders["dest_lon"] = cleaned_orders["state"].map(lambda value: LOCATION_COORDINATES.get(value, {}).get("lon"))
    cleaned_orders["destination_country"] = cleaned_orders["state"].map(lambda value: LOCATION_COORDINATES.get(value, {}).get("country"))

    factories.columns = [column.strip().lower().replace(" ", "_") for column in factories.columns]
    products.columns = [column.strip().lower().replace(" ", "_") for column in products.columns]
    if "product_name" in products.columns:
        products["product_name"] = normalize_text(products["product_name"])
    if "factory" in products.columns:
        products["factory"] = normalize_text(products["factory"])

    uploaded_factory_coords = cleaned_orders[["factory", "factory_lat", "factory_lon"]].dropna() if {"factory_lat", "factory_lon"}.issubset(cleaned_orders.columns) else pd.DataFrame(columns=["factory", "factory_lat", "factory_lon"])
    factory_coords = factories.rename(columns={"latitude": "factory_lat", "longitude": "factory_lon"})
    combined_factory_coords = pd.concat([factory_coords[["factory", "factory_lat", "factory_lon"]], uploaded_factory_coords], ignore_index=True).drop_duplicates(subset=["factory"], keep="last")
    cleaned_orders = cleaned_orders.drop(columns=[column for column in ["factory_lat", "factory_lon"] if column in cleaned_orders.columns])
    cleaned_orders = cleaned_orders.merge(combined_factory_coords, on="factory", how="left")

    coords_mask = cleaned_orders[["factory_lat", "factory_lon", "dest_lat", "dest_lon"]].notna().all(axis=1)
    if "route_distance_km" not in cleaned_orders.columns:
        cleaned_orders["route_distance_km"] = pd.NA
    cleaned_orders.loc[coords_mask & cleaned_orders["route_distance_km"].isna(), "route_distance_km"] = cleaned_orders.loc[coords_mask & cleaned_orders["route_distance_km"].isna()].apply(
        lambda row: haversine_km(float(row["factory_lat"]), float(row["factory_lon"]), float(row["dest_lat"]), float(row["dest_lon"])),
        axis=1,
    )

    quality_report = {
        "removed_rows": int(len(raw_orders) - len(cleaned_orders)),
        "missing_dates": missing_dates,
        "negative_lead_times": negative_lead_times,
        "lead_time_over_60_days": int((cleaned_orders["lead_time_days"] > 60).sum()),
        "duplicate_orders_removed": int(orders.duplicated(subset=["order_id"]).sum()) if "order_id" in orders.columns else 0,
    }

    return DataBundle(
        orders=cleaned_orders,
        factories=factories,
        products=products,
        quality_report=quality_report,
        data_source=data_source,
        validation_messages=tuple(warnings + mapping_messages + inferred_messages),
    )


@lru_cache(maxsize=2)
def load_data_bundle(project_root: Path) -> DataBundle:
    orders_path = project_root / "data.xlsx"
    factories_path = project_root / "Factories Coordinates.xlsx"
    products_path = project_root / "Products and Factories Correlation.xlsx"
    ingest_path = project_root / "incoming_shipments.jsonl"

    raw_orders = _load_excel(orders_path)
    if ingest_path.exists():
        incoming = pd.read_json(ingest_path, lines=True)
        rename_back = {
            "order_id": "Order ID",
            "order_date": "Order Date",
            "ship_date": "Ship Date",
            "ship_mode": "Ship Mode",
            "country_region": "Country/Region",
            "city": "City",
            "state": "State/Province",
            "region": "Region",
            "product_name": "Product Name",
            "sales": "Sales",
            "units": "Units",
            "gross_profit": "Gross Profit",
            "cost": "Cost",
            "factory": "Factory",
            "customer_id": "Customer ID",
        }
        incoming = incoming.rename(columns=rename_back)
        raw_orders = pd.concat([raw_orders, incoming], ignore_index=True, sort=False)
    factories = _load_excel(factories_path)
    products = _load_excel(products_path)
    return _build_bundle_from_frames(raw_orders, factories, products, data_source="Project workbook" + (" + live ingested rows" if ingest_path.exists() else ""))


def load_uploaded_data_bundle(project_root: Path, file_bytes: bytes, filename: str) -> DataBundle:
    factories_path = project_root / "Factories Coordinates.xlsx"
    products_path = project_root / "Products and Factories Correlation.xlsx"
    raw_orders = _read_uploaded_frame(file_bytes, filename)
    factories = _load_excel(factories_path)
    products = _load_excel(products_path)
    reference_bundle = load_data_bundle(project_root)
    return _build_bundle_from_frames(
        raw_orders,
        factories,
        products,
        data_source=f"Uploaded dataset ({filename})",
        reference_orders=reference_bundle.orders,
    )
