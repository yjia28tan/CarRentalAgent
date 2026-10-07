"""Rendering of the agent's structured response (cards, sources, tool trace). Formatting is pure and tested."""
from datetime import datetime
from typing import Any

import streamlit as st


def money(amount: float | None, currency: str | None) -> str:
    if amount is None:
        return "-"
    return f"{currency or ''} {amount:,.2f}".strip()


def signed_money(amount: float, currency: str) -> str:
    return ("+" if amount > 0 else "") + money(amount, currency)


def when(iso: str | None) -> str:
    if not iso:
        return "-"
    try:
        return datetime.fromisoformat(iso).strftime("%a %b %d %Y, %H:%M")
    except ValueError:
        return iso


def vehicle_option_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"Model": o["name"], "Category": o["category"], "Seats": o["seats"], "Large bags": o["large_bags"],
         "Small bags": o["small_bags"], "Available": o["available_count"],
         "Daily rate": money(o["daily_rate"], o["currency"])}
        for o in data.get("options", [])
    ]


def comparison_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    """One row per attribute, one column per model."""
    models = data.get("models", [])
    attributes = [("Category", "category"), ("Seats", "seats"), ("Large bags", "large_bags"),
                  ("Small bags", "small_bags"), ("Transmission", "transmission"), ("Fuel", "fuel_type")]
    rows = [{"Spec": label, **{m["name"]: m.get(key) for m in models}} for label, key in attributes]
    feature_names = sorted({f for m in models for f in (m.get("features") or {})})
    rows += [{"Spec": f.replace("_", " "), **{m["name"]: (m.get("features") or {}).get(f) for m in models}}
             for f in feature_names]
    return [{k: ("-" if v is None else str(v)) for k, v in row.items()} for row in rows]


def quote_rows(data: dict[str, Any]) -> list[dict[str, str]]:
    b = data.get("breakdown", {})
    currency = data.get("currency")
    rows = [{"Item": f"Base: {b.get('days')} day(s) x {money(b.get('daily_rate'), currency)}",
             "Amount": money(b.get("base"), currency)}]
    rows += [{"Item": f"{i['code']} ({money(i['daily_price'], currency)}/day)", "Amount": money(i["amount"], currency)}
             for i in b.get("insurance", [])]
    rows.append({"Item": "Taxes and fees", "Amount": money(b.get("taxes_fees"), currency)})
    rows.append({"Item": "Total", "Amount": money(data.get("total"), currency)})
    return rows


def reservation_card_from_api(r: dict[str, Any]) -> dict[str, Any]:
    """Card shape for a reservation object returned by /actions/{id}/confirm."""
    return {"type": "reservation", "data": {
        "confirmation_code": r["confirmation_code"], "status": r["status"], "vehicle": r["vehicle"]["name"],
        "pickup_location": r["pickup_location"]["name"], "pickup_at": r["pickup_at"], "return_at": r["return_at"],
        "payment_status": r["payment_status"], "total_amount": r["total_amount"], "currency": r["currency"],
        "insurance": [p["name"] for p in r.get("insurance", [])],
    }}


# ---------------------------------------------------------------- streamlit renderers


def _render_vehicle_options(data: dict[str, Any]) -> None:
    st.markdown(f"**Available vehicles** at {data['location']['name']}")
    st.caption(f"{when(data.get('pickup_at'))} to {when(data.get('return_at'))} ({data.get('days')} days)")
    rows = vehicle_option_rows(data)
    if rows:
        st.dataframe(rows, hide_index=True)
    else:
        st.info("No vehicle matches all requirements.")


def _render_comparison(data: dict[str, Any]) -> None:
    st.markdown("**Vehicle comparison**")
    st.dataframe(comparison_rows(data), hide_index=True)


def _render_quote(data: dict[str, Any]) -> None:
    st.markdown(f"**Price quote**: {data.get('model', '')}")
    st.caption(f"{when(data.get('pickup_at'))} to {when(data.get('return_at'))}")
    st.metric("Total", money(data.get("total"), data.get("currency")))
    st.dataframe(quote_rows(data), hide_index=True)
    st.caption(f"Quote {data.get('quote_id')} expires {when(data.get('expires_at'))}")


def _render_reservation(data: dict[str, Any]) -> None:
    with st.container(border=True):
        st.markdown(f"**Reservation {data['confirmation_code']}** ({data['status']})")
        st.write(f"{data['vehicle']} at {data.get('pickup_location', '')}")
        st.write(f"Pickup: {when(data['pickup_at'])}  \nReturn: {when(data['return_at'])}")
        st.write(f"Total: {money(data.get('total_amount'), data.get('currency'))} ({data['payment_status']})")
        if data.get("insurance"):
            st.write("Insurance: " + ", ".join(data["insurance"]))


def _render_insurance(data: dict[str, Any]) -> None:
    st.markdown("**Insurance purchased**")
    st.dataframe(
        [{"Plan": p["name"], "Code": p["code"], "Deductible": money(p["deductible"], p.get("currency"))}
         for p in data.get("plans", [])],
        hide_index=True,
    )


_RENDERERS = {
    "vehicle_options": _render_vehicle_options,
    "vehicle_comparison": _render_comparison,
    "price_quote": _render_quote,
    "reservation": _render_reservation,
    "insurance": _render_insurance,
}


def render_cards(cards: list[dict[str, Any]]) -> None:
    for card in cards:
        renderer = _RENDERERS.get(card.get("type"))
        if renderer:  # pending_action is rendered in the chat column with its buttons
            renderer(card["data"])


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown("**Sources**")
    for s in sources:
        st.markdown(f"- {s['title']}: {s['section']} (`{s['doc_id']}`)")


def render_tool_trace(trace: list[dict[str, Any]]) -> None:
    with st.expander(f"Tool trace ({len(trace)} call{'s' if len(trace) != 1 else ''})"):
        if trace:
            st.dataframe(
                [{"Tool": t["tool"], "Status": t["status"], "Latency (ms)": t["latency_ms"],
                  "Error": t.get("error_code") or ""} for t in trace],
                hide_index=True,
            )
        else:
            st.caption("No tools were called for this turn.")
