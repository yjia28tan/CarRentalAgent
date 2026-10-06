"""Trip context: updated deterministically by code from successful tool results (never by the LLM)."""
from typing import Any

MAX_DISCUSSED_MODELS = 10


def empty_context() -> dict[str, Any]:
    return {
        "location": None,
        "pickup_at": None,
        "return_at": None,
        "party": {"passengers": None, "adults": None, "children": None},
        "luggage": {"large": None},
        "last_search_id": None,
        "candidates": [],
        "discussed_models": [],
        "active_reservation_code": None,
        "last_quote_id": None,
    }


def apply_tool_result(trip: dict[str, Any], tool: str, args: dict[str, Any], result: dict[str, Any]) -> None:
    """`args` are the validated tool arguments; `result` is the successful tool payload."""
    if tool == "search_vehicles":
        trip["location"] = result["location"]["code"]
        trip["pickup_at"] = result["pickup_at"]
        trip["return_at"] = result["return_at"]
        trip["party"]["passengers"] = args["passengers"]
        trip["luggage"]["large"] = args.get("large_bags", 0)
        trip["last_search_id"] = result["search_id"]
        trip["candidates"] = [{"model_id": o["model_id"], "name": o["name"]} for o in result["options"]]
    elif tool == "get_vehicle_details":
        known = {m["model_id"] for m in trip["discussed_models"]}
        for m in result["models"]:
            if m["model_id"] not in known:
                trip["discussed_models"].append({"model_id": m["model_id"], "name": m["name"]})
                known.add(m["model_id"])
        trip["discussed_models"] = trip["discussed_models"][-MAX_DISCUSSED_MODELS:]
    elif tool == "get_my_reservation":
        trip["active_reservation_code"] = result["confirmation_code"]
    elif tool == "list_my_reservations" and len(result["reservations"]) == 1:
        # An unambiguous single match is as good as an explicit lookup for follow-ups.
        trip["active_reservation_code"] = result["reservations"][0]["confirmation_code"]
    elif tool == "get_price_quote":
        trip["last_quote_id"] = result["quote_id"]
