"""Mock services: functional behaviour of the business and KB routes."""
from datetime import datetime, timedelta
from uuid import uuid4

import pytest

from src.utils.utils import utcnow
from tests.helpers import local

# ---------- catalog / availability ----------


def _search(client, headers, anchor, **params):
    query = {
        "location": "Vancouver Airport",
        "pickup_at": local(anchor, 5).isoformat(),
        "return_at": local(anchor, 10).isoformat(),
        "passengers": 1,
        **params,
    }
    return client.get("/v1/availability", params=query, headers=headers)


def test_locations(mock_client, auth):
    codes = [l["code"] for l in mock_client.get("/v1/locations", headers=auth("daniel")).json()["locations"]]
    assert codes == ["YVR", "YVR-DT", "YYZ"]


def test_scenario1_capacity_filter(mock_client, auth, anchor):
    resp = _search(mock_client, auth("daniel"), anchor, passengers=4, large_bags=3)
    assert resp.status_code == 200
    body = resp.json()
    assert {o["name"] for o in body["options"]} == {"Toyota RAV4", "Toyota Highlander", "Chrysler Pacifica"}
    assert body["location"]["code"] == "YVR"
    assert body["days"] == 5
    assert body["search_id"].startswith("srch_")
    assert all(o["seats"] >= 4 and o["large_bags"] >= 3 and o["available_count"] > 0 for o in body["options"])
    assert [o["daily_rate"] for o in body["options"]] == sorted(o["daily_rate"] for o in body["options"])


def test_booked_units_reduce_available_count(mock_client, auth, anchor):
    # DLRAV4 holds one YVR RAV4 from anchor+1 to anchor+6.
    overlapping = _search(mock_client, auth("daniel"), anchor, pickup_at=local(anchor, 2).isoformat(),
                          return_at=local(anchor, 4).isoformat())
    free = _search(mock_client, auth("daniel"), anchor, pickup_at=local(anchor, 7).isoformat(),
                   return_at=local(anchor, 10).isoformat())
    rav = lambda r: next(o for o in r.json()["options"] if o["model"] == "RAV4")  # noqa: E731
    assert rav(overlapping)["available_count"] == 2
    assert rav(free)["available_count"] == 3


def test_category_and_max_rate_filters(mock_client, auth, anchor):
    suv = _search(mock_client, auth("daniel"), anchor, category="MIDSIZE_SUV").json()["options"]
    assert [o["model"] for o in suv] == ["RAV4"]
    cheap = _search(mock_client, auth("daniel"), anchor, max_daily_rate="60").json()["options"]
    assert {o["model"] for o in cheap} == {"Rio", "Corolla"}


def test_location_code_and_alias_resolution(mock_client, auth, anchor):
    for query in ("YVR", "yvr", "vancouver", "Vancouver International Airport", "YVR-DT", "toronto pearson"):
        assert _search(mock_client, auth("daniel"), anchor, location=query).status_code == 200, query


def test_unknown_location_lists_supported(mock_client, auth, anchor):
    resp = _search(mock_client, auth("daniel"), anchor, location="Hogwarts Airport")
    assert resp.status_code == 404
    err = resp.json()["error"]
    assert err["code"] == "LOCATION_NOT_FOUND"
    assert [l["code"] for l in err["details"]["supported_locations"]] == ["YVR", "YVR-DT", "YYZ"]
    assert "Vancouver International Airport" in err["message"]


@pytest.mark.parametrize(
    ("pickup_days", "return_days", "fragment"),
    [(5, 4, "after the pickup"), (-2, 3, "in the past"), (1, 70, "60 days")],
)
def test_rental_window_validation(mock_client, auth, anchor, pickup_days, return_days, fragment):
    resp = _search(mock_client, auth("daniel"), anchor, pickup_at=local(anchor, pickup_days).isoformat(),
                   return_at=local(anchor, return_days).isoformat())
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "VALIDATION_ERROR"
    assert fragment in resp.json()["error"]["message"]


def test_pickup_within_grace_period_is_allowed(mock_client, auth, anchor):
    pickup = (utcnow() - timedelta(minutes=5)).replace(microsecond=0)
    resp = _search(mock_client, auth("daniel"), anchor, pickup_at=pickup.isoformat(),
                   return_at=(pickup + timedelta(days=2)).isoformat())
    assert resp.status_code == 200


def test_too_many_passengers_is_validation_error(mock_client, auth, anchor):
    resp = _search(mock_client, auth("daniel"), anchor, passengers=12)
    assert resp.status_code == 422


def test_naive_datetimes_are_local_to_the_location(mock_client, auth, anchor):
    naive = datetime.combine(anchor + timedelta(days=5), datetime.min.time()).replace(hour=10)
    body = _search(mock_client, auth("daniel"), anchor, pickup_at=naive.isoformat(),
                   return_at=(naive + timedelta(days=5)).isoformat()).json()
    assert body["pickup_at"] == local(anchor, 5).isoformat()


def test_vehicle_models_by_name_and_id(mock_client, auth):
    body = mock_client.get("/v1/vehicle-models", params={"names": "Toyota RAV4,corolla,Tesla"},
                           headers=auth("daniel")).json()
    assert [m["name"] for m in body["models"]] == ["Toyota RAV4", "Toyota Corolla"]
    assert body["not_found"] == ["Tesla"]
    assert body["models"][0]["features"]["awd"] is True

    by_id = mock_client.get("/v1/vehicle-models", params={"ids": "4,5"}, headers=auth("daniel")).json()
    assert [m["model"] for m in by_id["models"]] == ["Highlander", "Pacifica"]


@pytest.mark.parametrize("params", [{}, {"ids": "a,b"}, {"names": ",".join(["x"] * 11)}])
def test_vehicle_models_validation(mock_client, auth, params):
    assert mock_client.get("/v1/vehicle-models", params=params, headers=auth("daniel")).status_code == 422


# ---------- quotes ----------

def _quote(client, headers, anchor, **body):
    payload = {
        "model_id": 3, "location": "YVR",
        "pickup_at": local(anchor, 5).isoformat(), "return_at": local(anchor, 10).isoformat(),
        **body,
    }
    return client.post("/v1/quotes", json=payload, headers=headers)


def test_quote_without_insurance(mock_client, auth, anchor):
    q = _quote(mock_client, auth("daniel"), anchor).json()
    assert q["quote_id"].startswith("qt_")
    assert (q["days"], q["daily_rate"], q["base"], q["taxes_fees"], q["total"]) == (5, 74.0, 370.0, 44.4, 414.4)
    assert q["currency"] == "CAD"
    assert q["model"]["name"] == "Toyota RAV4"


def test_quote_with_insurance(mock_client, auth, anchor):
    q = _quote(mock_client, auth("daniel"), anchor, insurance_codes=["premium_protect"]).json()
    assert q["insurance"] == [
        {"code": "PREMIUM_PROTECT", "name": "Premium Protection Plan", "daily_price": 32.0, "amount": 160.0}
    ]
    assert q["total"] == 593.6


@pytest.mark.parametrize(
    ("body", "status", "code"),
    [
        ({"insurance_codes": ["GOLD"]}, 422, "VALIDATION_ERROR"),
        ({"model_id": 99}, 422, "VALIDATION_ERROR"),
        ({"location": "Hogwarts"}, 404, "LOCATION_NOT_FOUND"),
        ({"customer_id": "someone-else"}, 422, "VALIDATION_ERROR"),
    ],
)
def test_quote_errors(mock_client, auth, anchor, body, status, code):
    resp = _quote(mock_client, auth("daniel"), anchor, **body)
    assert resp.status_code == status
    assert resp.json()["error"]["code"] == code


# ---------- reservations ----------

def test_reservation_detail(mock_client, auth, anchor):
    r = mock_client.get("/v1/me/reservations/DLRAV4", headers=auth("daniel")).json()
    assert r["status"] == "CONFIRMED" and r["payment_status"] == "AUTHORIZED"
    assert r["vehicle"]["name"] == "Toyota RAV4"
    assert r["pickup_location"]["code"] == "YVR"
    assert r["pickup_at"] == local(anchor, 1).isoformat()
    assert r["return_at"] == local(anchor, 6).isoformat()
    assert r["total_amount"] == 593.6 and r["currency"] == "CAD" and r["version"] == 1
    assert r["insurance"] == [{"code": "PREMIUM_PROTECT", "name": "Premium Protection Plan", "deductible": 0.0,
                               "currency": "CAD", "kb_doc_id": "insurance-premium-protect"}]


def test_list_filters(mock_client, auth, anchor):
    tomorrow = (anchor + timedelta(days=1)).isoformat()
    by_day = mock_client.get("/v1/me/reservations", params={"from_date": tomorrow, "to_date": tomorrow},
                             headers=auth("daniel")).json()["reservations"]
    assert [r["confirmation_code"] for r in by_day] == ["DLRAV4"]

    picked_up = mock_client.get("/v1/me/reservations", params={"status": ["PICKED_UP"]},
                                headers=auth("daniel")).json()["reservations"]
    assert [r["confirmation_code"] for r in picked_up] == ["DLCOR1"]

    bad = mock_client.get("/v1/me/reservations", params={"from_date": tomorrow, "to_date": anchor.isoformat()},
                          headers=auth("daniel"))
    assert bad.status_code == 422


# ---------- change preview + apply (scenario 8 at the service level) ----------

def _preview(client, headers, code, new_return_at):
    return client.post(f"/v1/me/reservations/{code}/change-previews",
                       json={"new_return_at": new_return_at.isoformat()}, headers=headers)


def _apply(client, headers, code, preview, key):
    return client.post(f"/v1/me/reservations/{code}/changes",
                       json={"preview_id": preview["preview_id"], "expected_version": preview["expected_version"]},
                       headers={**headers, "Idempotency-Key": key})


def test_preview_prices_extension_without_writing(mock_client, auth, anchor):
    resp = _preview(mock_client, auth("daniel"), "DLRAV4", local(anchor, 12))
    assert resp.status_code == 200
    p = resp.json()
    assert p["preview_id"].startswith("pv_")
    assert (p["old_total"], p["new_total"], p["price_delta"]) == (593.6, 1305.92, 712.32)
    assert p["expected_version"] == 1 and p["vehicle_changed"] is False
    assert p["new_return_at"] == local(anchor, 12).isoformat()

    r = mock_client.get("/v1/me/reservations/DLRAV4", headers=auth("daniel")).json()
    assert r["version"] == 1 and r["return_at"] == local(anchor, 6).isoformat()


def test_apply_change_and_idempotent_replay(mock_client, auth, anchor):
    p = _preview(mock_client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    key = f"idem-{uuid4()}"
    first = _apply(mock_client, auth("daniel"), "DLRAV4", p, key)
    assert first.status_code == 200
    assert first.json()["status"] == "APPLIED"
    res = first.json()["reservation"]
    assert res["version"] == 2
    assert res["return_at"] == local(anchor, 12).isoformat()
    assert res["total_amount"] == 1305.92

    replay = _apply(mock_client, auth("daniel"), "DLRAV4", p, key)
    assert replay.status_code == 200
    assert replay.json() == first.json()

    # The preview is consumed; a new key can't reuse it.
    again = _apply(mock_client, auth("daniel"), "DLRAV4", p, f"idem-{uuid4()}")
    assert again.status_code == 410
    assert mock_client.get("/v1/me/reservations/DLRAV4", headers=auth("daniel")).json()["version"] == 2


def test_stale_preview_is_version_conflict(mock_client, auth, anchor):
    p1 = _preview(mock_client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    p2 = _preview(mock_client, auth("daniel"), "DLRAV4", local(anchor, 9)).json()
    assert _apply(mock_client, auth("daniel"), "DLRAV4", p1, f"idem-{uuid4()}").status_code == 200
    stale = _apply(mock_client, auth("daniel"), "DLRAV4", p2, f"idem-{uuid4()}")
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "VERSION_CONFLICT"


def test_missing_idempotency_key(mock_client, auth, anchor):
    p = _preview(mock_client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    resp = mock_client.post("/v1/me/reservations/DLRAV4/changes",
                            json={"preview_id": p["preview_id"], "expected_version": 1}, headers=auth("daniel"))
    assert resp.status_code == 422


@pytest.mark.parametrize(
    ("days", "fragment"),
    [(0, "after the pickup"), (6, "already the return time"), (80, "60 days")],
)
def test_preview_validation(mock_client, auth, anchor, days, fragment):
    when = local(anchor, days, hour=9) if days == 0 else local(anchor, days)
    resp = _preview(mock_client, auth("daniel"), "DLRAV4", when)
    assert resp.status_code == 422
    assert fragment in resp.json()["error"]["message"]


def test_preview_in_the_past_rejected(mock_client, auth, anchor):
    # DLCOR1 was picked up 2 days ago; returning "yesterday" is after pickup but in the past.
    resp = _preview(mock_client, auth("daniel"), "DLCOR1", local(anchor, -1))
    assert resp.status_code == 422
    assert "future" in resp.json()["error"]["message"]


# ---------- in-memory only: scenarios that need direct state manipulation ----------

def _block(repo, vehicle_id, start, end):
    customer = next(iter(repo.customers))
    rid = uuid4()
    repo.reservations[rid] = {
        "id": rid, "confirmation_code": f"BLK{len(repo.reservations):03d}", "customer_id": customer,
        "vehicle_id": vehicle_id, "pickup_location_id": 1, "return_location_id": 1, "pickup_at": start,
        "return_at": end, "status": "CONFIRMED", "payment_status": "AUTHORIZED", "total_amount": 0,
        "currency": "CAD", "version": 1,
    }


def test_extension_switches_unit_when_same_vehicle_is_taken(memory_client, auth, anchor):
    client, repo, _ = memory_client
    rav = next(r for r in repo.reservations.values() if r["confirmation_code"] == "DLRAV4")
    _block(repo, rav["vehicle_id"], local(anchor, 7), local(anchor, 9))
    p = _preview(client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    assert p["vehicle_changed"] is True
    applied = _apply(client, auth("daniel"), "DLRAV4", p, f"idem-{uuid4()}").json()
    assert applied["reservation"]["vehicle"]["plate"] != "YVR-RAV1"
    assert len(repo.changes) == 1


def test_extension_not_available_when_all_units_taken(memory_client, auth, anchor):
    client, repo, _ = memory_client
    for v in repo.vehicles.values():
        if v["model_id"] == 3 and v["home_location_id"] == 1:
            _block(repo, v["id"], local(anchor, 7), local(anchor, 9))
    resp = _preview(client, auth("daniel"), "DLRAV4", local(anchor, 12))
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "NOT_AVAILABLE"


def test_conflict_appearing_after_preview_is_rejected(memory_client, auth, anchor):
    client, repo, _ = memory_client
    p = _preview(client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    rav = next(r for r in repo.reservations.values() if r["confirmation_code"] == "DLRAV4")
    _block(repo, rav["vehicle_id"], local(anchor, 7), local(anchor, 9))
    resp = _apply(client, auth("daniel"), "DLRAV4", p, f"idem-{uuid4()}")
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "NOT_AVAILABLE"
    assert repo.changes == []


@pytest.mark.parametrize("status", ["RETURNED", "CANCELLED"])
def test_closed_reservation_is_invalid_state(memory_client, auth, anchor, status):
    client, repo, _ = memory_client
    next(r for r in repo.reservations.values() if r["confirmation_code"] == "DLRAV4")["status"] = status
    resp = _preview(client, auth("daniel"), "DLRAV4", local(anchor, 12))
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "INVALID_STATE"


def test_expired_preview(memory_client, auth, anchor):
    client, repo, app = memory_client
    p = _preview(client, auth("daniel"), "DLRAV4", local(anchor, 12)).json()
    app.state.previews._items[p["preview_id"]].expires_at = utcnow() - timedelta(seconds=1)
    resp = _apply(client, auth("daniel"), "DLRAV4", p, f"idem-{uuid4()}")
    assert resp.status_code == 410
    assert resp.json()["error"]["code"] == "PREVIEW_EXPIRED"


# ---------- KB search ----------

def _kb(client, headers, **body):
    return client.post("/v1/kb/search", json=body, headers=headers)


def test_kb_plan_filter_includes_general_docs_only(mock_client, auth):
    results = _kb(mock_client, auth("daniel"), query="collision damage deductible", top_k=20,
                  filters={"plan_code": "BASIC_CDW"}).json()["results"]
    doc_ids = {r["doc_id"] for r in results}
    assert "insurance-basic-cdw" in doc_ids
    assert "insurance-premium-protect" not in doc_ids
    assert all(set(r) == {"doc_id", "title", "section", "content", "score"} for r in results)
    assert [r["score"] for r in results] == sorted((r["score"] for r in results), reverse=True)


def test_kb_category_filter(mock_client, auth):
    results = _kb(mock_client, auth("daniel"), query="what to do after an accident",
                  filters={"categories": ["ACCIDENT"]}).json()["results"]
    assert results and {r["doc_id"] for r in results} == {"accident-procedure"}


@pytest.mark.parametrize(
    "body",
    [{"query": "ab"}, {"query": "x" * 301}, {"query": "valid", "top_k": 0},
     {"query": "valid", "filters": {"categories": ["MOVIES"]}}],
)
def test_kb_validation(mock_client, auth, body):
    assert _kb(mock_client, auth("daniel"), **body).status_code == 422
