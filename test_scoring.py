"""
SUITE COMPLETA DE TESTS DE PRODUCCIÓN — Scoring Service v4

Cómo correr:
    pip install pytest httpx
    pytest test_scoring_service.py -v

O sin pytest, directamente:
    python test_scoring_service.py

La API debe estar corriendo en localhost:8001.
Ajustar BASE_URL si está en otro host/puerto.
"""

import requests
import json
import sys

BASE_URL = "http://localhost:8001"

PASS = 0
FAIL = 0

def check(nombre, condicion, detalle=""):
    global PASS, FAIL
    if condicion:
        print(f"  ✓ {nombre}")
        PASS += 1
    else:
        print(f"  ✗ {nombre}  ← {detalle}")
        FAIL += 1

def post(payload):
    r = requests.post(f"{BASE_URL}/predict_scoring", json=payload)
    return r.status_code, r.json() if r.status_code in (200, 422, 500) else {}

# ─── Lead base (referencia neutral) ──────────────────────────────────────────
BASE = {
    "avg_p_compra": 0.50, "max_p_compra": 0.65, "trend_p_compra": 0.10,
    "intent_entropy": 0.50, "n_mensajes": 5, "avg_response_time": 30.0,
    "conversation_days": 2, "n_channels": 1,
    "channel": "whatsapp", "ticket_estimate": "medium"
}

def lead(**overrides):
    l = BASE.copy()
    l.update(overrides)
    return l


print("=" * 60)
print("TEST 1 — Health check")
print("=" * 60)
r = requests.get(f"{BASE_URL}/health")
check("Endpoint /health responde 200", r.status_code == 200)
data = r.json()
check("modelo_gb disponible", data.get("gb_disponible") is True)
check("scoring_method es analytic_formula", data.get("scoring_method") == "analytic_formula")


print()
print("=" * 60)
print("TEST 2 — Estructura de la respuesta")
print("=" * 60)
status, resp = post(BASE)
check("HTTP 200", status == 200)
check("Tiene 'percentile'",  "percentile"  in resp)
check("Tiene 'priority'",    "priority"    in resp)
check("Tiene 'score'",       "score"       in resp)
check("Tiene 'prob_gb'",     "prob_gb"     in resp)
check("Tiene 'status'",      "status"      in resp)
check("status == 'success'", resp.get("status") == "success")
check("percentile en [0, 100]",
      0 <= resp.get("percentile", -1) <= 100,
      f"percentile={resp.get('percentile')}")
check("priority es valor válido",
      resp.get("priority") in {"alta", "media", "baja", "muy_baja"})


print()
print("=" * 60)
print("TEST 3 — Monotonicidad de avg_p_compra (la más importante)")
print("=" * 60)
# Subir avg_p_compra debe subir el percentil siempre
valores_avg = [0.05, 0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95]
percentiles_avg = []
for v in valores_avg:
    _, r = post(lead(avg_p_compra=v, max_p_compra=max(v, 0.65)))
    percentiles_avg.append(r.get("percentile", -1))

for i in range(len(valores_avg) - 1):
    check(
        f"avg_p={valores_avg[i]:.2f} → avg_p={valores_avg[i+1]:.2f}: percentil sube",
        percentiles_avg[i] <= percentiles_avg[i+1] + 0.5,  # tolerancia 0.5
        f"{percentiles_avg[i]:.1f} > {percentiles_avg[i+1]:.1f}"
    )


print()
print("=" * 60)
print("TEST 4 — Monotonicidad de avg_response_time (dirección inversa)")
print("=" * 60)
valores_resp = [2, 10, 25, 50, 90, 140, 180]
percentiles_resp = []
for v in valores_resp:
    _, r = post(lead(avg_response_time=float(v)))
    percentiles_resp.append(r.get("percentile", -1))

for i in range(len(valores_resp) - 1):
    check(
        f"resp={valores_resp[i]}min → resp={valores_resp[i+1]}min: percentil baja",
        percentiles_resp[i] >= percentiles_resp[i+1] - 0.5,
        f"{percentiles_resp[i]:.1f} < {percentiles_resp[i+1]:.1f}"
    )


print()
print("=" * 60)
print("TEST 5 — Monotonicidad de trend_p_compra")
print("=" * 60)
valores_trend = [-1.0, -0.5, -0.2, 0.0, 0.2, 0.5, 1.0]
percentiles_trend = []
for v in valores_trend:
    _, r = post(lead(trend_p_compra=v))
    percentiles_trend.append(r.get("percentile", -1))

for i in range(len(valores_trend) - 1):
    check(
        f"trend={valores_trend[i]:.1f} → trend={valores_trend[i+1]:.1f}: percentil sube",
        percentiles_trend[i] <= percentiles_trend[i+1] + 0.5,
        f"{percentiles_trend[i]:.1f} > {percentiles_trend[i+1]:.1f}"
    )


print()
print("=" * 60)
print("TEST 6 — Monotonicidad de intent_entropy (dirección inversa)")
print("=" * 60)
valores_ent = [0.05, 0.2, 0.4, 0.6, 0.8, 0.95]
percentiles_ent = []
for v in valores_ent:
    _, r = post(lead(intent_entropy=v))
    percentiles_ent.append(r.get("percentile", -1))

for i in range(len(valores_ent) - 1):
    check(
        f"entropy={valores_ent[i]:.2f} → entropy={valores_ent[i+1]:.2f}: percentil baja",
        percentiles_ent[i] >= percentiles_ent[i+1] - 0.5,
        f"{percentiles_ent[i]:.1f} < {percentiles_ent[i+1]:.1f}"
    )


print()
print("=" * 60)
print("TEST 7 — Jerarquía de canales")
print("=" * 60)
# Orden esperado por efecto: whatsapp > web > instagram > facebook = twitter > email
orden_canales = ["whatsapp", "web", "instagram", "facebook", "twitter", "email"]
pcts_canales = {}
for c in orden_canales:
    _, r = post(lead(channel=c))
    pcts_canales[c] = r.get("percentile", -1)

for canal, pct in sorted(pcts_canales.items(), key=lambda x: x[1], reverse=True):
    print(f"  {canal:<12} percentil={pct:.1f}")

check("whatsapp tiene percentil más alto", 
      pcts_canales["whatsapp"] == max(pcts_canales.values()),
      f"whatsapp={pcts_canales['whatsapp']:.1f}")
check("email tiene percentil más bajo",
      pcts_canales["email"] == min(pcts_canales.values()),
      f"email={pcts_canales['email']:.1f}")


print()
print("=" * 60)
print("TEST 8 — Jerarquía de ticket")
print("=" * 60)
pcts_ticket = {}
for t in ["low", "medium", "high"]:
    _, r = post(lead(ticket_estimate=t))
    pcts_ticket[t] = r.get("percentile", -1)

print(f"  low={pcts_ticket['low']:.1f}  medium={pcts_ticket['medium']:.1f}  high={pcts_ticket['high']:.1f}")
check("low > medium > high",
      pcts_ticket["low"] >= pcts_ticket["medium"] >= pcts_ticket["high"],
      f"low={pcts_ticket['low']:.1f} medium={pcts_ticket['medium']:.1f} high={pcts_ticket['high']:.1f}")


print()
print("=" * 60)
print("TEST 9 — Casos extremos (stress test)")
print("=" * 60)
# Lead ideal: todo en el mejor valor posible
_, r_ideal = post(lead(
    avg_p_compra=0.98, max_p_compra=0.99, trend_p_compra=0.95,
    intent_entropy=0.02, n_mensajes=15, avg_response_time=1.0,
    conversation_days=1, n_channels=3,
    channel="whatsapp", ticket_estimate="low"
))
check("Lead ideal tiene percentil > 95",
      r_ideal.get("percentile", 0) > 95,
      f"percentile={r_ideal.get('percentile')}")
check("Lead ideal tiene priority='alta'",
      r_ideal.get("priority") == "alta")

# Lead frío: todo en el peor valor posible
_, r_frio = post(lead(
    avg_p_compra=0.02, max_p_compra=0.05, trend_p_compra=-0.95,
    intent_entropy=0.98, n_mensajes=1, avg_response_time=180.0,
    conversation_days=10, n_channels=1,
    channel="email", ticket_estimate="high"
))
check("Lead frío tiene percentil < 10",
      r_frio.get("percentile", 100) < 10,
      f"percentile={r_frio.get('percentile')}")
check("Lead frío tiene priority en {'baja', 'muy_baja'}",
      r_frio.get("priority") in {"baja", "muy_baja"})

# Lead tóxico: intención alta pero comportamiento frío (el caso clásico)
_, r_toxico = post(lead(
    avg_p_compra=0.90, max_p_compra=0.95, trend_p_compra=-0.40,
    intent_entropy=0.10, n_mensajes=2, avg_response_time=180.0,
    conversation_days=1, n_channels=1,
    channel="web", ticket_estimate="high"
))
print(f"  Lead tóxico (intención alta, comportamiento frío): percentil={r_toxico.get('percentile'):.1f}")
check("Lead tóxico está por debajo del lead ideal",
      r_toxico.get("percentile", 100) < r_ideal.get("percentile", 0),
      f"toxico={r_toxico.get('percentile'):.1f} ideal={r_ideal.get('percentile'):.1f}")


print()
print("=" * 60)
print("TEST 10 — Validación de inputs inválidos (422)")
print("=" * 60)
casos_invalidos = [
    ("avg_p_compra fuera de rango",    lead(avg_p_compra=1.5)),
    ("trend_p_compra fuera de rango",  lead(trend_p_compra=2.0)),
    ("max < avg",                       lead(avg_p_compra=0.8, max_p_compra=0.5)),
    ("channel inválido",               lead(channel="telegram")),
    ("ticket inválido",                lead(ticket_estimate="ultra")),
    ("n_mensajes = 0",                 lead(n_mensajes=0)),
    ("avg_response_time negativo",     lead(avg_response_time=-1.0)),
]
for nombre, payload in casos_invalidos:
    status, _ = post(payload)
    check(f"'{nombre}' devuelve 422",
          status == 422,
          f"status={status}")


print()
print("=" * 60)
print("TEST 11 — Robustez con valores límite válidos")
print("=" * 60)
limites = [
    ("avg_p_compra=0.0",    lead(avg_p_compra=0.0, max_p_compra=0.0)),
    ("avg_p_compra=1.0",    lead(avg_p_compra=1.0, max_p_compra=1.0)),
    ("trend=-1.0",          lead(trend_p_compra=-1.0)),
    ("trend=1.0",           lead(trend_p_compra=1.0)),
    ("entropy=0.0",         lead(intent_entropy=0.0)),
    ("entropy=1.0",         lead(intent_entropy=1.0)),
    ("resp=180 (max)",      lead(avg_response_time=180.0)),
    ("resp=200 (se clipea)", lead(avg_response_time=200.0)),  # debe clipearse a 180
    ("n_mensajes=1",        lead(n_mensajes=1)),
    ("n_channels=3",        lead(n_channels=3)),
    ("channel=email (ref)", lead(channel="email")),
    ("ticket=high (ref)",   lead(ticket_estimate="high")),
]
for nombre, payload in limites:
    status, r = post(payload)
    check(f"'{nombre}' no da error (200)",
          status == 200,
          f"status={status}, detail={r.get('detail','')}")

# Verificar que resp=200 y resp=180 dan el mismo resultado (clipping)
_, r180 = post(lead(avg_response_time=180.0))
_, r200 = post(lead(avg_response_time=200.0))
check("resp=200 clipeada a 180 (mismo percentil)",
      r180.get("percentile") == r200.get("percentile"),
      f"p(180)={r180.get('percentile')} p(200)={r200.get('percentile')}")


print()
print("=" * 60)
print("TEST 12 — Estabilidad (mismo input → mismo output)")
print("=" * 60)
results = []
for _ in range(5):
    _, r = post(BASE)
    results.append(r.get("percentile"))

check("5 llamadas iguales dan el mismo percentil",
      len(set(results)) == 1,
      f"valores={results}")


print()
print("=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
total = PASS + FAIL
print(f"  Pasaron: {PASS}/{total}")
print(f"  Fallaron: {FAIL}/{total}")
if FAIL == 0:
    print("  ✓ LISTO PARA PRODUCCIÓN")
else:
    print("  ✗ HAY TESTS FALLANDO — revisar antes de desplegar")

sys.exit(0 if FAIL == 0 else 1)