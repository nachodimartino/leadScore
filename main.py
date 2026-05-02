from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
import pandas as pd
import numpy as np
import joblib
from scipy.stats import percentileofscore
 
app = FastAPI(title="Lead Scoring Service — v4 (score analítico)")
 
# ─── Carga ────────────────────────────────────────────────────────────────────
# El GB se mantiene para reportar AUC en la tesis.
# El score analítico produce el percentil/ranking del lead en producción.
modelo_gb = None
train_scores_analiticos = None
 
try:
    modelo_gb = joblib.load("models/modelo_scoring_gb_v3.pkl")
    # Distribución de scores ANALÍTICOS del dataset de entrenamiento
    # (generada en el notebook — ver celda de exportación actualizada)
    train_scores_analiticos = joblib.load("models/train_scores_analiticos_v3.pkl")
    print("✅ Modelos cargados")
except Exception as e:
    print(f"❌ Error cargando modelos: {e}")
 
# ─── Constantes ───────────────────────────────────────────────────────────────
CANALES_VALIDOS   = {"email", "facebook", "instagram", "twitter", "web", "whatsapp"}
TICKETS_VALIDOS   = {"high", "low", "medium"}
CANAL_REFERENCIA  = "email"
TICKET_REFERENCIA = "high"
 
# Efectos contextuales — mismos pesos que el simulador v3
CHANNEL_EFFECT = {
    "whatsapp":  0.06, "web":       0.03, "instagram": 0.01,
    "email":    -0.02, "twitter":  -0.02, "facebook": -0.02
}
TICKET_EFFECT = {"low": 0.08, "medium": 0.0, "high": -0.08}
 
 
# ─── Fórmula analítica de score ───────────────────────────────────────────────
def calcular_score_analitico(
    avg_p: float, max_p: float, trend: float, entropy: float,
    n_msg: int, resp: float, days: int, n_ch: int,
    channel: str, ticket: str
) -> float:
    """
    Score determinístico que replica exactamente la función generadora del simulador.
    Garantiza comportamiento monótonico: más intención → mayor score → mayor percentil.
 
    No tiene el problema de saturación de regiones que tiene el Gradient Boosting
    con max_depth=2, donde avg_p=0.42, 0.52 y 0.72 caen en la misma hoja del árbol
    y reciben el mismo score.
    """
    resp_norm = min(resp, 180.0) / 180.0
    msg_norm  = np.log1p(n_msg) / np.log1p(10)
 
    s = (
        2.5 * avg_p +
        0.6 * max_p +
        0.6 * trend +
        0.3 * msg_norm -
        0.7 * resp_norm -
        0.8 * entropy +
        0.8 * avg_p * (1 - resp_norm)   # interacción: intención × velocidad
    )
    s = float(np.tanh(s))
 
    s += CHANNEL_EFFECT.get(channel, 0.0)
    s += TICKET_EFFECT.get(ticket, 0.0)
    s += 0.03 * (n_ch - 1)
 
    return s
 
 
# ─── Schema ───────────────────────────────────────────────────────────────────
class LeadFeatures(BaseModel):
    avg_p_compra:      float
    max_p_compra:      float
    trend_p_compra:    float
    intent_entropy:    float
    n_mensajes:        int
    avg_response_time: float
    conversation_days: int
    n_channels:        int
    channel:           str
    ticket_estimate:   str
 
    @field_validator("avg_p_compra", "max_p_compra", "intent_entropy")
    @classmethod
    def rango_0_1(cls, v, info):
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"{info.field_name} debe estar en [0, 1]. Recibido: {v}")
        return round(float(v), 6)
 
    @field_validator("trend_p_compra")
    @classmethod
    def rango_trend(cls, v):
        if not (-1.0 <= v <= 1.0):
            raise ValueError(f"trend_p_compra debe estar en [-1, 1]. Recibido: {v}")
        return round(float(v), 6)
 
    @field_validator("channel")
    @classmethod
    def canal_valido(cls, v):
        v = v.lower().strip()
        if v not in CANALES_VALIDOS:
            raise ValueError(f"channel '{v}' inválido. Válidos: {sorted(CANALES_VALIDOS)}")
        return v
 
    @field_validator("ticket_estimate")
    @classmethod
    def ticket_valido(cls, v):
        v = v.lower().strip()
        if v not in TICKETS_VALIDOS:
            raise ValueError(f"ticket_estimate '{v}' inválido. Válidos: {sorted(TICKETS_VALIDOS)}")
        return v
 
    @field_validator("avg_response_time")
    @classmethod
    def clip_response(cls, v):
        if v < 0:
            raise ValueError("avg_response_time debe ser >= 0")
        return float(min(v, 180.0))
 
    @field_validator("n_mensajes", "conversation_days", "n_channels")
    @classmethod
    def positivo(cls, v, info):
        if v < 1:
            raise ValueError(f"{info.field_name} debe ser >= 1")
        return v
 
    @model_validator(mode="after")
    def max_mayor_avg(self):
        if self.max_p_compra < self.avg_p_compra:
            raise ValueError(
                f"max_p_compra ({self.max_p_compra}) no puede ser menor "
                f"que avg_p_compra ({self.avg_p_compra})"
            )
        return self
 
    model_config = {"json_schema_extra": {"examples": [{
        "avg_p_compra": 0.72, "max_p_compra": 0.90, "trend_p_compra": 0.45,
        "intent_entropy": 0.28, "n_mensajes": 8, "avg_response_time": 12.5,
        "conversation_days": 2, "n_channels": 2,
        "channel": "whatsapp", "ticket_estimate": "low"
    }]}}
 
 
# ─── Endpoint principal ───────────────────────────────────────────────────────
@app.post("/predict_scoring")
def predict_scoring(data: LeadFeatures):
    if train_scores_analiticos is None:
        raise HTTPException(503, "Distribución de scores no disponible")
 
    # ── Score analítico → percentil de ranking ────────────────────────────────
    score = calcular_score_analitico(
        avg_p=data.avg_p_compra, max_p=data.max_p_compra,
        trend=data.trend_p_compra, entropy=data.intent_entropy,
        n_msg=data.n_mensajes, resp=data.avg_response_time,
        days=data.conversation_days, n_ch=data.n_channels,
        channel=data.channel, ticket=data.ticket_estimate
    )
 
    # Percentil relativo a la distribución del dataset simulado
    # Indica qué % de leads tienen score menor → ranking del lead
    percentil = float(percentileofscore(train_scores_analiticos, score, kind="rank"))
 
    # ── Probabilidad de conversión (GB) — para métricas/tesis ────────────────
    # El GB predice p(converted=1). Se reporta como información adicional
    # pero NO se usa para el ranking (tiene el problema de saturación de regiones).
    prob_gb = None
    if modelo_gb is not None:
        try:
            row = {col: 0.0 for col in modelo_gb.feature_names_in_}
            for campo in ["avg_p_compra","max_p_compra","trend_p_compra",
                          "intent_entropy","n_mensajes","avg_response_time",
                          "conversation_days","n_channels"]:
                if campo in row:
                    row[campo] = getattr(data, campo)
            canal_col  = f"channel_{data.channel}"
            ticket_col = f"ticket_estimate_{data.ticket_estimate}"
            if canal_col  in row: row[canal_col]  = 1.0
            if ticket_col in row: row[ticket_col] = 1.0
            X = pd.DataFrame([row])[list(modelo_gb.feature_names_in_)]
            prob_gb = round(float(modelo_gb.predict_proba(X)[0, 1]), 4)
        except Exception:
            prob_gb = None
 
    return {
        # Estos dos campos son los que usa el CRM
        "percentile":  round(percentil, 2),
        "priority":    _priority(percentil),
        # Score crudo (para debug y análisis)
        "score":       round(score, 4),
        # Probabilidad del GB (para tesis/métricas, no para ranking)
        "probability":     prob_gb,
        "status":      "success",
        "scoring_method": "analytic_formula"
    }
 
 
def _priority(p: float) -> str:
    if p >= 80: return "alta"
    if p >= 50: return "media"
    if p >= 20: return "baja"
    return "muy_baja"
 
 
@app.get("/health")
def health():
    return {
        "status": "ok" if train_scores_analiticos is not None else "degraded",
        "scoring_method": "analytic_formula",
        "gb_disponible": modelo_gb is not None,
        "canales_validos": sorted(CANALES_VALIDOS),
        "tickets_validos": sorted(TICKETS_VALIDOS),
    }
 