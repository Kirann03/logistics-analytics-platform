from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from urllib import request as urllib_request

import pandas as pd


def trigger_sla_alerts(routes: pd.DataFrame, threshold: float) -> dict[str, object]:
    flagged = routes[routes["delay_rate"] >= threshold].copy()
    if flagged.empty:
        return {"sent": False, "count": 0, "message": "No routes crossed the SLA alert threshold."}

    payload = {
        "threshold": threshold,
        "routes": flagged[["route_label", "delay_rate", "avg_lead_time", "shipments"]].to_dict(orient="records"),
    }

    webhook_url = os.getenv("ALERT_WEBHOOK_URL")
    if webhook_url:
        try:
            req = urllib_request.Request(
                webhook_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            urllib_request.urlopen(req, timeout=10)
        except Exception as exc:
            return {"sent": False, "count": len(flagged), "message": f"Webhook alert failed: {exc}"}

    smtp_host = os.getenv("SMTP_HOST")
    smtp_to = os.getenv("SMTP_TO")
    smtp_user = os.getenv("SMTP_USER")
    smtp_password = os.getenv("SMTP_PASSWORD")
    if smtp_host and smtp_to and smtp_user and smtp_password:
        try:
            message = EmailMessage()
            message["Subject"] = "Logistics SLA Breach Alert"
            message["From"] = smtp_user
            message["To"] = smtp_to
            message.set_content(json.dumps(payload, indent=2))
            with smtplib.SMTP(smtp_host, int(os.getenv("SMTP_PORT", "587"))) as server:
                server.starttls()
                server.login(smtp_user, smtp_password)
                server.send_message(message)
        except Exception as exc:
            return {"sent": False, "count": len(flagged), "message": f"Email alert failed: {exc}"}

    return {"sent": bool(webhook_url or smtp_host), "count": len(flagged), "message": "Alert payload prepared and delivered to configured channels when available."}
