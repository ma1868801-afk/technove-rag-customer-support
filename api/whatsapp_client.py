"""
Meta WhatsApp Cloud API client: parse inbound webhook payloads, send
outbound text replies via the Graph API.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

import httpx

from src import config

logger = logging.getLogger("technova_rag.whatsapp")


def extract_inbound_message(payload: dict[str, Any]) -> Optional[dict[str, str]]:
    """
    Pull the first text message out of a Meta Cloud API webhook payload.

    Returns {"from": wa_id, "text": body, "message_id": wamid}, or None for
    payloads that aren't a new text message -- delivery/read status
    callbacks, non-text media, reactions, etc. are acknowledged (200 OK)
    but otherwise ignored by this bot.
    """
    try:
        entry = payload["entry"][0]
        change = entry["changes"][0]
        value = change["value"]
        messages = value.get("messages")
        if not messages:
            return None

        message = messages[0]
        if message.get("type") != "text":
            return None

        return {
            "from": message["from"],
            "text": message["text"]["body"],
            "message_id": message["id"],
        }
    except (KeyError, IndexError, TypeError) as exc:
        logger.warning("Could not parse inbound WhatsApp payload: %s", exc)
        return None


async def send_text_message(to: str, body: str) -> bool:
    """Send a plain-text WhatsApp reply via the Graph API. Returns True on success."""
    if not (config.WHATSAPP_ACCESS_TOKEN and config.WHATSAPP_PHONE_NUMBER_ID):
        logger.error("WhatsApp credentials not configured -- cannot send message.")
        return False

    url = (
        f"https://graph.facebook.com/{config.WHATSAPP_API_VERSION}/"
        f"{config.WHATSAPP_PHONE_NUMBER_ID}/messages"
    )
    headers = {"Authorization": f"Bearer {config.WHATSAPP_ACCESS_TOKEN}"}
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": body},
    }

    async with httpx.AsyncClient(timeout=15) as client:
        try:
            response = await client.post(url, headers=headers, json=payload)
            response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.error("Failed to send WhatsApp message to %s: %s", to, exc)
            return False
