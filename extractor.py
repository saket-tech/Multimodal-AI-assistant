"""
Structured field extraction from financial documents.
Returns vendor, line items, amounts, dates, taxes, totals as a dict.
"""

import json
import re

from groq import Groq


def extract_fields(client: Groq, model: str, document_text: str) -> dict:
    """
    Extract structured fields from financial document text.
    Returns dict with keys: vendor, date, line_items, subtotal, tax, total, payment_received.
    """
    prompt = """Extract the following fields from this financial document and return ONLY valid JSON.
If a field is not found, use null.

Fields to extract:
- vendor: merchant or company name
- statement_date: date of the statement
- line_items: list of {description, amount} objects
- subtotal: subtotal before tax
- tax: tax and fees amount
- total: total amount due
- payment_received: any payments received
- previous_balance: balance from prior period

Document:
""" + document_text + """

Return ONLY a JSON object, no explanation."""

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=512,
    )

    raw = response.choices[0].message.content.strip()

    # Extract JSON from response even if wrapped in markdown
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return {"raw_extraction": raw}
