from typing import Optional

import anthropic
import pandas as pd

from src.features.product_enrichment import create_product_context

MODEL = "claude-haiku-4-5"

SYSTEM_PROMPT = (
    "You are a retail marketing copywriter. Write short, on-brand ad copy "
    "for the given product, grounded only in the provided product context "
    "and similar catalog items. Do not invent product details (materials, "
    "price, or features) that aren't in the context. Keep it to 2-3 sentences. "
    "Respond with plain prose only — no Markdown formatting, headers, or "
    "bullet points, since this is rendered as plain text in the UI."
)


def build_grounding_context(
    product_row: pd.Series,
    similar_products: pd.DataFrame,
) -> str:
    lines = ["TARGET PRODUCT:", create_product_context(product_row), ""]

    if not similar_products.empty:
        lines.append("SIMILAR CATALOG PRODUCTS (for style/tone reference):")
        for _, row in similar_products.iterrows():
            if row["article_id"] == product_row.get("article_id"):
                continue
            lines.append(f"- {row.get('product_name', '')} ({row.get('product_type', '')})")

    return "\n".join(lines)


def generate_grounded_copy(
    product_row: pd.Series,
    similar_products: pd.DataFrame,
    customer_segment: str,
    promotion_strategy: str,
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    client = client or anthropic.Anthropic()

    context = build_grounding_context(product_row, similar_products)

    user_message = (
        f"{context}\n\n"
        f"Target customer segment: {customer_segment}\n"
        f"Promotion strategy: {promotion_strategy}\n\n"
        "Write the ad copy."
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=300,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return next(
        (block.text for block in response.content if block.type == "text"),
        "",
    )
