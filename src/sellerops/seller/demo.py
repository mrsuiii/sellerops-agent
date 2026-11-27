"""Starting content for the public demo: a couple of agent proposals waiting for the seller.

A fresh demo database has no proposals, so a visitor would land on an empty
queue. ``seed_demo`` adds two pending proposals written as the chat agent
would (``actor="agent"``), only when the store has no proposals and no
caption requests yet, so it never touches real work. Called at startup when
``SELLEROPS_DEMO_MODE=1``.
"""

from __future__ import annotations

from .service import SellerService

DEMO_PROPOSALS = (
    {
        "product_id": "FS180",
        "reason": "Deskripsi sekarang generik; tonjolkan motif dan momen pemakaian, tambahkan tag gaya.",
        "description": (
            "Dress Pea brings a playful pea print to everyday outfits. Easy to wear for brunch, park walks "
            "or a relaxed day in the city, with a 2 year warranty and 90 days return."),
        "tags": ["clothing", "dresses", "casual", "printed"],
    },
    {
        "product_id": "FS083",
        "reason": "Nama lebih jelas untuk pencarian: sebut motif kotak dan target pembeli.",
        "name": "Men's Blue & Black Check Shirt",
    },
)


def seed_demo(service: SellerService) -> int:
    """Add the demo proposals to an untouched store; returns how many were added."""
    if service.list_proposals(limit=1) or service.list_requests(limit=1):
        return 0
    for item in DEMO_PROPOSALS:
        fields = {key: item[key] for key in ("name", "description", "tags") if key in item}
        service.propose_caption(item["product_id"], item["reason"], actor="agent", **fields)
    return len(DEMO_PROPOSALS)
