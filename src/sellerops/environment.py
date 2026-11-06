"""Small deterministic SellerOps environment used as dataset source of truth."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .schemas import validate_action


CATALOG: tuple[dict[str, Any], ...] = (
    {"product_id": "P1001", "name": "Orbit Wireless Mouse", "category": "accessories", "brand": "Orbit", "color": "black", "price_idr": 249_000, "tags": ["wireless", "mouse", "ergonomic"]},
    {"product_id": "P1002", "name": "Kepler Mechanical Keyboard", "category": "accessories", "brand": "Kepler", "color": "black", "price_idr": 799_000, "tags": ["mechanical", "keyboard", "rgb"]},
    {"product_id": "P1003", "name": "Atlas Laptop Stand", "category": "accessories", "brand": "Atlas", "color": "silver", "price_idr": 349_000, "tags": ["laptop", "stand", "ergonomic"]},
    {"product_id": "P1004", "name": "Nova USB-C Hub", "category": "accessories", "brand": "Nova", "color": "gray", "price_idr": 459_000, "tags": ["usb-c", "hub", "adapter"]},
    {"product_id": "P1005", "name": "Echo Wireless Earbuds", "category": "audio", "brand": "Echo", "color": "white", "price_idr": 699_000, "tags": ["wireless", "earbuds", "bluetooth"]},
    {"product_id": "P1006", "name": "Luma Desk Lamp", "category": "home-office", "brand": "Luma", "color": "white", "price_idr": 299_000, "tags": ["lamp", "led", "desk"]},
    {"product_id": "P1007", "name": "Terra Water Bottle", "category": "lifestyle", "brand": "Terra", "color": "green", "price_idr": 179_000, "tags": ["bottle", "water", "reusable"]},
    {"product_id": "P1008", "name": "Comet Everyday Backpack", "category": "bags", "brand": "Comet", "color": "navy", "price_idr": 499_000, "tags": ["backpack", "travel", "daily"]},
    {"product_id": "P1009", "name": "Sprint Running Shoes", "category": "footwear", "brand": "Sprint", "color": "blue", "price_idr": 899_000, "tags": ["running", "shoes", "sport"]},
    {"product_id": "P1010", "name": "Brew Coffee Grinder", "category": "kitchen", "brand": "Brew", "color": "black", "price_idr": 549_000, "tags": ["coffee", "grinder", "kitchen"]},
    {"product_id": "P1011", "name": "Shield Phone Case", "category": "mobile", "brand": "Shield", "color": "clear", "price_idr": 129_000, "tags": ["phone", "case", "protective"]},
    {"product_id": "P1012", "name": "Volt Power Bank", "category": "mobile", "brand": "Volt", "color": "black", "price_idr": 399_000, "tags": ["power", "bank", "charger"]},
)

INVENTORY: tuple[dict[str, Any], ...] = (
    {"product_id": "P1001", "stock": 7},
    {"product_id": "P1002", "stock": 0},
    {"product_id": "P1003", "stock": 3},
    {"product_id": "P1004", "stock": 12},
    {"product_id": "P1005", "stock": 4},
    {"product_id": "P1006", "stock": 8},
    {"product_id": "P1007", "stock": 0},
    {"product_id": "P1008", "stock": 5},
    {"product_id": "P1009", "stock": 6},
    {"product_id": "P1010", "stock": 2},
    {"product_id": "P1011", "stock": 15},
    {"product_id": "P1012", "stock": 9},
)

POLICIES: tuple[dict[str, Any], ...] = (
    {"product_id": "P1001", "max_discount_percent": 10, "source": "discount-policy-v1#P1001"},
    {"product_id": "P1002", "max_discount_percent": 5, "source": "discount-policy-v1#P1002"},
    {"product_id": "P1003", "max_discount_percent": 15, "source": "discount-policy-v1#P1003"},
    {"product_id": "P1004", "max_discount_percent": 8, "source": "discount-policy-v1#P1004"},
    {"product_id": "P1005", "max_discount_percent": 12, "source": "discount-policy-v1#P1005"},
    {"product_id": "P1006", "max_discount_percent": 10, "source": "discount-policy-v1#P1006"},
    {"product_id": "P1007", "max_discount_percent": 0, "source": "discount-policy-v1#P1007"},
    {"product_id": "P1008", "max_discount_percent": 10, "source": "discount-policy-v1#P1008"},
    {"product_id": "P1009", "max_discount_percent": 7, "source": "discount-policy-v1#P1009"},
    {"product_id": "P1010", "max_discount_percent": 12, "source": "discount-policy-v1#P1010"},
    {"product_id": "P1011", "max_discount_percent": 5, "source": "discount-policy-v1#P1011"},
    {"product_id": "P1012", "max_discount_percent": 10, "source": "discount-policy-v1#P1012"},
)


def environment_fixtures() -> dict[str, list[dict[str, Any]]]:
    """Return independent catalog, inventory, and policy fixtures.

    Returns:
        A dictionary containing deep-copied lists that callers may mutate or
        serialize without changing the module-level source fixtures.
    """

    return {
        "catalog": deepcopy(list(CATALOG)),
        "inventory": deepcopy(list(INVENTORY)),
        "policies": deepcopy(list(POLICIES)),
    }


class SellerEnvironment:
    """Read-only deterministic tools for product search and safe offers."""

    def __init__(self) -> None:
        """Initialize independent in-memory copies of all environment data.

        Returns:
            ``None``; the initialized catalog, inventory, and policy mappings
            are stored on the new instance.
        """
        self.catalog = {item["product_id"]: deepcopy(item) for item in CATALOG}
        self.inventory = {item["product_id"]: deepcopy(item) for item in INVENTORY}
        self.policies = {item["product_id"]: deepcopy(item) for item in POLICIES}

    @staticmethod
    def _success(tool: str, result: dict[str, Any]) -> dict[str, Any]:
        """Wrap a successful tool result in the common observation envelope.

        Args:
            tool: Name of the tool that produced the result.
            result: Tool-specific successful payload.

        Returns:
            An observation with ``ok=True``, the result payload, and no error.
        """
        return {"tool": tool, "ok": True, "result": result, "error": None}

    @staticmethod
    def _failure(tool: str, code: str, message: str) -> dict[str, Any]:
        """Wrap a failed tool execution in the common observation envelope.

        Args:
            tool: Name of the tool that failed.
            code: Stable machine-readable failure code.
            message: Human-readable explanation of the failure.

        Returns:
            An observation with ``ok=False`` and a structured error object.
        """
        return {
            "tool": tool,
            "ok": False,
            "result": None,
            "error": {"code": code, "message": message},
        }

    def execute(self, action: dict[str, Any]) -> dict[str, Any]:
        """Validate and dispatch one tagged ``tool_call`` action.

        Args:
            action: Action object containing a supported tool and its arguments.

        Returns:
            The structured observation returned by the selected tool.

        Raises:
            SchemaError: If the action violates the action contract.
            ValueError: If the validated action is not a tool call.
        """
        validate_action(action)
        if action["type"] != "tool_call":
            raise ValueError("only tool_call actions can be executed")
        tool = action["tool"]
        arguments = action["arguments"]
        return getattr(self, tool)(**arguments)

    def search_products(self, query: str, filters: dict[str, Any]) -> dict[str, Any]:
        """Find up to five catalog products matching text and exact filters.

        Args:
            query: Case-insensitive text searched across product metadata.
            filters: Validated optional brand, category, color, stock, or price
                constraints.

        Returns:
            A success observation containing matching product summaries. The
            result is empty when no product matches.
        """
        query_lower = query.casefold()
        matches: list[dict[str, Any]] = []
        for product_id in sorted(self.catalog):
            product = self.catalog[product_id]
            searchable = " ".join(
                [product["name"], product["category"], product["brand"], *product["tags"]]
            ).casefold()
            if query_lower not in searchable:
                continue
            if "brand" in filters and product["brand"].casefold() != filters["brand"].casefold():
                continue
            if "category" in filters and product["category"] != filters["category"]:
                continue
            if "color" in filters and product["color"] != filters["color"]:
                continue
            if "max_price_idr" in filters and product["price_idr"] > filters["max_price_idr"]:
                continue
            stock = self.inventory[product_id]["stock"]
            if filters.get("in_stock") is True and stock <= 0:
                continue
            matches.append(
                {
                    "product_id": product_id,
                    "name": product["name"],
                    "price_idr": product["price_idr"],
                    "stock": stock,
                }
            )
        return self._success("search_products", {"products": matches[:5]})

    def get_inventory(self, product_id: str) -> dict[str, Any]:
        """Read the current stock count for one product without mutation.

        Args:
            product_id: Product identifier to look up.

        Returns:
            A success observation with ``product_id`` and ``stock`` or a
            ``product_not_found`` failure observation.
        """
        if product_id not in self.inventory:
            return self._failure("get_inventory", "product_not_found", f"Unknown product {product_id}")
        return self._success("get_inventory", deepcopy(self.inventory[product_id]))

    def get_discount_policy(self, product_id: str) -> dict[str, Any]:
        """Read the maximum discount policy for one product.

        Args:
            product_id: Product identifier to look up.

        Returns:
            A success observation with the discount limit and source, or a
            ``product_not_found`` failure observation.
        """
        if product_id not in self.policies:
            return self._failure(
                "get_discount_policy", "product_not_found", f"Unknown product {product_id}"
            )
        return self._success("get_discount_policy", deepcopy(self.policies[product_id]))

    def calculate_offer(
        self,
        product_id: str,
        discount_percent: int,
        quantity: int,
    ) -> dict[str, Any]:
        """Calculate a policy-compliant discounted total without changing stock.

        Args:
            product_id: Product identifier whose price and policy are used.
            discount_percent: Requested integer discount from 0 through the
                product's maximum allowed percentage.
            quantity: Positive number of units requested.

        Returns:
            A success observation containing unit and total prices, or a
            structured failure for an unknown product, insufficient stock, or
            policy violation.
        """
        if product_id not in self.catalog:
            return self._failure("calculate_offer", "product_not_found", f"Unknown product {product_id}")
        stock = self.inventory[product_id]["stock"]
        if quantity > stock:
            return self._failure(
                "calculate_offer",
                "insufficient_stock",
                f"Requested {quantity}, available {stock}",
            )
        maximum = self.policies[product_id]["max_discount_percent"]
        if discount_percent > maximum:
            return self._failure(
                "calculate_offer",
                "discount_policy_violation",
                f"Requested {discount_percent}%, maximum {maximum}%",
            )
        unit_price = self.catalog[product_id]["price_idr"]
        discounted_unit_price = unit_price * (100 - discount_percent) // 100
        return self._success(
            "calculate_offer",
            {
                "discount_percent": discount_percent,
                "product_id": product_id,
                "quantity": quantity,
                "total_idr": discounted_unit_price * quantity,
                "unit_price_idr": discounted_unit_price,
            },
        )
