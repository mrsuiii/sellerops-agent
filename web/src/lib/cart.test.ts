import { beforeEach, describe, expect, it } from "vitest";
import { cartActions } from "./cart";

const shirt = { product_id: "1", name: "Kemeja", thumbnail: "", final_price_idr: 100_000, stock: 2 };

describe("cart", () => {
  beforeEach(() => {
    localStorage.clear();
    cartActions.clear();
  });

  it("adds items and never exceeds stock", () => {
    cartActions.add(shirt);
    cartActions.add(shirt);
    cartActions.add(shirt);
    expect(JSON.parse(localStorage.getItem("kala-cart") ?? "{}")["1"].qty).toBe(2);
  });

  it("removes a line when the quantity drops to zero", () => {
    cartActions.add(shirt);
    cartActions.setQty("1", 0);
    expect(JSON.parse(localStorage.getItem("kala-cart") ?? "{}")).toEqual({});
  });

  it("survives a broken localStorage value", () => {
    localStorage.setItem("kala-cart", "{not json");
    cartActions.add(shirt);
    expect(JSON.parse(localStorage.getItem("kala-cart") ?? "{}")["1"].qty).toBe(1);
  });
});
