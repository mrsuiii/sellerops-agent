# Kala Studio demo store

A small fashion e-commerce site that serves as the product catalog for the
SellerOps agent. Backend: FastAPI (`src/sellerops/store/`). Frontend: plain
HTML/CSS/JS in this folder, served by the same app. No build step.

## Run

```bash
pip install -r configs/requirements.store.txt
uvicorn sellerops.store.api:app --reload
```

Open http://127.0.0.1:8000 (shop) and http://127.0.0.1:8000/docs (API).

## Data

`data/store/fashion_catalog.json` is a committed snapshot of the fashion
categories (54 products) from the MIT-licensed
[DummyJSON](https://github.com/Ovi/DummyJSON) database: shirts, tops, dresses,
shoes, bags, watches, jewellery, sunglasses, fragrances. Prices are converted
to IDR (1 USD = Rp16,000) and reviewer emails are dropped. Product images are
hot-linked from `cdn.dummyjson.com`, so they need internet. Rebuild with
`python scripts/import_fashion_catalog.py`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/products` | List/search: `q`, `category`, `audience`, `brand`, `min_price`, `max_price`, `min_rating`, `in_stock`, `sort`, `page`, `page_size` |
| GET | `/api/products/{product_id}` | Full detail incl. images and reviews |
| GET | `/api/categories` | Categories with counts, and brands |
| POST | `/api/recommend` | Ranked recommendations with `score` and `reasons` |

## Recommender

`Catalog.recommend` in `src/sellerops/store/catalog.py` is deliberately small:
hard filters (category, audience, price, rating, stock, exclusions), then BM25
over name/category/brand/tags/description, then
`0.6 * relevance + 0.3 * rating/5 + 0.1 * discount/50`. `similar_to` turns an
anchor product into the query. The same function is meant to back the agent's
`recommend_products` tool, so keep its arguments in sync with
`RecommendRequest` in `api.py`.

The cart lives in the browser (localStorage) and checkout is a demo.

## Seller page

`/seller` (`seller.html`, `seller.js`, `seller.css`) is the caption approval
UI: the queue of proposals, an old/new diff per field (word level; tags as
added/removed chips), and approve / reject (feedback + extra fields the
revision may change) / cancel through the approval workflow routes
(`/api/seller/caption-proposals/{id}/approval…`), plus revert for an applied
proposal and the product's audit log. "Masuk sebagai" sets the
`X-Seller-User` header: there is no login, so this is a demo only.

The shop shows each product's current caption from the seller database
(`shop_catalog` in `store/api.py`), so an approved caption appears in the shop
on the next request and a revert brings the old one back. `make store-fake`
runs everything with a free fake model drafting the revisions; `make store`
uses the real LLM for them.
