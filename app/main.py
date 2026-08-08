from fastapi import FastAPI
from fastapi.openapi.docs import get_swagger_ui_html

from app.api.query import router as query_router
from app.api.reindex import router as reindex_router
from app.api.upload import router as upload_router

app = FastAPI(title="Agentic RAG")


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,  # type: ignore
        title=app.title + " - Swagger UI",
        # Newer Swagger UI build (5.17+) with better OpenAPI 3.1 / contentMediaType support
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css",
    )


app.include_router(upload_router, prefix="/api/v1/knowledge", tags=["Knowledge Base"])

app.include_router(reindex_router, prefix="/api/v1/knowledge", tags=["Knowledge Base"])

app.include_router(query_router, prefix="/api/v1/query", tags=["Query"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=6001)
