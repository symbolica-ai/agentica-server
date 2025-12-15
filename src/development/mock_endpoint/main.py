import uvicorn


def get_app():
    from .server import app as asgi_app

    return asgi_app


def main() -> None:
    uvicorn.run("development.mock_endpoint.server:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
