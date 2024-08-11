import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.app:app",
                host="0.0.0.0",
                port=8000,
                workers=4,
                limit_concurrency=40,
                backlog=300)