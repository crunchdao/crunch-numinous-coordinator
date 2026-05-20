from .__init__ import app

if True:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
