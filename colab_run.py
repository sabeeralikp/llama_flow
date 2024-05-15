import os
import nest_asyncio
from pyngrok import ngrok
import uvicorn

# from main import app
from dotenv import load_dotenv
import multiprocessing

load_dotenv()

NGORK_AUTH_TOKEN = os.environ.get("NGORK_AUTH_TOKEN")

ngrok.set_auth_token(NGORK_AUTH_TOKEN)
ngrok_tunnel = ngrok.connect(8000)
print("Public URL:", ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run("main:app", port=8000, workers=multiprocessing.cpu_count())
