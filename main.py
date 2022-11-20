import os
from bot import DishDetectorBlocking
from dotenv import load_dotenv

load_dotenv()
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")


bot = DishDetectorBlocking(DISCORD_TOKEN)
