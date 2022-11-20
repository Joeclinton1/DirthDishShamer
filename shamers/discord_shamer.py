import os
import discord
import threading
from discord.ext import tasks
from dotenv import load_dotenv
import cv2

# Two exports: DishDetector and DishDetectorBlocking
# both can be used as constructors for the DishDetector class
# You can send messages through that class by appending to the msg_queue
# If you use the non-blocking version then you need to acquire the ready_lock before you can use it again


# TODO get these in a non-hard-coded way
SERVER_TARGET = 1043610399465029713
CHANNEL_TARGET = 1043619041207660668
people_to_shame = {
    "josh":460814426572980245,
    "joec":499882945281261588,
    "joes":187570697290252288
}


class DishDetector(discord.Client):
    def __init__(self, token, *args, **kwargs):
        self.msg_queue = []
        self.ready_lock = threading.Lock()
        self.ready_lock.acquire()

        if "intents" not in kwargs:
            kwargs["intents"] = discord.Intents.default()

        super().__init__(*args, **kwargs)
        
        def bot_thread():
            self.run(token)
        t = threading.Thread(target=bot_thread)
        t.start()

    async def on_ready(self):
        print(f"Logged in as {self.user}")
        self.server = await self.fetch_guild(SERVER_TARGET)
        self.channel = await self.server.fetch_channel(CHANNEL_TARGET)
        self.ready_lock.release()
        self.send_loop.start()

    @tasks.loop(seconds=1)
    async def send_loop(self):
        if len(self.msg_queue) > 0:
            txt = self.msg_queue.pop(0)
            await self.channel.send(txt)

    def shame(self, image, name):
        if name != None:
            self.msg_queue.append("SHAME! Shame upon <@{}>".format(people_to_shame[name]))
        else:
            self.msg_queue.append("SHAME! Shame upon you!")

def DishDetectorBlocking(token, *args, **kwargs):
    cl = DishDetector(token, *args, **kwargs)
    cl.ready_lock.acquire()
    return cl


if __name__=="__main__":
    load_dotenv()
    token = os.getenv("DISCORD_TOKEN")

    cl = DishDetectorBlocking(token)

    while True:
        # The main loop goes here
        txt = input(">")
        cl.msg_queue.append(txt)
