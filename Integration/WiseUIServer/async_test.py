import asyncio
import time

from stopwatch import StopWatch


async def say_step(id):
    print(id, 1)
    print(id, 2)
    print(id, 3)
    await asyncio.sleep(1)

async def main():
    stopWatch = StopWatch('test')
    stopWatch.start()
    task1 = asyncio.create_task(say_step('aa'))
    task2 = asyncio.create_task(say_step('bb'))
    await asyncio.gather(task1, task2)
    stopWatch.stop()
    print (stopWatch.get_elapsed_seconds())

async def say_after(delay, what):
    await asyncio.sleep(delay)
    print(what)


async def main1():
    stopWatch = StopWatch('test1')
    tasks = [say_after(1, "func1"), say_after(2, "func2")]
    stopWatch.start()
    await asyncio.gather(*tasks)
    stopWatch.stop()
    print (stopWatch.get_elapsed_seconds())

list_data = []
lock = asyncio.Lock()


async def func1(name, iter):
    for i in range(iter):
        # async with lock:
        list_data.append(i)
        print(name)
        await asyncio.sleep(0)


async def func2(name):
    while True:
        if list_data:
            # async with lock:
            data = list_data.pop(0)
            time.sleep(0.1)
            print(name, data)
        else:
            await asyncio.sleep(0)



async def main2():
    stopWatch = StopWatch('test2')
    task1 = asyncio.create_task(func1('aaa', 100))
    task2 = asyncio.create_task(func2('bbb'))

    stopWatch.start()
    await task1
    await task2
    stopWatch.stop()


asyncio.run(main())
