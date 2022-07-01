from mavsdk import System
import asyncio

async def run():
    drone = System()
    await drone.connect(system_address="serial:///dev/ttyUSB0:921600")

    await drone.action.arm()
    #asyncio.ensure_future(print_gps_info(drone))
    #asyncio.ensure_future(print_in_air(drone))
    #asyncio.ensure_future(print_position(drone))

async def print_battery(drone):
    async for battery in drone.telemetry.battery():
        print(f"Battery: {battery.remaining_percent}")


if __name__ == "__main__":
    asyncio.ensure_future(run())
    asyncio.get_event_loop().run_forever()
