import asyncio
from mavsdk import System
from mavsdk.offboard import (OffboardError, PositionNedYaw)


async def adjust(center, shape):
    """
    Takes coordinates of center point, calculates the required change in trajectory to stay on course
    """

    #print(adjustment := shape[1]/2 - center[0], shape[0]/2 - center[1])
    print(shape[1]/2 - center[0], shape[0]/2 - center[1])
    

    
