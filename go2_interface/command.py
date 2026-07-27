import time

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


MAX_VX = 0.5
MAX_VY = 0.5
MAX_VYAW = 0.6
MAX_T = 5.0


def clamp(x, lo, hi):
    return max(lo, min(x, hi))


def clamp_velocity(vx, vy, vyaw):
    return (
        clamp(vx, -MAX_VX, MAX_VX),
        clamp(vy, -MAX_VY, MAX_VY),
        clamp(vyaw, -MAX_VYAW, MAX_VYAW),
    )


def make_sport_client(iface, timeout=5.0):
    ChannelFactoryInitialize(0, iface)
    client = SportClient()
    client.SetTimeout(timeout)
    client.Init()
    return client


def stop(client):
    return client.StopMove()


def sit(client):
    client.StopMove()
    time.sleep(0.2)
    return client.Sit()


def free_avoid(client, enabled):
    return client.FreeAvoid(enabled)


def move(client, vx, vy, vyaw):
    vx, vy, vyaw = clamp_velocity(vx, vy, vyaw)
    return client.Move(vx, vy, vyaw), (vx, vy, vyaw)


def move_for(client, vx, vy, vyaw, seconds, hz=20):
    seconds = clamp(seconds, 0.0, MAX_T)
    vx, vy, vyaw = clamp_velocity(vx, vy, vyaw)

    dt = 1.0 / hz
    end = time.monotonic() + seconds
    move_ret = None

    try:
        while time.monotonic() < end:
            move_ret = client.Move(vx, vy, vyaw)
            time.sleep(dt)
    finally:
        stop_ret = stop(client)

    return move_ret, stop_ret, (vx, vy, vyaw), seconds
    