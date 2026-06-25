# DDS Messages for Fetch

Once powered on, the Go2 publishes and receives DDS messages. For this project, we use:

`rt/sportmodestate` 

- type: `unitree_go::msg::dds_::SportModeState_`
- stamp: `[sec, nanosec]`
- imu_state: `[quaternion[4], gyroscope[3], accelerometer[3], rpy[3]]`
- torso position: `[x, y, z]`
- torso velocity: `[vx, vy, vz]`
- yaw_speed: scalar
- body_height: scalar

`rt/api/sport/request`

- type: `unitree_api::msg::dds_::Request_`
- api_id: `header.identity.api_id`
- parameters: encoded in `parameter`
- Command [API ID] - explanation:
    - `Move(vx, vy, vyaw)` [1008] - velocity walking
    - `StopMove()` [1003] - stop motion
    - `FreeAvoid(flag)` [2048] - free obstacle avoidance
