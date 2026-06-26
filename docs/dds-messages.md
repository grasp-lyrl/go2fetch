# DDS Messages for Fetch

Once powered on, the Go2 publishes and receives DDS messages. For this project, we use:

`rt/sportmodestate`

- type: `unitree_go::msg::dds_::SportModeState_`
- stamp: `[sec, nanosec]`
- imu_state: `[quaternion[4], gyroscope[3], accelerometer[3], rpy[3]]`
- torso position: `[x, y, z]`
- torso velocity: `[vx, vy, vz]`

`rt/utlidar/cloud`

- type: `sensor_msgs::msg::dds_::PointCloud2`_
- stamp: `[sec, nanosec]`
- frame_id: `utlidar_lidar`
- size: `width * height` points
- data: raw binary point cloud buffer

`rt/api/sport/request`

- type: `unitree_api::msg::dds_::Request`_
- api_id: `header.identity.api_id`
- parameters: encoded in `parameter`
- Command [API ID] - explanation:
  - `Move(vx, vy, vyaw)` [1008] - velocity walking
  - `StopMove()` [1003] - stop motion
  - `FreeAvoid(flag)` [2048] - free obstacle avoidance

