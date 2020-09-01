# Mavros Offboard Control

根据 PX4 官方的 python 测试代码分析

## System setup

* wait for ros services

Template:

```python
 # ROS services
service_timeout = 30
rospy.loginfo("waiting for ROS services")
try:
    rospy.wait_for_service('mavros/param/get', service_timeout)
    rospy.loginfo("ROS services are up")
except rospy.ROSException:
    self.fail("failed to connect to services")
```

| service name | function |
| :-: | :-: |
| mavros/param/get | Return FCU parameter value |
| mavros/cmd/arming | Change Arming state |
| mavros/mission/push | Send new waypoint table |
| mavros/mission/clear | clear mission on FCU |
| mavros/set_mode | Set FCU operation mode |

Then set up corresponding service client.

## issues

有时候会出现

```bash
[WARN] [1597716758.154702551, 29.872000000]: CMD: Unexpected command 176, result 0
```

原因是在 `set_mode` 时 `base_mode` 设置有问题。`result 0` 能够正常运行，但是为 1 则不行。可以修改 `base_mode` 为其余值尝试。


