# Mavlink

## [Mavlink Configuration](https://docs.px4.io/master/en/peripherals/mavlink_peripherals.html)

Mavlink 是无人机与 GCS,Companion Computer,OSD(On-Screen Displays) 通信的协议。 mavlink 通过串口传输。无人机通过多个 `mavlink instance`同时多个设备通信。
> 下面 `X` 代表第 `X` 个 `mavlink instance`

[`MAV_X_CONFIG`](https://docs.px4.io/master/en/advanced_config/parameter_reference.html#MAV_0_CONFIG):设置 `mavlink instance` 的串口号，所有串口见 [`Serial Port`](https://docs.px4.io/master/en/peripherals/serial_configuration.html)

`MAV_X_MODE`: specify the telemetry mode/target. Default modes include:

1. ExtVision / ExtVisionMin: Messages for offboard vision systems.
2. Iridium: Iridium satellite communication system.
3. Config: Standard set of messages and rate configuration for a fast link(e.g. USB)

`MAV_X_RATE`: Set the minimum data rate for this instance(bytes/second)

`MAV_X_FORWARD`: Enable message packets transfered to other mavlink instances.

`SER_XXXX_BAUD`: set the baudrate for the specified serial port.

> 一般 `TELEM 1` 设置为与地面站通信的串口。所以 micro-usb 可能对应这个串口

## Mavlink Main Program

```bash
mavlink start <其他参数> # 启动 mavlink 服务
mavlink stop-all # 停止 mavlink 服务
```

* -d 指定设备接口

在使用 Pixhawk 与计算机连接时，都是采用串口的方式启动飞控程序的`Mavlink`服务。如果采用`USB`直连，使用是`/dev/ttyACM0`;如果使用数传，为`/dev/ttyS1`。

### Source Code

主程序位于 `/src/modules/mavlink/mavlink_main.cpp` 下 `Mavlink::task_main` 。先是配置参数，之后用 switch-case 语句配置不同模式下发送的消息类。eg. `MAVLINK_MODE_NORMAL`, `MAVLINK_MODE_ONBOARD`, ...

### Mavlink Stream 发送器基类

类中定义了 `send()` 虚函数作为不同消息的发送函数。定义 `update()` 函数，作为统一的发送更新函数，在 `task_main` 中调用。

### 消息子类

消息类继承 `MavlinkStream` 发送具体消息。在 `src/modules/mavlink/mavlink_messages.cpp` 中有不同的消息类。在 `/mavlink/v2.0/common` 目录下找到相关的头文件，里面有定义的消息类型和发送消息函数。发送机制与 ROS 相似。都是创建结构体（类）并赋值，调用头文件下的 `XXX_send_struct` 函数发送, eg.

```C
mavlink_msg_utm_global_position_send_struct(_mavlink->get_channel(), &msg);
```

## Receive Process

1. Parse the incoming stream into objects representing each packet(`mavlink_message_t`)
2. Decode the MAVLINK message contained in the packet payload into C struct (that has field mapping the original XML definition)

General work flow template:

```C
#include <common/mavlink.h>

mavlink_status_t status;
mavlink_message_t msg;
int chan = MAVLINK_COMM_0;

while(serial.bytesAvailable > 0)
{
  uint8_t byte = serial.getNextByte();
  if (mavlink_parse_char(chan, byte, &msg, &status))
    {
    printf("Received message with ID %d, sequence: %d from component %d of system %d\n", msg.msgid, msg.seq, msg.compid, msg.sysid);
    switch(msg.msgid) {
      case MAVLINK_MSG_ID_GLOBAL_POSITION_INT: // ID for GLOBAL_POSITION_INT
        {
          // Get all fields in payload (into global_position)
          mavlink_msg_global_position_int_decode(&msg, &global_position);

        }
        break;
      case MAVLINK_MSG_ID_GPS_STATUS:
        {
          // Get just one field from payload
          visible_sats = mavlink_msg_gps_status_get_satellites_visible(&msg);
        }
        break;
     default:
        break;
    }
}
```
