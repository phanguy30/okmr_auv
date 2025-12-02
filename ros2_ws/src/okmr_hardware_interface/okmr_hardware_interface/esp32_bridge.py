#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from okmr_msgs.msg import MotorThrottle
from okmr_msgs.msg import BatteryVoltage
from okmr_msgs.msg import MissionCommand
from okmr_msgs.msg import SensorReading
from okmr_msgs.msg import EnvironmentData
from std_msgs.msg import String

import serial
import seaport as sp


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


class ESP32BridgeNode(Node):
    def __init__(self):
        super().__init__("esp32_bridge")

        self.declare_parameter("serial_port", "/dev/ttyUSB0")
        self.declare_parameter("baud_rate", 115200)
        self.declare_parameter("killswitch_address", 66)
        self.declare_parameter("killswitch_index", 0)
        self.declare_parameter("mission_button_address", 66)
        self.declare_parameter("mission_button_index", 1)
        self.declare_parameter("mission_button_arm_time_ms", 5000)
        self.declare_parameter("battery_voltage_address", [67])
        self.declare_parameter("battery_voltage_index", [2, 3, 4, 5])
        self.declare_parameter(
            "battery_voltage_read_mult", [1.0, 3.2222, 5.4444, 5.4444]
        )
        self.declare_parameter("motor_count", 8)
        self.declare_parameter("motor_index_remapping", [0, 1, 2, 3, 4, 5, 6, 7])
        self.declare_parameter(
            "leak_sensor_addresses", [66, 67, 69]
        )  # addresses correspond with their indicies at offset of -66
        self.declare_parameter("leak_sensor_indicies", [1, 1, 1])
        self.declare_parameter(
            "leak_sensor_threshold", 2000.0
        )  # Analog threshold for leak detection
        self.declare_parameter("max_throttle", 1800.0)
        self.declare_parameter("min_throttle", 1200.0)

        serial_port = (
            self.get_parameter("serial_port").get_parameter_value().string_value
        )
        baud_rate = self.get_parameter("baud_rate").get_parameter_value().integer_value
        self.leak_sensor_addresses = (
            self.get_parameter("leak_sensor_addresses")
            .get_parameter_value()
            .integer_array_value
        )
        self.leak_sensor_indices = (
            self.get_parameter("leak_sensor_indicies")
            .get_parameter_value()
            .integer_array_value
        )
        self.leak_sensor_threshold = (
            self.get_parameter("leak_sensor_threshold")
            .get_parameter_value()
            .double_value
        )

        self.killswitch_address = (
            self.get_parameter("killswitch_address").get_parameter_value().integer_value
        )
        self.killswitch_index = (
            self.get_parameter("killswitch_index").get_parameter_value().integer_value
        )
        self.motor_count = (
            self.get_parameter("motor_count").get_parameter_value().integer_value
        )
        self.mission_button_address = (
            self.get_parameter("mission_button_address")
            .get_parameter_value()
            .integer_value
        )
        self.mission_button_index = (
            self.get_parameter("mission_button_index")
            .get_parameter_value()
            .integer_value
        )
        self.mission_button_arm_time_ms = (
            self.get_parameter("mission_button_arm_time_ms")
            .get_parameter_value()
            .integer_value
        )
        self.battery_voltage_address = (
            self.get_parameter("battery_voltage_address")
            .get_parameter_value()
            .integer_array_value
        )
        self.battery_voltage_index = (
            self.get_parameter("battery_voltage_index")
            .get_parameter_value()
            .integer_array_value
        )

        self.battery_voltage_read_mult = (
            self.get_parameter("battery_voltage_read_mult")
            .get_parameter_value()
            .double_array_value
        )

        self.motor_index_remapping = (
            self.get_parameter("motor_index_remapping")
            .get_parameter_value()
            .integer_array_value
        )

        self.max_throttle = (
            self.get_parameter("max_throttle").get_parameter_value().double_value
        )

        self.min_throttle = (
            self.get_parameter("min_throttle").get_parameter_value().double_value
        )

        self.get_logger().info(f"Motor remapping: {self.motor_index_remapping}")

        self.motor_throttle_sub = self.create_subscription(
            MotorThrottle, "/motor_throttle", self.motor_callback, 10
        )

        self.ping_sub = self.create_subscription(String, "/ping", self.ping_callback, 10)

        self.battery_voltage_pub = self.create_publisher(
            BatteryVoltage, "/battery_voltage", 10
        )

        self.mission_command_pub = self.create_publisher(
            MissionCommand, "/mission_command", 10
        )

        self.leak_sensor_pub = self.create_publisher(SensorReading, "/leak_sensor", 10)

        self.environment_data_pub = self.create_publisher(
            EnvironmentData, "/environment_data", 10
        )

        try:
            self.ser = serial.Serial(serial_port, baud_rate, timeout=1)
            self.seaport = sp.SeaPort(self.ser)
            self.get_logger().info(
                f"Serial connection to ESP32 established on {serial_port} at {baud_rate} baud."
            )
            self.get_logger().info(
                f"Killswitch configured: address={self.killswitch_address}, index={self.killswitch_index}"
            )
            self.get_logger().info(
                f"Mission button configured: address={self.mission_button_address}, index={self.mission_button_index}"
            )
            self.get_logger().info(
                f"Mission button arm time: {self.mission_button_arm_time_ms}ms"
            )
            self.get_logger().info(
                f"Leak sensors configured: addresses={self.leak_sensor_addresses}, indices={self.leak_sensor_indices}, threshold={self.leak_sensor_threshold}"
            )

            # self.seaport.subscribe(3, lambda data: imu_accel_callback(data))
            # self.seaport.subscribe(4, lambda data: imu_gyro_callback(data))
            # self.seaport.subscribe(5, lambda data: imu_meta_callback(data))
            self.seaport.subscribe(  # make new msg type for leak sensor
                6, lambda data: self.sensor_board_analog_reading_callback(data)
            )
            self.seaport.subscribe(  # handle killswitch and button callback logic in THIS callback
                7, lambda data: self.sensor_board_digital_reading_callback(data)
            )
            self.seaport.subscribe(254, lambda data: self.pong_callback(data))
            self.seaport.subscribe(
                2, lambda data: self.environment_sensor_callback(data)
            )
            self.seaport.start()

        except Exception as e:
            self.get_logger().error(f"Failed to connect to ESP32: {e}")
            self.ser = None
            self.seaport = None

        # Killswitch state tracking
        self.killswitch_active = False
        self.last_killswitch_state = False

        # Mission button state tracking
        self.last_mission_button_state = False
        self.mission_button_press_time = None
        self.mission_armed = False

    def environment_sensor_callback(self, data: dict):
        """Handle environment sensor data from ESP32"""
        try:
            # Create EnvironmentData message
            env_msg = EnvironmentData()
            env_msg.header.stamp = self.get_clock().now().to_msg()
            env_msg.header.frame_id = "environment_sensor"

            # Extract data from ESP32 message
            # Expected format: {"a": address, "temp": temperature, "hum": humidity, "press": pressure}
            env_msg.board_address = data.get("a", 0)
            env_msg.temperature = float(data.get("t", 0.0))
            env_msg.humidity = float(data.get("h", 0.0))
            env_msg.pressure = float(data.get("p", 0.0))

            # Publish environment data
            self.environment_data_pub.publish(env_msg)

            # Log environment data (can be disabled in production)
            self.get_logger().debug(
                f"Environment data from board {env_msg.board_address}: "
                f"T={env_msg.temperature:.1f}°C, H={env_msg.humidity:.1f}%, "
                f"P={env_msg.pressure:.1f}hPa"
            )

        except Exception as e:
            self.get_logger().error(f"Error processing environment sensor data: {e}")

    def pong_callback(self, data: dict):
        self.get_logger().info(f"Got pong data: {data}")
        pass

    def sensor_board_analog_reading_callback(self, data: dict):
        """Handle analog inputs including leak sensors"""
        # Check if this is leak sensor data
        address = data.get("a")
        index = data.get("i")
        value = data.get("v", 0.0)
        # self.get_logger().info(f"ANALOG READING -- a: {address}, i: {index}, v: {value} ")
        # Check if this matches any of our configured leak sensors
        for i, (leak_addr, leak_idx) in enumerate(
            zip(self.leak_sensor_addresses, self.leak_sensor_indices)
        ):
            if address == leak_addr and index == leak_idx:
                self.leak_sensor_callback(data, i)
                break
        for i, (batt_addr, batt_idx) in enumerate(
            zip(self.battery_voltage_address, self.battery_voltage_index)
        ):
            if address == batt_addr and index == batt_idx:
                self.battery_voltage_callback(data, i)
                break

    def sensor_board_digital_reading_callback(self, data: dict):
        """Handle digital inputs including killswitch from sensor boards"""
        # self.get_logger().info(f"Got digital data: {data}")

        # Check if this is killswitch data (configurable address)
        if (
            data.get("a") == self.killswitch_address
            and data.get("i") == self.killswitch_index
        ):
            self.killswitch_callback(data)

        # Check if this is mission button data (configurable address)
        # elif (
        #    data.get("a") == self.mission_button_address
        #    and data.get("i") == self.mission_button_index
        # ):
        #    self.mission_button_callback(data)

    def killswitch_callback(self, data: dict):
        """Handle killswitch state changes from ESP32"""

        try:
            # Expect data format: {"a": address, "i": index, "v": 0/1}
            # Killswitch is a dial - when pulled it stays active until physically reset
            killswitch_pulled = data["v"]

            # Update current physical state
            self.last_killswitch_state = killswitch_pulled

            # If killswitch is pulled and wasn't already active, trigger kill
            if killswitch_pulled and not self.killswitch_active:
                self.killswitch_active = True
                self.get_logger().error("HARDWARE KILLSWITCH ACTIVATED!")
                self.stop_all_motors()

                # Disarm mission button if armed
                if self.mission_armed:
                    self.mission_armed = False
                    self.get_logger().warn(
                        "Mission button disarmed due to killswitch activation"
                    )

                # Publish hardware kill command
                msg = MissionCommand()
                msg.command = MissionCommand.HARDWARE_KILL
                self.mission_command_pub.publish(msg)

            elif not killswitch_pulled and self.killswitch_active:
                # Killswitch dial was reset - automatically clear kill state
                self.killswitch_active = False
                self.get_logger().info("Hardware killswitch reset - system operational")

        except Exception as e:
            self.get_logger().error(f"Error processing killswitch data: {e}")

    def mission_button_callback(self, data: dict):
        """Handle mission button state changes from ESP32"""
        try:
            # Expect data format: {"a": address, "i": index, "v": 0/1}
            button_pressed = not data["v"]
            current_time = self.get_clock().now()

            # Detect button press (rising edge)
            if button_pressed and not self.last_mission_button_state:
                self.get_logger().info("Mission button pressed - starting hold timer")
                self.mission_button_press_time = current_time

            # Button is being held - check for arming
            elif button_pressed and self.last_mission_button_state:
                if self.mission_button_press_time is not None:
                    hold_duration_ms = (
                        current_time - self.mission_button_press_time
                    ).nanoseconds / 1e6

                    # Check if we should arm the system and start mission
                    if (
                        not self.mission_armed
                        and hold_duration_ms >= self.mission_button_arm_time_ms
                    ):
                        # Check killswitch before allowing arming
                        if self.killswitch_active:
                            self.get_logger().warn(
                                "Cannot arm system - hardware killswitch active"
                            )
                        else:
                            self.mission_armed = True
                            self.get_logger().info(
                                f"Mission button held for {hold_duration_ms:.0f}ms - SYSTEM ARMED"
                            )

                            # Immediately start mission when armed
                            msg = MissionCommand()
                            msg.command = MissionCommand.START_MISSION
                            self.mission_command_pub.publish(msg)
                            self.get_logger().info(
                                "Mission start command published - MISSION STARTED"
                            )

            # Detect button release (falling edge)
            elif not button_pressed and self.last_mission_button_state:
                if self.mission_button_press_time is not None:
                    hold_duration_ms = (
                        current_time - self.mission_button_press_time
                    ).nanoseconds / 1e6
                    self.get_logger().info(
                        f"Mission button released after {hold_duration_ms:.0f}ms"
                    )

                    # Small pulse (< arm time) = disarm if currently armed
                    if hold_duration_ms < self.mission_button_arm_time_ms:
                        if self.mission_armed:
                            self.mission_armed = False
                            self.get_logger().info(
                                "Mission button pulse detected - SYSTEM DISARMED"
                            )
                            self.stop_all_motors()
                        else:
                            self.get_logger().info(
                                f"Button released before arming time ({self.mission_button_arm_time_ms}ms)"
                            )

                    # Long press release - no action needed (mission already started when armed)
                    else:
                        self.get_logger().info(
                            "Mission button released after long press - no action needed"
                        )

                # Reset press time
                self.mission_button_press_time = None

            # Update button state
            self.last_mission_button_state = button_pressed

        except Exception as e:
            self.get_logger().error(f"Error processing mission button data: {e}")

    def stop_all_motors(self):
        try:
            for i in range(self.motor_count):
                data = {str(i): 1500.0}
                self.seaport.publish(1, data)
        except Exception as e:
            self.get_logger().error(f"Failed to send safety stop to ESP32: {e}")

    def battery_voltage_callback(self, data: dict, sensor_index: int):
        """Handle battery voltage readings and publish"""
        try:
            # Expect data format: {"a": address, "i": index, "v": analog_value}
            address = data.get("a")
            index = data.get("i")
            raw_value = data.get("v", 0.0)

            # Apply multiplication factor for this sensor
            if sensor_index < len(self.battery_voltage_read_mult):
                voltage_reading = (
                    raw_value * self.battery_voltage_read_mult[sensor_index]
                )
            else:
                voltage_reading = raw_value

            # Store reading for cell voltage calculation
            # We expect 4 readings corresponding to pins 2,3,4,5 (cells 1,2,3,4)
            if not hasattr(self, "battery_readings"):
                self.battery_readings = {}

            self.battery_readings[index] = voltage_reading

            # Check if we have all 4 cell readings (indices 2,3,4,5)
            expected_indices = [2, 3, 4, 5]
            if all(idx in self.battery_readings for idx in expected_indices):
                # Sort readings by voltage (ascending order)
                sorted_readings = sorted(
                    [(idx, self.battery_readings[idx]) for idx in expected_indices],
                    key=lambda x: x[1],
                )

                # Calculate individual cell voltages
                # Cell 1 = reading 1
                # Cell 2 = reading 2 - reading 1
                # Cell 3 = reading 3 - reading 2
                # Cell 4 = reading 4 - reading 3
                cell_voltages = []
                prev_voltage = 0.0

                for i, (idx, voltage) in enumerate(sorted_readings):
                    cell_voltage = voltage - prev_voltage
                    cell_voltages.append(cell_voltage)
                    prev_voltage = voltage

                # Create and publish battery voltage message
                battery_msg = BatteryVoltage()
                battery_msg.header.stamp = self.get_clock().now().to_msg()
                battery_msg.header.frame_id = f"battery_voltage_{address}"
                battery_msg.cell_voltages = cell_voltages
                battery_msg.total_voltage = sum(cell_voltages)

                self.battery_voltage_pub.publish(battery_msg)

                self.get_logger().debug(
                    f"Battery voltages - Cell 1: {cell_voltages[0]:.2f}V, "
                    f"Cell 2: {cell_voltages[1]:.2f}V, Cell 3: {cell_voltages[2]:.2f}V, "
                    f"Cell 4: {cell_voltages[3]:.2f}V, Total: {battery_msg.total_voltage:.2f}V"
                )

                # Clear readings for next cycle
                # self.battery_readings.clear()

        except Exception as e:
            self.get_logger().error(f"Error processing battery voltage data: {e}")

    def leak_sensor_callback(self, data: dict, sensor_index: int):
        """Handle leak sensor readings and publish warnings"""
        try:
            # Expect data format: {"a": address, "i": index, "v": analog_value}
            address = data.get("a")
            index = data.get("i")
            value = data.get("v", 0.0)

            # Create and publish sensor reading message
            sensor_msg = SensorReading()
            sensor_msg.header.stamp = self.get_clock().now().to_msg()
            sensor_msg.header.frame_id = f"leak_sensor_{address}_{index}"
            sensor_msg.data = float(value)

            self.leak_sensor_pub.publish(sensor_msg)

            # Check threshold and issue warning
            if value > self.leak_sensor_threshold:
                self.get_logger().error(
                    f"LEAK DETECTED! Sensor {address}:{index} reading {value:.1f} "
                    f"exceeds threshold {self.leak_sensor_threshold:.1f}"
                )
                msg = MissionCommand()
                msg.command = MissionCommand.KILL_MISSION
                self.mission_command_pub.publish(msg)
            else:
                self.get_logger().debug(
                    f"Leak sensor {address}:{index} reading: {value:.1f}"
                )

        except Exception as e:
            self.get_logger().error(f"Error processing leak sensor data: {e}")

    def ping_callback(self, msg):
        self.seaport.publish(254, {"cmd": "ping"})

    def motor_callback(self, msg: MotorThrottle):
        if not self.seaport:
            self.get_logger().warn("No serial connection. Message not sent.")
            return

        # Check killswitch state before sending motor commands
        if self.killswitch_active:
            self.get_logger().warn("Motor command blocked - disarmed")
            # Send zero throttle to all motors as safety measure
            self.stop_all_motors()
            return
        try:
            for i, throttle in enumerate(msg.throttle):
                if throttle != 0.0:
                    data = {
                        str(self.motor_index_remapping[i]): clamp(
                            float(throttle), self.min_throttle, self.max_throttle
                        )
                    }
                    self.seaport.publish(1, data)
                # self.get_logger().info(f"Sent to ESP32: {data}")
        except Exception as e:
            self.get_logger().error(f"Failed to send to ESP32: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ESP32BridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.ser:
            node.ser.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
