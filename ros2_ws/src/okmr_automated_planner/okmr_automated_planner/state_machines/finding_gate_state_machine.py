from okmr_msgs.msg import MovementCommand
from okmr_msgs.msg import GoalVelocity
from okmr_msgs.srv import Status, SetInferenceCamera, ChangeModel
from geometry_msgs.msg import Vector3

from okmr_msgs.msg import BoundingBox

from okmr_automated_planner.base_state_machine import BaseStateMachine
# from okmr_utils.okmr_utils.logging import make_green_log


class FindingGateStateMachine(BaseStateMachine):
    # finding_gate.yaml

    PARAMETERS = [
        {
            "name": "frame_confidence_threshold",
            "value": 0.8,
            "descriptor": 'threshold that decides if a gate detection is "good enough"',
        },
        {
            "name": "initial_detection_frame_threshold",
            "value": 3,
            "descriptor": "how many detected frames to begin rotating towards detection",
        },
        {
            "name": "true_positive_frame_threshold",
            "value": 20,
            "descriptor": "how many frames to consider detection as true positive",
        },
        {
            "name": "max_scan_attempts",
            "value": 4,
            "descriptor": "how many times do we scan back and forth before giving up",
        },
        # above params not used
        {
            "name": "scan_speed",
            "value": 30.0,
            "descriptor": "how fast to rotate while looking for gate",
        },
        {
            "name": "scan_angle",
            "value": 360.0,
            "descriptor": "how much of an angle to cover while searching for gate",
        },
        {
            "name": "image_width",
            "value": 640.0,
            "descriptor": "width of camera image in pixels",
        },
        {
            "name": "image_height",
            "value": 480.0,
            "descriptor": "height of camera image in pixels",
        },
        {
            "name": "centering_threshold_pixels",
            "value": 50.0,
            "descriptor": "how many pixels from center the bounding box can be",
        },
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # parameters
        self.confidence_threshold = self.get_local_parameter(
            "frame_confidence_threshold"
        )
        self.initial_detection_frame_threshold = self.get_local_parameter(
            "initial_detection_frame_threshold"
        )
        self.true_positive_frame_threshold = self.get_local_parameter(
            "true_positive_frame_threshold"
        )
        self.max_scan_attempts = self.get_local_parameter("gate_max_scan_attempts")
        self.scan_speed = self.get_local_parameter("scan_speed")
        self.scan_angle = self.get_local_parameter("scan_angle")
        self.image_width = self.get_local_parameter("image_width")
        self.image_height = self.get_local_parameter("image_height")
        self.centering_threshold_pixels = self.get_local_parameter("centering_threshold_pixels")

        self.high_confidence_frame_count = 0
        self.scans_completed = 0
        self.cached_gate_bounding_box = None

        self.image_center_x = self.image_width / 2.0
        self.image_center_y = self.image_height / 2.0

        # Subscribe to bounding box detection topic
        self.detection_subscription = self.ros_node.create_subscription(
            BoundingBox, "/bounding_box", self.detection_callback, 10
        )
        self._subscriptions.append(self.detection_subscription)

    def set_inference_camera(self, camera_mode):
        request = SetInferenceCamera.Request()
        request.camera_mode = camera_mode
        self.send_service_request(SetInferenceCamera, "/set_inference_camera", request, lambda f: None)

    def change_model(self, model_id):
        request = ChangeModel.Request()
        request.model_id = model_id
        self.send_service_request(ChangeModel, "/change_model", request, lambda f: None)

    def detection_callback(self, msg):
        self.cached_gate_bounding_box = msg
        
        offset_x = abs(msg.x_coordinate - self.image_center_x)
        offset_y = abs(msg.y_coordinate - self.image_center_y)
        
        if offset_x > self.centering_threshold_pixels or offset_y > self.centering_threshold_pixels:
            if not self.is_following_detection():
                self.follow_detection()

    def on_enter_initializing(self):
        self.change_model(ChangeModel.Request.GATE)
        self.set_inference_camera(SetInferenceCamera.Request.FRONT_CAMERA)
        self.queued_method = self.initializing_done

    def on_enter_scanning_cw(self):
        movement_msg = MovementCommand()
        movement_msg.command = MovementCommand.SET_VELOCITY
        movement_msg.goal_velocity = GoalVelocity()

        movement_msg.goal_velocity.twist.angular.z = -self.scan_speed
        movement_msg.goal_velocity.duration = self.scan_angle / abs(self.scan_speed)
        movement_msg.goal_velocity.integrate = True

        movement_msg.timeout_sec = self.scan_angle / abs(self.scan_speed) * 2

        success = self.movement_client.send_movement_command(
            movement_msg,
            on_success=self.scanning_cw_done,
            on_failure=self.handle_movement_failure,
        )

        if not success:
            self.ros_node.get_logger().error(
                "Failed to send scanning CW movement command"
            )
            self.queued_method = self.abort

    def on_enter_scanning_ccw(self):
        movement_msg = MovementCommand()
        movement_msg.command = MovementCommand.SET_VELOCITY
        movement_msg.goal_velocity = GoalVelocity()

        movement_msg.goal_velocity.twist.angular.z = self.scan_speed
        movement_msg.goal_velocity.duration = self.scan_angle / abs(self.scan_speed)
        movement_msg.goal_velocity.integrate = True

        movement_msg.timeout_sec = self.scan_angle / abs(self.scan_speed) * 2

        success = self.movement_client.send_movement_command(
            movement_msg,
            on_success=self.scanning_ccw_done,
            on_failure=self.handle_movement_failure,
        )

        if not success:
            self.ros_node.get_logger().error(
                "Failed to send scanning CCW movement command"
            )
            self.queued_method = self.abort

    def on_enter_following_detection(self):
        """Turning towards detected object to verify if it's a true positive"""
        if self.cached_gate_bounding_box is None:
            self.ros_node.get_logger().warn("No bounding box cached, resuming scan")
            self.resume_scan()
            return

        offset_x = self.cached_gate_bounding_box.x_coordinate - self.image_center_x
        normalized_offset = offset_x / self.image_center_x
        
        # Scale to rotation angle needed
        # Assuming 90 degree FOV, normalized_offset of 1.0 = 45 degrees rotation needed
        calculated_yaw_rotation = 43.0 * normalized_offset

        movement_msg = MovementCommand()
        movement_msg.command = MovementCommand.MOVE_RELATIVE
        movement_msg.rotation = Vector3(
            x=0.0, y=0.0, z=calculated_yaw_rotation
        )  
        movement_msg.timeout_sec = 15.0  # generous time estimate to rotate

        success = self.movement_client.send_movement_command(
            movement_msg,
            on_success=self.check_centered_on_gate,
            on_failure=self.handle_movement_failure,
        )

        if not success:
            self.ros_node.get_logger().error(
                "Failed to send look at gate movement command"
            )
            self.queued_method = self.abort

    def check_centered_on_gate(self):
        """Check if the gate is now centered in view"""
        if self.cached_gate_bounding_box:
            offset_x = abs(self.cached_gate_bounding_box.x_coordinate - self.image_center_x)
            offset_y = abs(self.cached_gate_bounding_box.y_coordinate - self.image_center_y)
            
            if offset_x < self.centering_threshold_pixels and offset_y < self.centering_threshold_pixels:
                self.following_detection_done()
                return
        
        self.resume_scan()

    '''
    def validate_detection_is_true_positive(self):
        """Validate if we have enough high confidence frames to confirm the detected object is a true positive"""

        if self.high_confidence_frame_count >= self.true_positive_frame_threshold:
            self.object_detection_true_positive()
        else:
            self.ros_node.get_logger().warn(
                f"Gate validation failed. Only {self.high_confidence_frame_count} frames received, but {self.true_positive_frame_threshold} needed to continue"
            )
            self.object_detection_false_positive()
    '''

    def on_completion(self):
        self.set_inference_camera(SetInferenceCamera.Request.DISABLED)
        self.ros_node.get_logger().info("FindingGate state machine completed")

    def handle_movement_failure(self):
        """Handle movement action failure"""
        self.ros_node.get_logger().error("Movement action failed")
        self.abort()
