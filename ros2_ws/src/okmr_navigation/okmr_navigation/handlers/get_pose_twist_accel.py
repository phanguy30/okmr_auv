from okmr_msgs.srv import GetPoseTwistAccel
from okmr_navigation.navigator_action_server import NavigatorActionServer


def get_current_pose():
    """Get current pose using the GetPoseTwistAccel service"""
    response = _call_get_pose_twist_accel_service()
    return response.pose if response else None


def get_current_twist():
    """Get current twist using the GetPoseTwistAccel service"""
    response = _call_get_pose_twist_accel_service()
    return response.twist if response else None


def get_current_accel():
    """Get current acceleration using the GetPoseTwistAccel service"""
    response = _call_get_pose_twist_accel_service()
    return response.accel if response else None


def _call_get_pose_twist_accel_service():
    """Helper function to call the GetPoseTwistAccel service"""
    node = NavigatorActionServer.get_instance()
    
    try:
        client = node.create_client(GetPoseTwistAccel, '/get_pose_twist_accel')
        
        # Wait for service to be available
        if not client.wait_for_service(timeout_sec=2.0):
            node.get_logger().error('get_pose_twist_accel service not available after 2s timeout')
            return None
        
        # Make the blocking service call
        request = GetPoseTwistAccel.Request()
        response = client.call(request)
        
        # Check response
        if response is not None and hasattr(response, 'success') and response.success:
            return response
        else:
            node.get_logger().error('get_pose_twist_accel service returned failure or invalid response')
            return None
            
    except Exception as e:
        node.get_logger().error(f'Exception in _call_get_pose_twist_accel_service: {str(e)}')
        return None
