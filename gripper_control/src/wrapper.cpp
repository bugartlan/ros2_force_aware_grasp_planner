#include "rclcpp/rclcpp.hpp"
#include "ur_msgs/srv/set_io.hpp"
#include "std_srvs/srv/set_bool.hpp"

class GripperControl : public rclcpp::Node
{
public:
    GripperControl() : Node("gripper_control")
    {
        client_ = this->create_client<ur_msgs::srv::SetIO>("/io_and_status_controller/set_io");
        while (!client_->wait_for_service(std::chrono::seconds(1))) {
            if (!rclcpp::ok()) {
                RCLCPP_ERROR(this->get_logger(), "Interrupted while waiting for the service. Exiting.");
                return;
            }
            RCLCPP_INFO(this->get_logger(), "Service not available, waiting again...");
        }

        server_ = this->create_service<std_srvs::srv::SetBool>(
            "gripper_command",
            std::bind(&GripperControl::server_callback, this, std::placeholders::_1, std::placeholders::_2)
        );
        RCLCPP_INFO(this->get_logger(), "Gripper control service is ready.");
    }

private:
    rclcpp::Client<ur_msgs::srv::SetIO>::SharedPtr client_;
    rclcpp::Service<std_srvs::srv::SetBool>::SharedPtr server_;

    void set_io(int8_t pin, float state) {
        auto set_io_request = std::make_shared<ur_msgs::srv::SetIO::Request>();
        set_io_request->fun = 1; // Set function code for digital output
        set_io_request->pin = pin;
        set_io_request->state = state;

        client_->async_send_request(set_io_request);

    }

    void server_callback(const std::shared_ptr<std_srvs::srv::SetBool::Request> request,
                         std::shared_ptr<std_srvs::srv::SetBool::Response> response)
    {
        // true = open gripper, false = close gripper.
        if (request->data) {
            RCLCPP_INFO(this->get_logger(), "Opening gripper...");
            set_io(9, 1.0);
            response->message = "Gripper opened";
        } else {
            RCLCPP_INFO(this->get_logger(), "Closing gripper...");
            set_io(8, 1.0);
            response->message = "Gripper closed";
        }
        response->success = true;
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<GripperControl>());
    rclcpp::shutdown();
    return 0;
}