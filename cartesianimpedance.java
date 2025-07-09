package application;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.motionModel.LIN;
import com.kuka.roboticsAPI.motionModel.controlModeModel.PositionControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.JointImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.PositionHold;
import com.kuka.roboticsAPI.sensorModel.ForceSensorData;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.geometricModel.math.Transformation;
import com.kuka.common.ThreadUtil;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import java.net.*;
import java.io.*;
import java.util.*;
import java.util.concurrent.atomic.AtomicBoolean;

public class Cartesianimpedance extends RoboticsAPIApplication {
    
    // Robot device
    private LBR robot;
    
    // TCP Communication
    private ServerSocket trajectoryServerSocket;
    private Socket trajectoryClientSocket;
    private BufferedReader trajectoryIn;
    private PrintWriter trajectoryOut;
    
    // TCP Client for sending torque data
    private Socket torqueDataSocket;
    private PrintWriter torqueDataOut;
    
    // Control parameters
    private double stiffnessX = 50.0; // N/m
    private double stiffnessY = 50.0;
    private double stiffnessZ = 50.0;
    private double stiffnessRot = 20.0; // Nm/rad
    private double damping = 0.7; // Damping ratio
    
    // Control flags
    private AtomicBoolean isRunning = new AtomicBoolean(true);
    private AtomicBoolean isExecutingTrajectory = new AtomicBoolean(false);
    
    // Current motion container for stopping execution
    private IMotionContainer currentMotionContainer = null;
    
    // External torque threshold for human interaction
    private double forceThreshold = 10.0; // N
    private double torqueThreshold = 2.0; // Nm
    
    // ROS2 PC IP address - Same network as KUKA
    private String ros2PCIP = "192.170.10.1"; // ROS2 PC IP (same network)
    
    // Initial position for the robot
    private static final JointPosition INITIAL_POSITION = new JointPosition(0.0, -0.7854, 0.0, 1.3962, 0.0, 0.6109, 0.0);
    
    // Position hold for compliant waiting
    private PositionHold positionHold;
    
    @Override
    public void initialize() {
        getLogger().info("=== INITIALIZATION START ===");
        
        // Load configuration first
        loadConfigurationFromDataXML();
        
        // Get robot device
        robot = (LBR) getContext().getDeviceFromType(LBR.class);
        
        // DIAGNOSTIC CHECK 1: Robot device status
        if (robot == null) {
            getLogger().error("DIAGNOSTIC ERROR: Robot device is NULL!");
            return;
        }
        
        getLogger().info("DIAGNOSTIC CHECK 1: Robot device acquired successfully");
        getLogger().info("Robot name: " + robot.getName());
        getLogger().info("Robot ready to move: " + robot.isReadyToMove());
        getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
        
        getLogger().info("Initializing Cartesian Impedance Controller...");
        
        // Setup server socket for receiving trajectory commands
        try {
            trajectoryServerSocket = new ServerSocket(30002);
            getLogger().info("DIAGNOSTIC CHECK 2: Trajectory server started on port 30002");
        } catch (IOException e) {
            getLogger().error("DIAGNOSTIC ERROR: Could not start trajectory server socket: " + e.getMessage());
        }
        
        // Setup client socket for sending torque data
        try {
            torqueDataSocket = new Socket(ros2PCIP, 30003);
            torqueDataOut = new PrintWriter(torqueDataSocket.getOutputStream(), true);
            getLogger().info("DIAGNOSTIC CHECK 3: Connected to ROS2 PC for torque data transmission");
        } catch (IOException e) {
            getLogger().error("DIAGNOSTIC ERROR: Could not connect to ROS2 PC for torque data: " + e.getMessage());
        }
        
        getLogger().info("=== INITIALIZATION COMPLETE ===");
    }

    @Override
    public void run() {
        getLogger().info("=== RUN METHOD START ===");
        getLogger().info("Starting Cartesian Impedance Controller...");
        
        // DIAGNOSTIC CHECK 4: Robot status before initial motion
        getLogger().info("DIAGNOSTIC CHECK 4: Robot status before initial motion");
        getLogger().info("Robot ready to move: " + robot.isReadyToMove());
        getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
        getLogger().info("Current joint position: " + robot.getCurrentJointPosition());
        
        // DIAGNOSTIC TEST: Run basic robot functionality test
        testRobotFunctionality();
        
        // Move to initial position first
        getLogger().info("DIAGNOSTIC CHECK 5: Attempting to move to initial position...");
        try {
            robot.move(ptp(INITIAL_POSITION).setJointVelocityRel(0.2));
            getLogger().info("DIAGNOSTIC CHECK 5: Initial position motion completed successfully");
        } catch (Exception e) {
            getLogger().error("DIAGNOSTIC ERROR: Failed to move to initial position: " + e.getMessage());
            e.printStackTrace();
        }
        
        // Setup PositionHold with impedance control for compliant waiting
        getLogger().info("DIAGNOSTIC CHECK 6: Setting up PositionHold with impedance control...");
        try {
            JointImpedanceControlMode impedanceMode = new JointImpedanceControlMode(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
            impedanceMode.setStiffnessForAllJoints(0.0); // Very low stiffness for compliance
            getLogger().info("DIAGNOSTIC CHECK 6: JointImpedanceControlMode created successfully");
            
            positionHold = new PositionHold(impedanceMode, -1, java.util.concurrent.TimeUnit.MINUTES);
            getLogger().info("DIAGNOSTIC CHECK 6: PositionHold created successfully");
            
            // Start PositionHold to keep robot compliant while waiting
            getLogger().info("DIAGNOSTIC CHECK 7: Starting PositionHold - robot should now be compliant");
            currentMotionContainer = robot.moveAsync(positionHold);
            getLogger().info("DIAGNOSTIC CHECK 7: PositionHold started successfully");
            getLogger().info("PositionHold motion container: " + (currentMotionContainer != null ? "Created" : "NULL"));
            
        } catch (Exception e) {
            getLogger().error("DIAGNOSTIC ERROR: Failed to setup PositionHold: " + e.getMessage());
            e.printStackTrace();
        }
        
        try {
            // DIAGNOSTIC CHECK 8: Robot status after PositionHold setup
            getLogger().info("DIAGNOSTIC CHECK 8: Robot status after PositionHold setup");
            getLogger().info("Robot ready to move: " + robot.isReadyToMove());
            getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
            getLogger().info("Current joint position: " + robot.getCurrentJointPosition());
            
            // Wait for ROS2 connection with timeout
            getLogger().info("DIAGNOSTIC CHECK 9: Waiting for ROS2 trajectory connection (timeout: 30 seconds)...");
            trajectoryServerSocket.setSoTimeout(30000); // 30 second timeout
            trajectoryClientSocket = trajectoryServerSocket.accept();
            trajectoryIn = new BufferedReader(new InputStreamReader(trajectoryClientSocket.getInputStream()));
            trajectoryOut = new PrintWriter(trajectoryClientSocket.getOutputStream(), true);
            
            getLogger().info("DIAGNOSTIC CHECK 9: ROS2 trajectory client connected from: " + trajectoryClientSocket.getInetAddress());
            
            // Send ready signal
            trajectoryOut.println("READY");
            
            // Main control loop
            while (isRunning.get()) {
                String trajectoryLine = trajectoryIn.readLine();
                
                if (trajectoryLine == null) {
                    getLogger().warn("Trajectory client disconnected");
                    break;
                }
                
                // Parse trajectory command
                if (trajectoryLine.startsWith("TRAJECTORY:")) {
                    executeTrajectory(trajectoryLine.substring(11));
                } else if (trajectoryLine.equals("STOP")) {
                    stopExecution();
                } else if (trajectoryLine.equals("PAUSE")) {
                    pauseExecution();
                } else if (trajectoryLine.equals("RESUME")) {
                    resumeExecution();
                } else if (trajectoryLine.startsWith("IMPEDANCE:")) {
                    updateImpedanceParameters(trajectoryLine.substring(10));
                } else if (trajectoryLine.equals("GET_POSE")) {
                    sendCurrentPose();
                } else {
                    getLogger().warn("Unknown command: " + trajectoryLine);
                }
            }
            
        } catch (IOException e) {
            getLogger().error("DIAGNOSTIC ERROR: IO Exception in main loop: " + e.getMessage());
            getLogger().info("DIAGNOSTIC CHECK 10: Continuing with PositionHold for manual operation");
            
            // DIAGNOSTIC CHECK 11: Robot status during manual operation mode
            getLogger().info("DIAGNOSTIC CHECK 11: Robot status during manual operation mode");
            getLogger().info("Robot ready to move: " + robot.isReadyToMove());
            getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
            getLogger().info("PositionHold active: " + (currentMotionContainer != null && !currentMotionContainer.isFinished()));
            
            // Keep PositionHold active for manual operation
            while (isRunning.get()) {
                ThreadUtil.milliSleep(1000); // Check every second
            }
        } finally {
            getLogger().info("=== RUN METHOD ENDING - CLEANUP STARTING ===");
            cleanup();
        }
    }
    
    private void executeTrajectory(String trajectoryData) {
        try {
            // Parse trajectory points
            String[] points = trajectoryData.split(";");
            List<Frame> trajectoryFrames = new ArrayList<Frame>();
            
            for (String point : points) {
                String[] coords = point.trim().split(",");
                if (coords.length >= 6) {
                    double x = Double.parseDouble(coords[0]);
                    double y = Double.parseDouble(coords[1]);
                    double z = Double.parseDouble(coords[2]);
                    double alpha = Double.parseDouble(coords[3]);
                    double beta = Double.parseDouble(coords[4]);
                    double gamma = Double.parseDouble(coords[5]);
                    
                    Frame frame = new Frame(x, y, z, alpha, beta, gamma);
                    trajectoryFrames.add(frame);
                }
            }
            
            if (!trajectoryFrames.isEmpty()) {
                getLogger().info("Executing trajectory with " + trajectoryFrames.size() + " points");
                executeTrajectoryPoints(trajectoryFrames);
            }
            
        } catch (NumberFormatException e) {
            getLogger().error("Error parsing trajectory data: " + e.getMessage());
        }
    }
    
    // DIAGNOSTIC METHOD: Test basic robot functionality
    private void testRobotFunctionality() {
        getLogger().info("=== ROBOT FUNCTIONALITY TEST ===");
        
        try {
            // Test 1: Check robot status
            getLogger().info("TEST 1: Robot status check");
            getLogger().info("Robot name: " + robot.getName());
            getLogger().info("Robot ready to move: " + robot.isReadyToMove());
            getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
            getLogger().info("Current joint position: " + robot.getCurrentJointPosition());
            getLogger().info("Current cartesian position: " + robot.getCurrentCartesianPosition(robot.getFlange()));
            
            // Test 2: Try a simple PTP motion
            getLogger().info("TEST 2: Simple PTP motion test");
            JointPosition currentPos = robot.getCurrentJointPosition();
            getLogger().info("Current position: " + currentPos);
            
            // Move slightly from current position
            JointPosition testPos = new JointPosition(
                currentPos.get(0) + 0.1,
                currentPos.get(1),
                currentPos.get(2),
                currentPos.get(3),
                currentPos.get(4),
                currentPos.get(5),
                currentPos.get(6)
            );
            
            getLogger().info("Test position: " + testPos);
            robot.move(ptp(testPos).setJointVelocityRel(0.1));
            getLogger().info("TEST 2: PTP motion completed successfully");
            
            // Test 3: Check if robot can be moved manually
            getLogger().info("TEST 3: Manual movement capability");
            getLogger().info("Robot should now be in a state where manual movement is possible");
            
        } catch (Exception e) {
            getLogger().error("ROBOT FUNCTIONALITY TEST FAILED: " + e.getMessage());
            e.printStackTrace();
        }
        
        getLogger().info("=== ROBOT FUNCTIONALITY TEST COMPLETE ===");
    }
    
    private void loadConfigurationFromDataXML() {
        try {
            // Load ROS2 communication settings
            ros2PCIP = getApplicationData().getProcessData("ros2_pc_ip").getValue().toString();
            int trajectoryPort = Integer.parseInt(getApplicationData().getProcessData("trajectory_server_port").getValue().toString());
            int torquePort = Integer.parseInt(getApplicationData().getProcessData("torque_data_port").getValue().toString());
        
            // Load impedance parameters
            stiffnessX = Double.parseDouble(getApplicationData().getProcessData("stiffness_x").getValue().toString());
            stiffnessY = Double.parseDouble(getApplicationData().getProcessData("stiffness_y").getValue().toString());
            stiffnessZ = Double.parseDouble(getApplicationData().getProcessData("stiffness_z").getValue().toString());
            stiffnessRot = Double.parseDouble(getApplicationData().getProcessData("stiffness_rot").getValue().toString());
            damping = Double.parseDouble(getApplicationData().getProcessData("damping_ratio").getValue().toString());
        
            // Load interaction thresholds
            forceThreshold = Double.parseDouble(getApplicationData().getProcessData("force_threshold").getValue().toString());
            torqueThreshold = Double.parseDouble(getApplicationData().getProcessData("torque_threshold").getValue().toString());
        
            // Load control flags
            boolean enableHumanInteraction = Boolean.parseBoolean(getApplicationData().getProcessData("enable_human_interaction").getValue().toString());
            boolean sendTorqueData = Boolean.parseBoolean(getApplicationData().getProcessData("send_torque_data").getValue().toString());
        
            getLogger().info("Configuration loaded from data.xml");
            getLogger().info("ROS2 PC IP: " + ros2PCIP);
            getLogger().info("Stiffness: X=" + stiffnessX + ", Y=" + stiffnessY + ", Z=" + stiffnessZ);
        
        } catch (Exception e) {
            getLogger().error("Error loading configuration: " + e.getMessage());
        }
    }
    
    private void executeTrajectoryPoints(List<Frame> trajectoryFrames) {
        getLogger().info("=== TRAJECTORY EXECUTION START ===");
        isExecutingTrajectory.set(true);
        
        try {
            // DIAGNOSTIC CHECK 12: Robot status before trajectory execution
            getLogger().info("DIAGNOSTIC CHECK 12: Robot status before trajectory execution");
            getLogger().info("Robot ready to move: " + robot.isReadyToMove());
            getLogger().info("Robot has active motion command: " + robot.hasActiveMotionCommand());
            getLogger().info("Current motion container: " + (currentMotionContainer != null ? "Exists" : "NULL"));
            
            // Cancel PositionHold before starting trajectory
            if (currentMotionContainer != null && !currentMotionContainer.isFinished()) {
                getLogger().info("DIAGNOSTIC CHECK 13: Cancelling PositionHold for trajectory execution");
                currentMotionContainer.cancel();
                getLogger().info("DIAGNOSTIC CHECK 13: PositionHold cancelled successfully");
            } else {
                getLogger().info("DIAGNOSTIC CHECK 13: No active PositionHold to cancel");
            }
            
            // Setup Position Control Mode with compliance monitoring
            getLogger().info("DIAGNOSTIC CHECK 14: Setting up Position Control Mode");
            PositionControlMode positionMode = new PositionControlMode();
            getLogger().info("DIAGNOSTIC CHECK 14: Position Control Mode created successfully");
            
            // Configure position control parameters for more compliant behavior
            // Note: In standard KUKA Sunrise Workbench, we use position control
            // but monitor external forces to simulate impedance-like behavior
            
            getLogger().info("DIAGNOSTIC CHECK 14: Position Control Mode configured with force monitoring");
            getLogger().info("Stiffness parameters (for reference): X=" + stiffnessX + ", Y=" + stiffnessY + ", Z=" + stiffnessZ + ", Rot=" + stiffnessRot);
            getLogger().info("Damping: " + damping);
            
            // Execute trajectory point by point
            for (int i = 0; i < trajectoryFrames.size(); i++) {
                Frame targetFrame = trajectoryFrames.get(i);
                
                if (!isExecutingTrajectory.get()) {
                    getLogger().info("DIAGNOSTIC CHECK 15: Trajectory execution stopped by user");
                    break;
                }
                
                getLogger().info("DIAGNOSTIC CHECK 15: Executing trajectory point " + (i+1) + "/" + trajectoryFrames.size());
                getLogger().info("Target frame: " + targetFrame);
                getLogger().info("Robot ready to move: " + robot.isReadyToMove());
                
                // Move to target with position control
                try {
                    LIN motion = new LIN(targetFrame);
                    motion.setMode(positionMode);
                    getLogger().info("DIAGNOSTIC CHECK 15: LIN motion created successfully");
                    
                    currentMotionContainer = robot.moveAsync(motion);
                    getLogger().info("DIAGNOSTIC CHECK 15: Motion started successfully");
                    
                } catch (Exception e) {
                    getLogger().error("DIAGNOSTIC ERROR: Failed to execute motion: " + e.getMessage());
                    e.printStackTrace();
                    break;
                }
                
                // Monitor execution and external forces
                while (!currentMotionContainer.isFinished() && isExecutingTrajectory.get()) {
                    // Get external force/torque data
                    ForceSensorData forceData = robot.getExternalForceTorque(robot.getFlange());
                    
                    // Send torque data to ROS2
                    sendTorqueData(forceData);
                    
                    // Check for human interaction (external forces)
                    if (detectHumanInteraction(forceData)) {
                        getLogger().info("Human interaction detected - allowing trajectory modification");
                        // Here you can implement trajectory modification logic
                        // For now, we just log the interaction
                        
                        // Optional: Pause execution when significant force is detected
                        // This simulates impedance-like behavior
                        if (getForceMagnitude(forceData) > forceThreshold * 2) {
                            getLogger().warn("High force detected - considering pause for safety");
                        }
                    }
                    
                    ThreadUtil.milliSleep(10); // 100 Hz monitoring
                }
                
                // Send completion status
                trajectoryOut.println("POINT_COMPLETE");
            }
            
            trajectoryOut.println("TRAJECTORY_COMPLETE");
            getLogger().info("Trajectory execution completed");
            
            // Restart PositionHold after trajectory completion
            getLogger().info("Restarting PositionHold for compliant waiting");
            currentMotionContainer = robot.moveAsync(positionHold);
            
        } catch (Exception e) {
            getLogger().error("Error during trajectory execution: " + e.getMessage());
            trajectoryOut.println("ERROR:" + e.getMessage());
            
            // Restart PositionHold even if trajectory failed
            getLogger().info("Restarting PositionHold after trajectory error");
            currentMotionContainer = robot.moveAsync(positionHold);
        } finally {
            isExecutingTrajectory.set(false);
        }
    }
    
    private void sendTorqueData(ForceSensorData forceData) {
        try {
            // Format: timestamp,fx,fy,fz,tx,ty,tz
            String torqueMsg = String.format(Locale.US, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                System.currentTimeMillis(),
                forceData.getForce().getX(), forceData.getForce().getY(), forceData.getForce().getZ(),
                forceData.getTorque().getX(), forceData.getTorque().getY(), forceData.getTorque().getZ());
            
            if (torqueDataOut != null) {
                torqueDataOut.println(torqueMsg);
            }
        } catch (Exception e) {
            getLogger().error("Error sending torque data: " + e.getMessage());
        }
    }
    
    private boolean detectHumanInteraction(ForceSensorData forceData) {
        double forceMagnitude = getForceMagnitude(forceData);
        double torqueMagnitude = getTorqueMagnitude(forceData);
        
        return forceMagnitude > forceThreshold || torqueMagnitude > torqueThreshold;
    }
    
    private double getForceMagnitude(ForceSensorData forceData) {
        return Math.sqrt(
            Math.pow(forceData.getForce().getX(), 2) +
            Math.pow(forceData.getForce().getY(), 2) +
            Math.pow(forceData.getForce().getZ(), 2)
        );
    }
    
    private double getTorqueMagnitude(ForceSensorData forceData) {
        return Math.sqrt(
            Math.pow(forceData.getTorque().getX(), 2) +
            Math.pow(forceData.getTorque().getY(), 2) +
            Math.pow(forceData.getTorque().getZ(), 2)
        );
    }
    
    private void stopExecution() {
        isExecutingTrajectory.set(false);
        
        // Cancel current motion if active
        if (currentMotionContainer != null && !currentMotionContainer.isFinished()) {
            currentMotionContainer.cancel();
            getLogger().info("Current motion cancelled");
        }
        
        getLogger().info("Execution stopped");
        trajectoryOut.println("STOPPED");
    }
    
    private void pauseExecution() {
        isExecutingTrajectory.set(false);
        getLogger().info("Execution paused");
        trajectoryOut.println("PAUSED");
    }
    
    private void resumeExecution() {
        isExecutingTrajectory.set(true);
        getLogger().info("Execution resumed");
        trajectoryOut.println("RESUMED");
    }
    
    private void sendCurrentPose() {
        try {
            // Get current cartesian position
            Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
            
            // Format: POSE:x,y,z,alpha,beta,gamma
            String poseResponse = String.format("POSE:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                currentFrame.getX(), currentFrame.getY(), currentFrame.getZ(),
                currentFrame.getAlphaRad(), currentFrame.getBetaRad(), currentFrame.getGammaRad());
            
            trajectoryOut.println(poseResponse);
            getLogger().info("Sent current pose: " + poseResponse);
            
        } catch (Exception e) {
            getLogger().error("Error sending current pose: " + e.getMessage());
            trajectoryOut.println("ERROR:POSE_RETRIEVAL_FAILED");
        }
    }
    
    private void updateImpedanceParameters(String parameters) {
        try {
            String[] params = parameters.split(",");
            if (params.length >= 4) {
                stiffnessX = Double.parseDouble(params[0]);
                stiffnessY = Double.parseDouble(params[1]);
                stiffnessZ = Double.parseDouble(params[2]);
                damping = Double.parseDouble(params[3]);
                
                getLogger().info("Impedance parameters updated: Kx=" + stiffnessX + 
                               ", Ky=" + stiffnessY + ", Kz=" + stiffnessZ + 
                               ", Damping=" + damping);
                trajectoryOut.println("IMPEDANCE_UPDATED");
            }
        } catch (NumberFormatException e) {
            getLogger().error("Error parsing impedance parameters: " + e.getMessage());
        }
    }
    
    private void cleanup() {
        isRunning.set(false);
        isExecutingTrajectory.set(false);
        
        // Cancel any active motion
        if (currentMotionContainer != null && !currentMotionContainer.isFinished()) {
            currentMotionContainer.cancel();
            getLogger().info("Active motion cancelled during cleanup");
        }
        
        try {
            if (trajectoryIn != null) trajectoryIn.close();
            if (trajectoryOut != null) trajectoryOut.close();
            if (trajectoryClientSocket != null) trajectoryClientSocket.close();
            if (trajectoryServerSocket != null) trajectoryServerSocket.close();
            if (torqueDataOut != null) torqueDataOut.close();
            if (torqueDataSocket != null) torqueDataSocket.close();
        } catch (IOException e) {
            getLogger().error("Error during cleanup: " + e.getMessage());
        }
        
        getLogger().info("Cleanup completed");
    }
    
    @Override
    public void dispose() {
        cleanup();
        super.dispose();
    }
}