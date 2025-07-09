package application;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.motionModel.LIN;
import com.kuka.roboticsAPI.motionModel.controlModeModel.PositionControlMode;
import com.kuka.roboticsAPI.sensorModel.ForceSensorData;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.geometricModel.math.Transformation;
import com.kuka.common.ThreadUtil;

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
    private double stiffnessX = 2000.0; // N/m
    private double stiffnessY = 2000.0;
    private double stiffnessZ = 2000.0;
    private double stiffnessRot = 200.0; // Nm/rad
    private double damping = 0.7; // Damping ratio
    
    // Control flags
    private AtomicBoolean isRunning = new AtomicBoolean(true);
    private AtomicBoolean isExecutingTrajectory = new AtomicBoolean(false);
    
    // Current motion container for stopping execution
    private IMotionContainer currentMotionContainer = null;
    
    // External torque threshold for human interaction
    private double forceThreshold = 10.0; // N
    private double torqueThreshold = 2.0; // Nm
    
    // ROS2 PC IP address - CHANGE THIS TO YOUR ROS2 PC IP
    private String ros2PCIP = "192.170.10.1"; // Replace with your ROS2 PC IP
    
    @Override
    public void initialize() {
        // Load configuration first
        loadConfigurationFromDataXML();
        
        // Get robot device
        robot = (LBR) getContext().getDeviceFromType(LBR.class);
        
        getLogger().info("Initializing Cartesian Impedance Controller...");
        
        // Setup server socket for receiving trajectory commands
        try {
            trajectoryServerSocket = new ServerSocket(30002);
            getLogger().info("Trajectory server started on port 30002");
        } catch (IOException e) {
            getLogger().error("Could not start trajectory server socket: " + e.getMessage());
        }
        
        // Setup client socket for sending torque data
        try {
            torqueDataSocket = new Socket(ros2PCIP, 30003);
            torqueDataOut = new PrintWriter(torqueDataSocket.getOutputStream(), true);
            getLogger().info("Connected to ROS2 PC for torque data transmission");
        } catch (IOException e) {
            getLogger().error("Could not connect to ROS2 PC for torque data: " + e.getMessage());
        }
    }

    @Override
    public void run() {
        getLogger().info("Starting Cartesian Impedance Controller...");
        
        try {
            // Wait for ROS2 connection
            getLogger().info("Waiting for ROS2 trajectory connection...");
            trajectoryClientSocket = trajectoryServerSocket.accept();
            trajectoryIn = new BufferedReader(new InputStreamReader(trajectoryClientSocket.getInputStream()));
            trajectoryOut = new PrintWriter(trajectoryClientSocket.getOutputStream(), true);
            
            getLogger().info("ROS2 trajectory client connected from: " + trajectoryClientSocket.getInetAddress());
            
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
                } else {
                    getLogger().warn("Unknown command: " + trajectoryLine);
                }
            }
            
        } catch (IOException e) {
            getLogger().error("IO Exception in main loop: " + e.getMessage());
        } finally {
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
        isExecutingTrajectory.set(true);
        
        try {
            // Setup Position Control Mode with compliance monitoring
            PositionControlMode positionMode = new PositionControlMode();
            
            // Configure position control parameters for more compliant behavior
            // Note: In standard KUKA Sunrise Workbench, we use position control
            // but monitor external forces to simulate impedance-like behavior
            
            getLogger().info("Position Control Mode configured with force monitoring");
            getLogger().info("Stiffness parameters (for reference): X=" + stiffnessX + ", Y=" + stiffnessY + ", Z=" + stiffnessZ + ", Rot=" + stiffnessRot);
            getLogger().info("Damping: " + damping);
            
            // Execute trajectory point by point
            for (Frame targetFrame : trajectoryFrames) {
                if (!isExecutingTrajectory.get()) {
                    getLogger().info("Trajectory execution stopped");
                    break;
                }
                
                // Move to target with position control
                LIN motion = new LIN(targetFrame);
                motion.setMode(positionMode);
                
                currentMotionContainer = robot.moveAsync(motion);
                
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
            
        } catch (Exception e) {
            getLogger().error("Error during trajectory execution: " + e.getMessage());
            trajectoryOut.println("ERROR:" + e.getMessage());
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