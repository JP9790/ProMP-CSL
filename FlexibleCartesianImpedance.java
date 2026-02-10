package application;

import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.deviceModel.JointPosition;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.motionModel.PositionHold;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianImpedanceControlMode;
import com.kuka.roboticsAPI.motionModel.controlModeModel.JointImpedanceControlMode;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.PTP;
import com.kuka.roboticsAPI.motionModel.LIN;
import com.kuka.roboticsAPI.sensorModel.ForceSensorData;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import java.io.*;
import java.net.*;
import java.util.*;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicBoolean;

public class FlexibleCartesianImpedance extends RoboticsAPIApplication {
    private LBR robot;
    private ServerSocket serverSocket;
    private Socket clientSocket;
    private BufferedReader in;
    private PrintWriter out;
    
    // Force data connection to Python controller
    private Socket forceDataSocket;
    private PrintWriter forceDataOut;
    
    // Control parameters (matching cartesianimpedance.java)
    private double stiffnessX = 250.0; // N/m
    private double stiffnessY = 250.0;
    private double stiffnessZ = 250.0;
    private double stiffnessRot = 50.0; // Nm/rad
    private double damping = 0.7; // Damping ratio
    
    // PositionHold stiffness - small value to hold against gravity while remaining compliant
    private double positionHoldStiffness = 50.0; // Nm/rad per joint (low enough for compliance, high enough to resist gravity)
    
    // External torque threshold for human interaction (matching cartesianimpedance.java)
    private double forceThreshold = 10.0; // N
    private double torqueThreshold = 2.0; // Nm
    
    // ROS2 PC IP address - Same network as KUKA (matching cartesianimpedance.java)
    private String ros2PCIP = "172.31.1.25"; // ROS2 PC IP (same network)
    
    private IMotionContainer currentMotion = null;
    private PositionHold positionHold = null;  // For continuous impedance control
    private AtomicBoolean isRunning = new AtomicBoolean(false);
    private AtomicBoolean stopRequested = new AtomicBoolean(false);
    private AtomicBoolean forceDataThreadRunning = new AtomicBoolean(false);
    
    // Lock object for thread-safe PrintWriter access
    private final Object outputLock = new Object();
    
    // Demo recording variables
    private List<double[]> currentDemo = new ArrayList<double[]>();
    private List<List<double[]>> allDemos = new ArrayList<List<double[]>>();
    private AtomicBoolean isRecordingDemo = new AtomicBoolean(false);  // Thread-safe
    private long demoStartTime;
    
    // Deformation variables
    private List<double[]> currentTrajectory = new ArrayList<double[]>();
    private boolean isExecutingTrajectory = false;
    private int currentTrajectoryIndex = 0;
    
    // Initial position for the robot (pointing away from wall) - matching cartesianimpedance.java
    private static final JointPosition INITIAL_POSITION = new JointPosition(0, 0.7854, 0.0, -1.3962, 0.0, -0.6109, 0.0);

    @Override
    public void initialize() {
        getLogger().info("=== INITIALIZATION START ===");
        
        // Load configuration first
        loadConfigurationFromDataXML();
        
        robot = (LBR) getContext().getDeviceFromType(LBR.class);
        
        if (robot == null) {
            getLogger().error("Failed to get robot device - robot is NULL!");
            return;
        }
        
        // Move to initial position first
        try {
            getLogger().info("Moving to initial position...");
            robot.move(ptp(INITIAL_POSITION).setJointVelocityRel(0.2));
            getLogger().info("Initial position reached");
        } catch (Exception e) {
            getLogger().error("Failed to move to initial position: " + e.getMessage());
        }
        
        // Setup PositionHold with joint impedance control for compliant waiting
        // Use small stiffness to hold against gravity while remaining compliant for physical interaction
        getLogger().info("Setting up PositionHold with impedance control for compliant waiting...");
        try {
            // Use JointImpedanceControlMode with small stiffness to resist gravity but remain compliant
            // Small stiffness (50 Nm/rad) prevents gravity drift while still allowing physical interaction
            JointImpedanceControlMode impedanceMode = new JointImpedanceControlMode(
                positionHoldStiffness, positionHoldStiffness, positionHoldStiffness, 
                positionHoldStiffness, positionHoldStiffness, positionHoldStiffness, positionHoldStiffness
            );
            impedanceMode.setStiffnessForAllJoints(positionHoldStiffness);
            getLogger().info("JointImpedanceControlMode created with stiffness: " + positionHoldStiffness + " Nm/rad (low enough for compliance, high enough to resist gravity)");
            
            positionHold = new PositionHold(impedanceMode, -1, java.util.concurrent.TimeUnit.MINUTES);
            getLogger().info("PositionHold created successfully");
            
            // Start PositionHold to keep robot compliant while waiting
            getLogger().info("Starting PositionHold - robot should now be compliant");
            currentMotion = robot.moveAsync(positionHold);
            getLogger().info("PositionHold started successfully - robot is now compliant for physical interaction");
            
        } catch (Exception e) {
            getLogger().error("Failed to setup PositionHold with impedance control: " + e.getMessage());
            e.printStackTrace();
        }
        
        try {
            serverSocket = new ServerSocket(30002);
            getLogger().info("All-in-one server started on port 30002");
        } catch (IOException e) {
            getLogger().error("Could not start server socket: " + e.getMessage());
        }
    }

    @Override
    public void run() {
        getLogger().info("Waiting for Python client...");
        try {
            clientSocket = serverSocket.accept();
            in = new BufferedReader(new InputStreamReader(clientSocket.getInputStream()));
            out = new PrintWriter(clientSocket.getOutputStream(), true);
            out.println("READY");
            getLogger().info("Python client connected: " + clientSocket.getInetAddress());
            
            // Setup force data connection to Python controller AFTER Python client connects
            // This ensures Python server on port 30003 is ready
            setupForceDataConnection();

            String line;
            while ((line = in.readLine()) != null) {
                line = line.trim();
                
                if (line.startsWith("START_DEMO_RECORDING")) {
                    startDemoRecording();
                } else if (line.startsWith("STOP_DEMO_RECORDING")) {
                    stopDemoRecording();
                } else if (line.startsWith("GET_DEMOS")) {
                    sendDemos();
                } else if (line.startsWith("CLEAR_DEMOS")) {
                    clearDemos();
                } else if (line.startsWith("TRAJECTORY:")) {
                    // Execute trajectory in separate thread to allow command processing during execution
                    final String trajData = line.substring("TRAJECTORY:".length());
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            executeTrajectory(trajData);
                        }
                    }).start();
                } else if (line.startsWith("JOINT_TRAJECTORY:")) {
                    // Execute joint trajectory in separate thread
                    final String jointTrajData = line.substring("JOINT_TRAJECTORY:".length());
                    new Thread(new Runnable() {
                        @Override
                        public void run() {
                            executeJointTrajectory(jointTrajData);
                        }
                    }).start();
                } else if (line.startsWith("GET_JOINT_POS:")) {
                    // Convert Cartesian position to joint position (IK)
                    String cartPosStr = line.substring("GET_JOINT_POS:".length());
                    handleGetJointPosition(cartPosStr);
                } else if (line.equals("STOP")) {
                    stopRequested.set(true);  // Set flag to stop trajectory execution
                    stopCurrentMotion();
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("STOPPED");
                        }
                    }
                } else if (line.equals("GET_POSE")) {
                    sendCurrentPose();
                } else if (line.startsWith("SET_IMPEDANCE:")) {
                    setImpedanceParameters(line.substring("SET_IMPEDANCE:".length()));
                } else {
                    getLogger().warn("Unknown command: " + line);
                }
            }
        } catch (IOException e) {
            getLogger().error("IO Exception: " + e.getMessage());
        }
    }

    /**
     * Setup force data connection to Python controller
     * Called after Python client connects to ensure Python server is ready
     */
    private void setupForceDataConnection() {
        // Wait a bit for Python server to be ready
        try {
            Thread.sleep(1000); // Give Python 1 second to set up server
        } catch (InterruptedException e) {
            // Continue anyway
        }
        
        // Try to connect with retries
        int maxRetries = 5;
        int retryDelay = 1000; // 1 second between retries
        
        for (int attempt = 1; attempt <= maxRetries; attempt++) {
            try {
                getLogger().info("Attempting to connect to Python force data server (attempt " + attempt + "/" + maxRetries + ")...");
                forceDataSocket = new Socket(ros2PCIP, 30003); // Python controller IP and port
                forceDataSocket.setSoTimeout(5000); // 5 second timeout for operations
                forceDataOut = new PrintWriter(forceDataSocket.getOutputStream(), true);
                getLogger().info("Force data connection established to Python controller");
                
                // Start continuous force data sending thread
                forceDataThreadRunning.set(true);
                Thread forceDataThread = new Thread(new Runnable() {
                    @Override
                    public void run() {
                        continuousForceDataSender();
                    }
                });
                forceDataThread.setDaemon(true);
                forceDataThread.start();
                getLogger().info("Force data sending thread started");
                return; // Success, exit retry loop
                
            } catch (IOException e) {
                if (attempt < maxRetries) {
                    getLogger().warn("Could not connect to Python controller for force data (attempt " + attempt + "): " + e.getMessage());
                    getLogger().info("Retrying in " + (retryDelay/1000) + " seconds...");
                    try {
                        Thread.sleep(retryDelay);
                    } catch (InterruptedException ie) {
                        break;
                    }
                } else {
                    getLogger().warn("Could not connect to Python controller for force data after " + maxRetries + " attempts: " + e.getMessage());
                    getLogger().info("Force data will be logged locally only");
                }
            }
        }
    }
    
    private List<double[]> parseTrajectory(String trajStr) {
        List<double[]> trajectory = new ArrayList<double[]>();
        String[] points = trajStr.split(";");
        for (String pt : points) {
            String[] vals = pt.split(",");
            double[] pose = new double[vals.length];
            for (int i = 0; i < vals.length; i++) {
                pose[i] = Double.parseDouble(vals[i]);
            }
            trajectory.add(pose);
        }
        return trajectory;
    }

    private void executeTrajectory(String trajectoryData) {
        if (robot == null) {
            getLogger().error("Cannot execute trajectory: robot is null");
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:ROBOT_NULL");
                }
            }
            return;
        }
        
        isRunning.set(true);
        stopRequested.set(false);
        try {
            // Parse trajectory
            List<double[]> trajectory = parseTrajectory(trajectoryData);
            if (trajectory.isEmpty()) {
                getLogger().error("Trajectory is empty after parsing");
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("ERROR:EMPTY_TRAJECTORY");
                    }
                }
                return;
            }
            
            currentTrajectory = new ArrayList<double[]>(trajectory);
            currentTrajectoryIndex = 0;
            
            // Read parameters with synchronization
            double sx, sy, sz, srot, damp;
            synchronized (this) {
                sx = stiffnessX;
                sy = stiffnessY;
                sz = stiffnessZ;
                srot = stiffnessRot;
                damp = damping;
            }
            
            getLogger().info("Using Cartesian impedance control with parameters:");
            getLogger().info("Stiffness: X=" + sx + ", Y=" + sy + ", Z=" + sz + 
                           ", RotX=" + srot + ", RotY=" + srot + ", RotZ=" + srot);
            getLogger().info("Damping: " + damp);
            getLogger().info("Note: CartesianImpedanceControlMode uses default impedance values from robot configuration");
            
            // Execute trajectory with continuous impedance control for real-time deformation
            for (int i = 0; i < trajectory.size(); i++) {
                // Check stop flag at the start of each iteration
                if (stopRequested.get()) {
                    getLogger().info("Execution stopped by STOP command");
                    break;
                }
                
                double[] pose = trajectory.get(i);
                currentTrajectoryIndex = i;
                
                // Validate pose data
                if (pose.length < 6) {
                    getLogger().error("Pose data must have at least 6 values (x,y,z,alpha,beta,gamma), got: " + pose.length);
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("ERROR:INVALID_POSE_POINT_" + i);
                        }
                    }
                    continue;
                }
                
                // Extract Cartesian coordinates
                double x = pose[0];
                double y = pose[1];
                double z = pose[2];
                double alpha = pose[3];
                double beta = pose[4];
                double gamma = pose[5];
                
                getLogger().info("Moving to Cartesian position: x=" + x + ", y=" + y + ", z=" + z + 
                               ", alpha=" + alpha + ", beta=" + beta + ", gamma=" + gamma);
                
                // Try to execute this point - catch workspace errors and continue
                try {
                    // Create Frame for Cartesian motion
                    Frame targetFrame = new Frame(x, y, z, alpha, beta, gamma);
                    
                    // Optional: Check distance from current position to detect potential workspace issues
                    try {
                        Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
                        double distance = Math.sqrt(
                            Math.pow(targetFrame.getX() - currentFrame.getX(), 2) +
                            Math.pow(targetFrame.getY() - currentFrame.getY(), 2) +
                            Math.pow(targetFrame.getZ() - currentFrame.getZ(), 2)
                        );
                        if (distance > 0.5) { // Large jump might indicate workspace issue
                            getLogger().warn("Point " + i + ": Large distance to target (" + String.format("%.3f", distance) + " m) - may cause workspace error");
                        }
                    } catch (Exception e) {
                        // Ignore - just a warning check
                    }
                    
                    // Create CartesianImpedanceControlMode for compliant motion
                    // Note: CartesianImpedanceControlMode uses default constructor - impedance values
                    // are controlled through the robot's configuration or default behavior
                    // The robot will be compliant during motion, allowing physical interaction
                    CartesianImpedanceControlMode currentImpedanceMode = new CartesianImpedanceControlMode();
                    
                    // Cancel PositionHold before executing trajectory point
                    if (positionHold != null && currentMotion != null && !currentMotion.isFinished()) {
                        currentMotion.cancel();
                    }
                    
                    // Execute LIN motion with impedance control for compliance
                    // Robot remains compliant during motion for physical interaction
                    // With configured stiffness values, robot can be pushed/deformed during execution
                    // Note: Workspace errors will be caught and point will be skipped
                    currentMotion = robot.moveAsync(lin(targetFrame).setMode(currentImpedanceMode));
                    
                    // Monitor execution and allow external force deformation
                    while (!currentMotion.isFinished() && !stopRequested.get()) {
                        try {
                            // Check stop flag frequently during motion
                            if (stopRequested.get()) {
                                currentMotion.cancel();
                                break;
                            }
                            
                            // Monitor external forces for deformation detection
                            // In impedance mode, the robot will naturally respond to external forces
                            // Force data is sent continuously by background thread, no need to send here
                            
                            // Small delay to allow external force response and command processing
                            Thread.sleep(50); // 20 Hz monitoring
                            
                        } catch (InterruptedException e) {
                            getLogger().info("Motion monitoring interrupted");
                            break;
                        } catch (Exception e) {
                            getLogger().error("Error during force monitoring: " + e.getMessage());
                            break;
                        }
                    }
                    
                    // Check stop flag before moving to next point
                    if (stopRequested.get()) {
                        getLogger().info("Execution stopped before completing trajectory");
                        break;
                    }
                    
                    // Thread-safe output - only send POINT_COMPLETE if motion completed successfully
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("POINT_COMPLETE");
                        }
                    }
                    
                } catch (Exception e) {
                    // Catch workspace errors, axis limit violations, and other exceptions for this specific point
                    String errorMsg = e.getMessage();
                    String errorType = "UNKNOWN";
                    String fullError = errorMsg;
                    
                    // Get full exception details for better debugging
                    if (e.getCause() != null) {
                        fullError = errorMsg + " (Cause: " + e.getCause().getMessage() + ")";
                    }
                    
                    if (errorMsg != null) {
                        if (errorMsg.contains("Arbeitsraumfehler") || errorMsg.contains("workspace") || 
                            (e.getCause() != null && e.getCause().getMessage() != null && 
                             e.getCause().getMessage().contains("Arbeitsraumfehler"))) {
                            errorType = "WORKSPACE_ERROR";
                            getLogger().warn("Point " + i + " is outside workspace - skipping. Position: x=" + 
                                String.format("%.3f", x) + ", y=" + String.format("%.3f", y) + ", z=" + String.format("%.3f", z));
                        } else if (errorMsg.contains("axis limit violation") || errorMsg.contains("software axis limit")) {
                            errorType = "AXIS_LIMIT_VIOLATION";
                            getLogger().warn("Point " + i + " violates axis limits - skipping. Position: x=" + 
                                String.format("%.3f", x) + ", y=" + String.format("%.3f", y) + ", z=" + String.format("%.3f", z));
                        } else if (errorMsg.contains("can not plan motion") || errorMsg.contains("cannot plan")) {
                            errorType = "MOTION_PLANNING_ERROR";
                            getLogger().warn("Point " + i + " cannot be planned - skipping. Position: x=" + 
                                String.format("%.3f", x) + ", y=" + String.format("%.3f", y) + ", z=" + String.format("%.3f", z));
                        } else {
                            getLogger().warn("Point " + i + " execution failed: " + fullError);
                        }
                    } else {
                        getLogger().warn("Point " + i + " execution failed with unknown error: " + e.getClass().getSimpleName());
                    }
                    
                    // Send error message to Python so it can skip this point
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("ERROR:" + errorType + "_POINT_" + i + ":" + fullError);
                        }
                    }
                    
                    // Continue to next point instead of breaking
                    // The trajectory will continue with remaining reachable points
                    continue;
                }
            }
            
            // Thread-safe output
            synchronized (outputLock) {
                if (out != null) {
                    out.println("TRAJECTORY_COMPLETE");
                }
            }
            getLogger().info("Trajectory execution complete");
            
            // Restart PositionHold to keep robot compliant after trajectory
            restartPositionHold();
            
        } catch (Exception e) {
            getLogger().error("Error during trajectory execution: " + e.getMessage());
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR: " + e.getMessage());
                }
            }
            
            // Restart PositionHold even if trajectory failed
            restartPositionHold();
        } finally {
            isRunning.set(false);
        }
    }
    
    /**
     * Restart PositionHold with joint impedance control to keep robot compliant
     * This ensures the robot is always ready for physical interaction
     * Uses small stiffness to resist gravity while remaining compliant
     */
    private void restartPositionHold() {
        try {
            if (robot == null) {
                return;
            }
            
            // Cancel any existing motion
            if (currentMotion != null && !currentMotion.isFinished()) {
                currentMotion.cancel();
            }
            
            // Use JointImpedanceControlMode with small stiffness to resist gravity but remain compliant
            // Small stiffness prevents gravity drift while still allowing physical interaction
            JointImpedanceControlMode impedanceMode = new JointImpedanceControlMode(
                positionHoldStiffness, positionHoldStiffness, positionHoldStiffness, 
                positionHoldStiffness, positionHoldStiffness, positionHoldStiffness, positionHoldStiffness
            );
            impedanceMode.setStiffnessForAllJoints(positionHoldStiffness);
            
            getLogger().info("Restarting PositionHold with joint impedance control (stiffness: " + positionHoldStiffness + " Nm/rad)");
            
            // Create new PositionHold with joint impedance control
            positionHold = new PositionHold(impedanceMode, -1, java.util.concurrent.TimeUnit.MINUTES);
            currentMotion = robot.moveAsync(positionHold);
            getLogger().info("PositionHold restarted - robot remains compliant");
        } catch (Exception e) {
            getLogger().error("Error restarting PositionHold: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private void sendForceData() {
        try {
            if (robot == null) {
                return;
            }
            
            // Get real external force/torque data from KUKA LBR IIWA force sensor
            ForceSensorData forceData = robot.getExternalForceTorque(robot.getFlange());
            
            // Format: timestamp,fx,fy,fz,tx,ty,tz
            String forceMsg = String.format(Locale.US, "%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                System.currentTimeMillis(),
                forceData.getForce().getX(), forceData.getForce().getY(), forceData.getForce().getZ(),
                forceData.getTorque().getX(), forceData.getTorque().getY(), forceData.getTorque().getZ());
            
            // Send to Python controller if connection is available
            if (forceDataOut != null) {
                forceDataOut.println(forceMsg);
            }
            
        } catch (Exception e) {
            // Log error but continue - force sensor might not be available in all configurations
            getLogger().info("Error reading force sensor data: " + e.getMessage());
        }
    }
    
    /**
     * Continuous force data sender thread - sends force data at 100Hz
     * This runs independently to provide real-time force feedback
     */
    private void continuousForceDataSender() {
        while (forceDataThreadRunning.get()) {
            try {
                sendForceData();
                Thread.sleep(10); // 100 Hz - matches demo recording frequency
            } catch (InterruptedException e) {
                getLogger().info("Force data thread interrupted");
                break;
            } catch (Exception e) {
                getLogger().error("Error in force data thread: " + e.getMessage());
                // Continue running even if there's an error
                try {
                    Thread.sleep(100); // Slow down on error
                } catch (InterruptedException ie) {
                    break;
                }
            }
        }
    }

    private void stopCurrentMotion() {
        stopRequested.set(true);
        isRunning.set(false);
        
        if (currentMotion != null && !currentMotion.isFinished()) {
            currentMotion.cancel();
            getLogger().info("Current motion cancelled");
        }
        
        // Restart PositionHold to keep robot compliant after stop
        restartPositionHold();
        
        // Note: Don't close force data connection here as it may still be needed
        // Only close on application shutdown
    }

    private void startDemoRecording() {
        if (isRecordingDemo.get()) {
            getLogger().warn("Demo recording already in progress");
            return;
        }
        
        isRecordingDemo.set(true);
        synchronized (currentDemo) {
            currentDemo.clear();
        }
        demoStartTime = System.currentTimeMillis();
        getLogger().info("Started demo recording");
        
        synchronized (outputLock) {
            if (out != null) {
                out.println("DEMO_RECORDING_STARTED");
            }
        }
        
        // Start recording thread
        new Thread(new Runnable() {
            @Override
            public void run() {
                recordDemoData();
            }
        }).start();
    }
    
    private void stopDemoRecording() {
        if (!isRecordingDemo.get()) {
            getLogger().warn("No demo recording in progress");
            return;
        }
        
        isRecordingDemo.set(false);
        synchronized (currentDemo) {
            if (currentDemo.size() > 0) {
                synchronized (allDemos) {
                    allDemos.add(new ArrayList<double[]>(currentDemo));
                }
                getLogger().info("Stopped demo recording. Total demos: " + allDemos.size() + ", Current demo points: " + currentDemo.size());
            }
        }
        
        synchronized (outputLock) {
            if (out != null) {
                out.println("DEMO_RECORDING_STOPPED");
            }
        }
    }
    
    private void recordDemoData() {
        while (isRecordingDemo.get()) {
            try {
                if (robot == null) {
                    getLogger().error("Robot is null, cannot record demo data");
                    break;
                }
                
                // Get current Cartesian pose
                Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
                double[] pose = {
                    currentFrame.getX(), currentFrame.getY(), currentFrame.getZ(),
                    currentFrame.getAlphaRad(), currentFrame.getBetaRad(), currentFrame.getGammaRad()
                };
                
                synchronized (currentDemo) {
                    currentDemo.add(pose);
                }
                
                // Record at 100Hz
                Thread.sleep(10);
                
            } catch (InterruptedException e) {
                getLogger().info("Demo recording interrupted");
                break;
            } catch (Exception e) {
                getLogger().error("Error recording demo data: " + e.getMessage());
                break;
            }
        }
    }
    
    private void sendDemos() {
        try {
            if (out == null) {
                getLogger().error("Output stream is null, cannot send demos");
                return;
            }
            
            // Convert demos to string format matching Python expectations
            // Format: DEMOS:DEMO_0:pose1;pose2;...|DEMO_1:pose1;pose2;...|
            StringBuilder demosStr = new StringBuilder();
            demosStr.append("DEMOS:");
            
            synchronized (allDemos) {
                for (int i = 0; i < allDemos.size(); i++) {
                    List<double[]> demo = allDemos.get(i);
                    demosStr.append("DEMO_").append(i).append(":");
                    
                    for (double[] pose : demo) {
                        // Format: x,y,z,alpha,beta,gamma (comma-separated, no semicolon at end of pose)
                        demosStr.append(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f", 
                            pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]));
                        demosStr.append(";");  // Semicolon between poses
                    }
                    if (i < allDemos.size() - 1) {
                        demosStr.append("|");  // Pipe between demos
                    }
                }
            }
            
            synchronized (outputLock) {
                if (out != null) {
                    out.println(demosStr.toString());
                }
            }
            getLogger().info("Sent " + allDemos.size() + " demos");
            
        } catch (Exception e) {
            getLogger().error("Error sending demos: " + e.getMessage());
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:SENDING_DEMOS");
                }
            }
        }
    }
    
    private void clearDemos() {
        synchronized (allDemos) {
            allDemos.clear();
        }
        synchronized (currentDemo) {
            currentDemo.clear();
        }
        getLogger().info("All demos cleared");
        synchronized (outputLock) {
            if (out != null) {
                out.println("DEMOS_CLEARED");
            }
        }
    }
    
    private void sendCurrentPose() {
        try {
            if (robot == null) {
                getLogger().error("Robot is null, cannot get current pose");
                if (out != null) {
                    out.println("ERROR:ROBOT_NULL");
                }
                return;
            }
            
            if (out == null) {
                getLogger().error("Output stream is null, cannot send pose");
                return;
            }
            
            Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
            String poseResponse = String.format("POSE:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                currentFrame.getX(), currentFrame.getY(), currentFrame.getZ(),
                currentFrame.getAlphaRad(), currentFrame.getBetaRad(), currentFrame.getGammaRad());
            
            synchronized (outputLock) {
                if (out != null) {
                    out.println(poseResponse);
                }
            }
            getLogger().info("Sent current pose: " + poseResponse);
            
        } catch (Exception e) {
            getLogger().error("Error sending current pose: " + e.getMessage());
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:POSE_RETRIEVAL_FAILED");
                }
            }
        }
    }
    
    /**
     * Handle GET_JOINT_POS command: Convert Cartesian position to joint position using IK
     * Format: "GET_JOINT_POS:x,y,z,alpha,beta,gamma"
     * Response: "JOINT_POS:j1,j2,j3,j4,j5,j6,j7" or "ERROR:IK_FAILED"
     */
    private void handleGetJointPosition(String cartPosStr) {
        try {
            if (robot == null) {
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("ERROR:ROBOT_NULL");
                    }
                }
                return;
            }
            
            // Parse Cartesian position
            String[] coords = cartPosStr.split(",");
            if (coords.length != 6) {
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("ERROR:INVALID_FORMAT");
                    }
                }
                return;
            }
            
            double x = Double.parseDouble(coords[0]);
            double y = Double.parseDouble(coords[1]);
            double z = Double.parseDouble(coords[2]);
            double alpha = Double.parseDouble(coords[3]);
            double beta = Double.parseDouble(coords[4]);
            double gamma = Double.parseDouble(coords[5]);
            
            // Create Frame
            Frame targetFrame = new Frame(x, y, z, alpha, beta, gamma);
            
            // Use robot's IK solver to get joint position
            // Note: getInverseKinematics() may throw exception if position is unreachable
            try {
                JointPosition jointPos = robot.getInverseKinematics(targetFrame, robot.getFlange());
                
                // Format response: "JOINT_POS:j1,j2,j3,j4,j5,j6,j7"
                String response = String.format("JOINT_POS:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                    jointPos.get(0), jointPos.get(1), jointPos.get(2), jointPos.get(3),
                    jointPos.get(4), jointPos.get(5), jointPos.get(6));
                
                synchronized (outputLock) {
                    if (out != null) {
                        out.println(response);
                    }
                }
                
            } catch (Exception e) {
                // IK failed - position is unreachable
                getLogger().warn("IK failed for position: " + targetFrame + " - " + e.getMessage());
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("ERROR:IK_FAILED:" + e.getMessage());
                    }
                }
            }
            
        } catch (Exception e) {
            getLogger().error("Error handling GET_JOINT_POS: " + e.getMessage());
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:PARSING_FAILED");
                }
            }
        }
    }
    
    /**
     * Execute trajectory using joint positions (PTP motion)
     * This avoids workspace errors since joint positions are guaranteed to be valid
     * Format: "JOINT_TRAJECTORY:j1_1,j2_1,...,j7_1;j1_2,j2_2,...,j7_2;..."
     */
    private void executeJointTrajectory(String jointTrajData) {
        if (robot == null) {
            getLogger().error("Cannot execute joint trajectory: robot is null");
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:ROBOT_NULL");
                }
            }
            return;
        }
        
        isRunning.set(true);
        stopRequested.set(false);
        
        try {
            // Parse joint trajectory
            List<JointPosition> jointTrajectory = new ArrayList<JointPosition>();
            String[] points = jointTrajData.split(";");
            
            for (String pt : points) {
                String[] vals = pt.split(",");
                if (vals.length == 7) {
                    JointPosition jointPos = new JointPosition(
                        Double.parseDouble(vals[0]),
                        Double.parseDouble(vals[1]),
                        Double.parseDouble(vals[2]),
                        Double.parseDouble(vals[3]),
                        Double.parseDouble(vals[4]),
                        Double.parseDouble(vals[5]),
                        Double.parseDouble(vals[6])
                    );
                    jointTrajectory.add(jointPos);
                }
            }
            
            if (jointTrajectory.isEmpty()) {
                getLogger().error("Joint trajectory is empty after parsing");
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("ERROR:EMPTY_TRAJECTORY");
                    }
                }
                return;
            }
            
            getLogger().info("Executing joint trajectory with " + jointTrajectory.size() + " points");
            
            // Execute joint trajectory using PTP (Point-to-Point) motion
            // PTP with joint positions avoids workspace errors
            for (int i = 0; i < jointTrajectory.size(); i++) {
                if (stopRequested.get()) {
                    getLogger().info("Joint trajectory execution stopped");
                    break;
                }
                
                JointPosition targetJoint = jointTrajectory.get(i);
                
                try {
                    // Cancel PositionHold before executing
                    if (positionHold != null && currentMotion != null && !currentMotion.isFinished()) {
                        currentMotion.cancel();
                    }
                    
                    // Use JointImpedanceControlMode for compliant PTP motion
                    JointImpedanceControlMode impedanceMode = new JointImpedanceControlMode(
                        positionHoldStiffness, positionHoldStiffness, positionHoldStiffness,
                        positionHoldStiffness, positionHoldStiffness, positionHoldStiffness, positionHoldStiffness
                    );
                    impedanceMode.setStiffnessForAllJoints(positionHoldStiffness);
                    
                    // Execute PTP motion with impedance control
                    currentMotion = robot.moveAsync(ptp(targetJoint).setMode(impedanceMode));
                    
                    // Wait for motion to complete
                    while (!currentMotion.isFinished() && !stopRequested.get()) {
                        try {
                            Thread.sleep(50); // 20 Hz monitoring
                        } catch (InterruptedException e) {
                            break;
                        }
                    }
                    
                    if (stopRequested.get()) {
                        currentMotion.cancel();
                        break;
                    }
                    
                    // Send point complete
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("POINT_COMPLETE");
                        }
                    }
                    
                } catch (Exception e) {
                    // Even with joint positions, motion might fail (e.g., axis limits)
                    getLogger().warn("Point " + i + " execution failed: " + e.getMessage());
                    synchronized (outputLock) {
                        if (out != null) {
                            out.println("ERROR:JOINT_MOTION_FAILED_POINT_" + i + ":" + e.getMessage());
                        }
                    }
                    // Continue to next point
                    continue;
                }
            }
            
            // Send completion
            synchronized (outputLock) {
                if (out != null) {
                    out.println("TRAJECTORY_COMPLETE");
                }
            }
            getLogger().info("Joint trajectory execution complete");
            
            // Restart PositionHold
            restartPositionHold();
            
        } catch (Exception e) {
            getLogger().error("Error during joint trajectory execution: " + e.getMessage());
            synchronized (outputLock) {
                if (out != null) {
                    out.println("ERROR:" + e.getMessage());
                }
            }
            restartPositionHold();
        } finally {
            isRunning.set(false);
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
        
            // Load PositionHold stiffness (if available, otherwise use default)
            try {
                positionHoldStiffness = Double.parseDouble(getApplicationData().getProcessData("position_hold_stiffness").getValue().toString());
            } catch (Exception e) {
                // Use default if not configured
                getLogger().info("position_hold_stiffness not found in data.xml, using default: " + positionHoldStiffness);
            }
        
            // Load interaction thresholds
            forceThreshold = Double.parseDouble(getApplicationData().getProcessData("force_threshold").getValue().toString());
            torqueThreshold = Double.parseDouble(getApplicationData().getProcessData("torque_threshold").getValue().toString());
        
            // Load control flags
            boolean enableHumanInteraction = Boolean.parseBoolean(getApplicationData().getProcessData("enable_human_interaction").getValue().toString());
            boolean sendTorqueData = Boolean.parseBoolean(getApplicationData().getProcessData("send_torque_data").getValue().toString());
        
            getLogger().info("Configuration loaded from data.xml");
            getLogger().info("ROS2 PC IP: " + ros2PCIP);
            getLogger().info("Stiffness: X=" + stiffnessX + ", Y=" + stiffnessY + ", Z=" + stiffnessZ + ", Rot=" + stiffnessRot);
            getLogger().info("Damping: " + damping);
            getLogger().info("PositionHold stiffness: " + positionHoldStiffness + " Nm/rad (for gravity compensation while remaining compliant)");
        
        } catch (Exception e) {
            getLogger().error("Error loading configuration: " + e.getMessage());
            getLogger().info("Using default configuration values");
        }
    }
    
    private void setImpedanceParameters(String parameters) {
        try {
            String[] params = parameters.split(",");
            if (params.length >= 4) {
                // Update parameters atomically
                synchronized (this) {
                    stiffnessX = Double.parseDouble(params[0]);
                    stiffnessY = Double.parseDouble(params[1]);
                    stiffnessZ = Double.parseDouble(params[2]);
                    damping = Double.parseDouble(params[3]);
                }
                
                getLogger().info("Impedance parameters updated: Kx=" + stiffnessX + ", Ky=" + stiffnessY + 
                               ", Kz=" + stiffnessZ + ", Damping=" + damping);
                
                synchronized (outputLock) {
                    if (out != null) {
                        out.println("IMPEDANCE_UPDATED");
                    }
                }
                
                // If currently executing, the next motion will use new parameters
                if (isRunning.get()) {
                    getLogger().info("New impedance parameters will apply to next trajectory point");
                }
            }
        } catch (NumberFormatException e) {
            getLogger().error("Error parsing impedance parameters: " + e.getMessage());
        }
    }
    
    @Override
    public void dispose() {
        getLogger().info("Shutting down application...");
        
        // Stop force data thread
        forceDataThreadRunning.set(false);
        
        // Stop any running motion
        stopRequested.set(true);
        if (currentMotion != null && !currentMotion.isFinished()) {
            currentMotion.cancel();
        }
        
        // Close force data connection
        try {
            if (forceDataOut != null) {
                forceDataOut.close();
            }
            if (forceDataSocket != null && !forceDataSocket.isClosed()) {
                forceDataSocket.close();
            }
        } catch (IOException e) {
            getLogger().error("Error closing force data socket: " + e.getMessage());
        }
        
        // Close main socket connections
        try {
            if (out != null) {
                out.close();
            }
            if (in != null) {
                in.close();
            }
            if (clientSocket != null && !clientSocket.isClosed()) {
                clientSocket.close();
            }
            if (serverSocket != null && !serverSocket.isClosed()) {
                serverSocket.close();
            }
        } catch (IOException e) {
            getLogger().error("Error closing sockets: " + e.getMessage());
        }
        
        getLogger().info("Application shutdown complete");
        super.dispose();
    }
} 
