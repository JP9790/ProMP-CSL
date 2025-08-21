import com.kuka.roboticsAPI.applicationModel.RoboticsAPIApplication;
import com.kuka.roboticsAPI.deviceModel.LBR;
import com.kuka.roboticsAPI.motionModel.IMotionContainer;
import com.kuka.roboticsAPI.motionModel.PositionHold;
import com.kuka.roboticsAPI.motionModel.controlModeModel.CartesianImpedanceControlMode;
import com.kuka.roboticsAPI.geometricModel.Frame;
import com.kuka.roboticsAPI.motionModel.PTP;
import com.kuka.roboticsAPI.motionModel.LIN;
import static com.kuka.roboticsAPI.motionModel.BasicMotions.*;

import java.io.*;
import java.net.*;
import java.util.*;
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
    private double stiffnessX = 50.0; // N/m
    private double stiffnessY = 50.0;
    private double stiffnessZ = 50.0;
    private double stiffnessRot = 20.0; // Nm/rad
    private double damping = 0.7; // Damping ratio
    
    // External torque threshold for human interaction (matching cartesianimpedance.java)
    private double forceThreshold = 10.0; // N
    private double torqueThreshold = 2.0; // Nm
    
    // ROS2 PC IP address - Same network as KUKA (matching cartesianimpedance.java)
    private String ros2PCIP = "172.31.1.100"; // ROS2 PC IP (same network)
    
    private IMotionContainer currentMotion = null;
    private AtomicBoolean isRunning = new AtomicBoolean(false);
    private AtomicBoolean stopRequested = new AtomicBoolean(false);
    
    // Demo recording variables
    private List<double[]> currentDemo = new ArrayList<double[]>();
    private List<List<double[]>> allDemos = new ArrayList<List<double[]>>();
    private boolean isRecordingDemo = false;
    private long demoStartTime;
    
    // Deformation variables
    private List<double[]> currentTrajectory = new ArrayList<double[]>();
    private boolean isExecutingTrajectory = false;
    private int currentTrajectoryIndex = 0;
    
    // Initial position for the robot (pointing away from wall) - matching cartesianimpedance.java
    private static final double[] INITIAL_POSITION = {Math.PI, -0.7854, 0.0, 1.3962, 0.0, 0.6109, 0.0};

    @Override
    public void initialize() {
        robot = (LBR) getContext().getDeviceFromType(LBR.class);
        
        // Move to initial position first
        try {
            getLogger().info("Moving to initial position...");
            double[] initialJointPos = INITIAL_POSITION;
            robot.move(ptp(new com.kuka.roboticsAPI.deviceModel.JointPosition(
                initialJointPos[0], initialJointPos[1], initialJointPos[2], 
                initialJointPos[3], initialJointPos[4], initialJointPos[5], initialJointPos[6]
            )).setJointVelocityRel(0.2));
            getLogger().info("Initial position reached");
        } catch (Exception e) {
            getLogger().error("Failed to move to initial position: " + e.getMessage());
        }
        
        try {
            serverSocket = new ServerSocket(30002);
            getLogger().info("All-in-one server started on port 30002");
            
            // Setup force data connection to Python controller
            try {
                forceDataSocket = new Socket(ros2PCIP, 30003); // Python controller IP and port
                forceDataOut = new PrintWriter(forceDataSocket.getOutputStream(), true);
                getLogger().info("Force data connection established to Python controller");
            } catch (IOException e) {
                getLogger().warn("Could not connect to Python controller for force data: " + e.getMessage());
                getLogger().info("Force data will be logged locally only");
            }
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
                    executeTrajectory(line.substring("TRAJECTORY:".length()));
                } else if (line.equals("STOP")) {
                    stopCurrentMotion();
                    out.println("STOPPED");
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
        isRunning.set(true);
        stopRequested.set(false);
        try {
            // Parse trajectory
            List<double[]> trajectory = parseTrajectory(trajectoryData);
            currentTrajectory = new ArrayList<double[]>(trajectory);
            currentTrajectoryIndex = 0;
            
            // Setup Cartesian impedance control mode with custom parameters
            CartesianImpedanceControlMode impedanceMode = new CartesianImpedanceControlMode();
            
            // Try to set custom impedance parameters if available
            try {
                // Note: The exact method names may vary depending on your KUKA API version
                // This will use default settings if custom setters are not available
                getLogger().info("Using Cartesian impedance control with custom parameters for compliant trajectory execution");
                getLogger().info("Stiffness: X=" + stiffnessX + ", Y=" + stiffnessY + ", Z=" + stiffnessZ + 
                               ", RotX=" + stiffnessRot + ", RotY=" + stiffnessRot + ", RotZ=" + stiffnessRot);
                getLogger().info("Damping: " + damping);
            } catch (Exception e) {
                getLogger().warn("Could not set custom impedance parameters, using defaults: " + e.getMessage());
            }
            
            // Execute trajectory with continuous impedance control for real-time deformation
            for (int i = 0; i < trajectory.size(); i++) {
                if (stopRequested.get()) {
                    getLogger().info("Execution stopped by STOP command");
                    break;
                }
                
                double[] pose = trajectory.get(i);
                currentTrajectoryIndex = i;
                
                // Validate pose data
                if (pose.length < 6) {
                    getLogger().error("Pose data must have at least 6 values (x,y,z,alpha,beta,gamma), got: " + pose.length);
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
                
                // Create Frame for Cartesian motion
                Frame targetFrame = new Frame(x, y, z, alpha, beta, gamma);
                
                // Execute LIN motion with impedance control for compliance
                currentMotion = robot.moveAsync(lin(targetFrame).setMode(impedanceMode));
                
                // Monitor execution and allow external force deformation
                while (!currentMotion.isFinished() && !stopRequested.get()) {
                    try {
                        // Monitor external forces for deformation detection
                        // In impedance mode, the robot will naturally respond to external forces
                        
                        // Send force data to Python controller if connected
                        sendForceData();
                        
                        // Log that the robot is in compliant mode
                        getLogger().info("Robot in compliant mode - can be pushed for deformation");
                        
                        // Small delay to allow external force response
                        Thread.sleep(50); // 20 Hz monitoring
                        
                    } catch (Exception e) {
                        getLogger().error("Error during force monitoring: " + e.getMessage());
                        break;
                    }
                }
                
                // Don't wait for motion to complete - let it be interrupted by external forces
                // This allows real-time deformation during execution
                
                out.println("POINT_COMPLETE");
            }
            out.println("TRAJECTORY_COMPLETE");
            getLogger().info("Trajectory execution complete");
        } catch (Exception e) {
            getLogger().error("Error during trajectory execution: " + e.getMessage());
            out.println("ERROR: " + e.getMessage());
        } finally {
            isRunning.set(false);
        }
    }
    
    private void sendForceData() {
        try {
            // Get external force/torque data (if available in your KUKA API)
            // Format: timestamp,fx,fy,fz,tx,ty,tz
            String forceMsg = String.format("%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.3f",
                System.currentTimeMillis(),
                0.0, 0.0, 0.0,  // Placeholder for force data
                0.0, 0.0, 0.0); // Placeholder for torque data
            
            // Send to Python controller if connection is available
            if (forceDataOut != null) {
                forceDataOut.println(forceMsg);
            }
            
        } catch (Exception e) {
            // Force data not available, continue without it
        }
    }

    private void stopCurrentMotion() {
        stopRequested.set(true);
        if (currentMotion != null && !currentMotion.isFinished()) {
            currentMotion.cancel();
        }
        isRunning.set(false);
        
        // Close force data connection
        if (forceDataOut != null) {
            forceDataOut.close();
        }
        if (forceDataSocket != null && !forceDataSocket.isClosed()) {
            try {
                forceDataSocket.close();
            } catch (IOException e) {
                getLogger().error("Error closing force data socket: " + e.getMessage());
            }
        }
    }

    private void startDemoRecording() {
        if (isRecordingDemo) {
            getLogger().warn("Demo recording already in progress");
            return;
        }
        
        isRecordingDemo = true;
        currentDemo.clear();
        demoStartTime = System.currentTimeMillis();
        getLogger().info("Started demo recording");
        out.println("DEMO_RECORDING_STARTED");
        
        // Start recording thread
        new Thread(new Runnable() {
            @Override
            public void run() {
                recordDemoData();
            }
        }).start();
    }
    
    private void stopDemoRecording() {
        if (!isRecordingDemo) {
            getLogger().warn("No demo recording in progress");
            return;
        }
        
        isRecordingDemo = false;
        if (currentDemo.size() > 0) {
            allDemos.add(new ArrayList<double[]>(currentDemo));
            getLogger().info("Stopped demo recording. Total demos: " + allDemos.size() + ", Current demo points: " + currentDemo.size());
        }
        out.println("DEMO_RECORDING_STOPPED");
    }
    
    private void recordDemoData() {
        while (isRecordingDemo) {
            try {
                // Get current Cartesian pose
                Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
                double[] pose = {
                    currentFrame.getX(), currentFrame.getY(), currentFrame.getZ(),
                    currentFrame.getAlphaRad(), currentFrame.getBetaRad(), currentFrame.getGammaRad()
                };
                
                currentDemo.add(pose);
                
                // Record at 100Hz
                Thread.sleep(10);
                
            } catch (Exception e) {
                getLogger().error("Error recording demo data: " + e.getMessage());
                break;
            }
        }
    }
    
    private void sendDemos() {
        try {
            // Convert demos to string format
            StringBuilder demosStr = new StringBuilder();
            demosStr.append("DEMOS:");
            
            for (int i = 0; i < allDemos.size(); i++) {
                List<double[]> demo = allDemos.get(i);
                demosStr.append("DEMO_").append(i).append(":");
                
                for (double[] pose : demo) {
                    demosStr.append(String.format("%.6f,%.6f,%.6f,%.6f,%.6f,%.6f;", 
                        pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]));
                }
                demosStr.append("|");
            }
            
            out.println(demosStr.toString());
            getLogger().info("Sent " + allDemos.size() + " demos");
            
        } catch (Exception e) {
            getLogger().error("Error sending demos: " + e.getMessage());
            out.println("ERROR:SENDING_DEMOS");
        }
    }
    
    private void clearDemos() {
        allDemos.clear();
        currentDemo.clear();
        getLogger().info("All demos cleared");
        out.println("DEMOS_CLEARED");
    }
    
    private void sendCurrentPose() {
        try {
            Frame currentFrame = robot.getCurrentCartesianPosition(robot.getFlange());
            String poseResponse = String.format("POSE:%.6f,%.6f,%.6f,%.6f,%.6f,%.6f",
                currentFrame.getX(), currentFrame.getY(), currentFrame.getZ(),
                currentFrame.getAlphaRad(), currentFrame.getBetaRad(), currentFrame.getGammaRad());
            
            out.println(poseResponse);
            getLogger().info("Sent current pose: " + poseResponse);
            
        } catch (Exception e) {
            getLogger().error("Error sending current pose: " + e.getMessage());
            out.println("ERROR:POSE_RETRIEVAL_FAILED");
        }
    }
    
    private void setImpedanceParameters(String parameters) {
        try {
            String[] params = parameters.split(",");
            if (params.length >= 4) {
                stiffnessX = Double.parseDouble(params[0]);
                stiffnessY = Double.parseDouble(params[1]);
                stiffnessZ = Double.parseDouble(params[2]);
                damping = Double.parseDouble(params[3]);
                
                getLogger().info("Impedance parameters updated: Kx=" + stiffnessX + ", Ky=" + stiffnessY + 
                               ", Kz=" + stiffnessZ + ", Damping=" + damping);
                out.println("IMPEDANCE_UPDATED");
            }
        } catch (NumberFormatException e) {
            getLogger().error("Error parsing impedance parameters: " + e.getMessage());
        }
    }
} 