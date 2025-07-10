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
    private IMotionContainer currentMotion = null;
    private AtomicBoolean isRunning = new AtomicBoolean(false);
    private AtomicBoolean stopRequested = new AtomicBoolean(false);

    @Override
    public void initialize() {
        robot = (LBR) getContext().getDeviceFromType(LBR.class);
        try {
            serverSocket = new ServerSocket(30002);
            getLogger().info("Trajectory server started on port 30002");
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
                if (line.startsWith("TRAJECTORY:")) {
                    getLogger().info("Received new trajectory command");
                    if (isRunning.get()) {
                        stopCurrentMotion();
                    }
                    List<double[]> trajectory = parseTrajectory(line.substring("TRAJECTORY:".length()));
                    executeTrajectory(trajectory);
                } else if (line.equals("STOP")) {
                    getLogger().info("Received STOP command");
                    stopCurrentMotion();
                    out.println("STOPPED");
                } else {
                    getLogger().warn("Unknown command: " + line);
                }
            }
        } catch (IOException e) {
            getLogger().error("IO Exception: " + e.getMessage());
        }
    }

    private List<double[]> parseTrajectory(String trajStr) {
        List<double[]> trajectory = new ArrayList<>();
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

    private void executeTrajectory(List<double[]> trajectory) {
        isRunning.set(true);
        stopRequested.set(false);
        try {
            for (double[] pose : trajectory) {
                if (stopRequested.get()) {
                    getLogger().info("Execution stopped by STOP command");
                    break;
                }
                // Example: Move in joint space (replace with your own logic for Cartesian)
                currentMotion = robot.moveAsync(ptp(pose));
                currentMotion.await();
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

    private void stopCurrentMotion() {
        stopRequested.set(true);
        if (currentMotion != null && currentMotion.isActive()) {
            currentMotion.cancel();
        }
        isRunning.set(false);
    }
} 