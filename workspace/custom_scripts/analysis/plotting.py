import pandas as pd
import matplotlib.pyplot as plt
import os 

def plot_joint_torques(csv_file="torque_log.csv"):
    # Load CSV into a DataFrame

    if not os.path.exists(csv_file):
        print(f"csv_file not found at {csv_file}")
        return
    df = pd.read_csv(csv_file)

    # Step column
    steps = df["step"]

    # Plot each joint separately
    for col in df.columns[1:]:  # skip "step"
        plt.figure(figsize=(8,4))
        plt.plot(steps, df[col], label=col)
        plt.xlabel("Step")
        plt.ylabel("Torque (Nm)")
        plt.title(f"Torque vs Step for {col}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__=="__main__":
    file_path = os.path.join("custom_scripts", "logs", "ppo_factory", "csv_files")

    
    file_path = os.path.join(file_path, "dof_torques.csv")

    plot_joint_torques(file_path)