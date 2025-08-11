# Scoring.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ====== 文件路径 ======
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "online_prediction_results", "online_prediction_details.csv")
save_img_path = os.path.join(base_dir, "online_prediction_results", "online_confusion_matrix.png")

# ====== 读取 CSV ======
df = pd.read_csv(csv_path)

# 移除 end 行
df = df[df["Track_ID"] != "end"]

# 转换数据类型
df["Track_ID"] = df["Track_ID"].astype(int)
df["Point_ID"] = df["Point_ID"].astype(int)
df["Predicted_Label"] = df["Predicted_Label"].astype(int)
df["True_Label"] = df["True_Label"].astype(int)

# ====== 统计通过样本数和未通过编号 ======
acc_num = 0
err_list = []
err_label=[]
total_num=0
for track_id, group in df.groupby("Track_ID"):
    total_points = len(group)
    correct_points = (group["Predicted_Label"] == group["True_Label"]).sum()
    accuracy = correct_points / total_points
    total_num+=1
    if accuracy >= 0.9:
        acc_num += 1
    else:
        err_list.append(track_id)
        err_label.append(group[group["Track_ID"] == track_id]["True_Label"])

# ====== 统计首次正确识别的平均点数 ======
first_correct_points = []
for track_id, group in df.groupby("Track_ID"):
    correct_indices = group[group["Predicted_Label"] == group["True_Label"]]["Point_ID"]
    if not correct_indices.empty:
        first_correct_points.append(correct_indices.iloc[0])

if first_correct_points:
    average_point = sum(first_correct_points) / len(first_correct_points)
else:
    average_point = None

# ====== 输出统计结果 ======
print(f"总样本为: {total_num}")
print(f"通过样本数量 (acc_num): {acc_num}")
print(f"通过率为: {acc_num/total_num*100}%")
print(f"未通过样本编号 (err_list): {err_list}")
print(f"对应的label为: {err_label}")
print(f"首次正确识别的平均点数 (average_point): {average_point}")

# ====== 绘制混淆矩阵 ======
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(df["True_Label"], df["Predicted_Label"])
labels = sorted(list(set(df["True_Label"]) | set(df["Predicted_Label"])))

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(save_img_path)
plt.close()

print(f"混淆矩阵已保存到: {save_img_path}")
