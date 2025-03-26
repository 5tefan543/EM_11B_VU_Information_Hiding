import numpy as np
import matplotlib.pyplot as plt

def a(y_true, y_predicted_prob):
    print("\na) Build the confusion matrix for the decision threshold τ = 0.5.")
    threshold = 0.5
    y_predicted = [1 if prob >= threshold else 0 for prob in y_predicted_prob]
    print(f"y_predicted: {y_predicted}")

    # Step 2: Confusion matrix
    TN = sum([1 for t, p in zip(y_true, y_predicted) if t == 0 and p == 0])
    FN = sum([1 for t, p in zip(y_true, y_predicted) if t == 1 and p == 0])
    FP = sum([1 for t, p in zip(y_true, y_predicted) if t == 0 and p == 1])
    TP = sum([1 for t, p in zip(y_true, y_predicted) if t == 1 and p == 1])

    # Step 3: Define the confusion matrix
    print(f"Confusion matrix:")
    print(f"TN: {TN}    FN: {FN}")
    print(f"FP: {FP}    TP: {TP}")

    return TN, FN, FP, TP

def b(TN, FN, FP, TP):
    print("\nb) Calculate the accuracy, precision, and recall metrics.")

    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) # if it was a stego object, how often was it right?
    recall = TP / (TP + FN) # if it was a stego object, how often was it detected?

    # Print metrics
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")

def compute_roc_curve(y_true, y_predicted_prob):
    # Sort predicted probabilities and corresponding true labels in descending order
    desc_score_indices = np.argsort(y_predicted_prob)[::-1]
    y_predicted_prob = np.array(y_predicted_prob)[desc_score_indices]
    y_true = np.array(y_true)[desc_score_indices]

    TPRs = [] # True Positive Rates
    FPRs = [] # False Positive Rates
    thresholds = np.unique(y_predicted_prob)[::-1]
    print(f"Thresholds for ROC curve: {thresholds}")

    for t in thresholds:
        y_pred = y_predicted_prob >= t
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        TP = np.sum((y_pred == 1) & (y_true == 1))

        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0 # if it was a stego object, how often was it detected? (Recall)
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0 # if it was a stego object, how often was it wrong?

        TPRs.append(TPR)
        FPRs.append(FPR)

    return FPRs, TPRs

def compute_auc(FPRs, TPRs):
    auc = 0.0
    for i in range(1, len(FPRs)):
        auc += (FPRs[i] - FPRs[i - 1]) * (TPRs[i] + TPRs[i - 1]) / 2 # trapezoidal rule for integration
    return auc

def plot_roc(FPRs, TPRs, auc):
    plt.plot(FPRs, TPRs, marker='o', label=f"ROC Curve (AUC = {auc:.3f})")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("roc_curve.png")
    plt.close()

def c(y_true, y_predicted_prob):
    print("\nc) Draw the ROC curve. Calculate ROC AUC, equal-error rate, and the probability of error PE.")

    # Compute ROC curve
    FPRs, TPRs = compute_roc_curve(y_true, y_predicted_prob)

    # Calculate AUC
    auc = compute_auc(FPRs, TPRs)

    # Plot ROC curve
    plot_roc(FPRs, TPRs, auc)

    # Calculate equal-error rate: the point where FPR = FNR
    FNRs = [1 - tpr for tpr in TPRs]
    FPRs_np = np.array(FPRs)
    FNRs_np = np.array(FNRs)
    eer_index = np.argmin(np.abs(FPRs_np - FNRs_np))
    eer_threshold_fpr = FPRs_np[eer_index]
    eer_threshold_fnr = FNRs_np[eer_index]
    eer = (eer_threshold_fpr + eer_threshold_fnr) / 2
    print(f"Equal-Error Rate (EER): {eer:.3f}")

    FNRs = [1 - tpr for tpr in TPRs]
    PEs = [(fpr + fnr) / 2 for fpr, fnr in zip(FPRs, FNRs)]
    PE = min(PEs)
    print(f"Probability of Error (PE): {PE:.3f}")

def main():
    print("Data batch X with labels y is passed to a detector that produces predictions ŷ.")
    y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1]
    y_predicted_prob = [0.4, 0.45, 0.55, 0.45, 0.4, 0.6, 0.55, 0.45, 0.6]
    print(f"y_true: {y_true}")
    print(f"y_predicted_prob: {y_predicted_prob}")

    TN, FN, FP, TP = a(y_true, y_predicted_prob)
    b(TN, FN, FP, TP)
    c(y_true, y_predicted_prob)

if __name__ == "__main__":
    main()