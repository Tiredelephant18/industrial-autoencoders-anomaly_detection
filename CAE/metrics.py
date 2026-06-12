import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc, fbeta_score,confusion_matrix
import os
from scipy.stats import ks_2samp
from scipy import stats

class Evaluation:

    def full_evaluation(self, labels, scores, save_path=None, exp_id=0):

        fpr, tpr, roc_thresholds   = roc_curve(labels, scores)
        precision_p, recall_p, pr_thresholds = precision_recall_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        pr_auc  = auc(recall_p, precision_p)

        def calc_metrics(thresh):
            preds = (scores >= thresh).astype(int)
            rec   = np.sum(preds[labels == 1]) / np.sum(labels) if np.sum(labels) > 0 else 0
            prec  = np.mean(labels[preds == 1]) if np.sum(preds) > 0 else 0
            f2    = fbeta_score(labels, preds, beta=2, zero_division=0)
            return rec, prec, f2


        target_recall = 0.98
        valid_idx = np.where(recall_p[:-1] >= target_recall)[0]
        thresh_1  = pr_thresholds[valid_idx[np.argmax(precision_p[valid_idx])]] if len(valid_idx) > 0 \
                    else pr_thresholds[np.argmax(recall_p[:-1])]
        rec_1, prec_1, f2_1 = calc_metrics(thresh_1)

    
        thresh_2 = roc_thresholds[np.argmax(tpr - fpr)]
        rec_2, prec_2, f2_2 = calc_metrics(thresh_2)


        f2_scores   = [fbeta_score(labels, (scores >= t).astype(int), beta=2, zero_division=0)
                    for t in pr_thresholds]
        best_idx    = np.argmax(f2_scores)
        thresh_3    = pr_thresholds[best_idx]
        rec_3, prec_3, f2_3 = calc_metrics(thresh_3)


        print(f"\n{'=' * 65}")
        print(f"  Сравнение методов  |  ROC-AUC: {roc_auc:.4f}  PR-AUC: {pr_auc:.4f}")
        print(f"{'=' * 65}")
        print(f"  {'Метод':<20} {'Порог':>10} {'Recall':>8} {'Precision':>10} {'F2':>8}")
        print(f"  {'-' * 60}")
        print(f"  {'target_recall=0.98':<20} {thresh_1:>10.6f} {rec_1:>8.4f} {prec_1:>10.4f} {f2_1:>8.4f}")
        print(f"  {'youden_index':<20} {thresh_2:>10.6f} {rec_2:>8.4f} {prec_2:>10.4f} {f2_2:>8.4f}")
        print(f"  {'max_f2':<20} {thresh_3:>10.6f} {rec_3:>8.4f} {prec_3:>10.4f} {f2_3:>8.4f}")
        print(f"{'=' * 65}\n")


        threshold_rows = [
            {"exp_id": exp_id, "method": "target_recall",
            "threshold": thresh_1, "recall": rec_1, "precision": prec_1,
            "f2": f2_1, "roc_auc": roc_auc, "pr_auc": pr_auc},
            {"exp_id": exp_id, "method": "youden_index",
            "threshold": thresh_2, "recall": rec_2, "precision": prec_2,
            "f2": f2_2, "roc_auc": roc_auc, "pr_auc": pr_auc},
            {"exp_id": exp_id, "method": "max_f2",
            "threshold": thresh_3, "recall": rec_3, "precision": prec_3,
            "f2": f2_3, "roc_auc": roc_auc, "pr_auc": pr_auc},
        ]

     
        final_metrics = {
            "method":    "max_f2",
            "threshold": thresh_3,
            "recall":    rec_3,
            "precision": prec_3,
            "f2":        f2_3,
            "roc_auc":   roc_auc,
            "pr_auc":    pr_auc,
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(fpr, tpr, lw=3, label=f'ROC AUC: {roc_auc:.3f}', color='darkorange')
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.set_title('ROC Curve')
            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax2.plot(recall_p, precision_p, lw=3, label=f'PR AUC: {pr_auc:.3f}', color='dodgerblue')
            ax2.set_title('Precision-Recall Curve')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"curves_{exp_id}.png"))
            plt.close()

        return final_metrics, threshold_rows
    
    def specificity_evaluation(self, labels, scores, save_path=None, exp_id=0):


        def calc_metrics(thresh):
            preds = (scores >= thresh).astype(int)
            cm = confusion_matrix(labels, preds, labels=[0, 1])
            tn, fp = cm[0, 0], cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            return  specificity

 
        #thresh_3    = np.max(scores)
        thresh_3 = np.percentile(scores, 90)
        specificity = calc_metrics(thresh_3)

        print(f"  {'max_sp':<20} {thresh_3:>10.6f} ")


        final_metrics = {
            "method":    "max_specificity",
            "threshold": thresh_3,
            "specificity":    specificity,

        }
        return final_metrics
    
    def treshold_eval(self,losses, labels,exp_id):
        
        losses = np.array(losses)
        labels = np.array(labels)

        normal_losses  = losses[labels == 0]
        def calc_all_metrics(thresh):
            preds = (losses >= thresh).astype(int)
            
            tp = np.sum((preds == 1) & (labels == 1))
            fp = np.sum((preds == 1) & (labels == 0))
            fn = np.sum((preds == 0) & (labels == 1))
            tn = np.sum((preds == 0) & (labels == 0))
            
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            f2 = fbeta_score(labels, preds, beta=2, zero_division=0)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            return rec, prec, f2, specificity

        fpr, tpr, roc_thresholds   = roc_curve(labels, losses)
        precision_p, recall_p, pr_thresholds = precision_recall_curve(labels, losses)

        target_recall = 0.98
        valid_idx = np.where(recall_p[:-1] >= target_recall)[0]
        thresh_1  = pr_thresholds[valid_idx[np.argmax(precision_p[valid_idx])]] if len(valid_idx) > 0 \
                    else pr_thresholds[np.argmax(recall_p[:-1])]
        rec_1, prec_1, f2_1, sp_1 = calc_all_metrics(thresh_1)

    
        thresh_2 = roc_thresholds[np.argmax(tpr - fpr)]
        rec_2, prec_2, f2_2,sp_2 = calc_all_metrics(thresh_2)

        f2_scores   = [fbeta_score(labels, (losses>= t).astype(int), beta=2, zero_division=0)
                    for t in pr_thresholds]
        best_idx    = np.argmax(f2_scores)
        thresh_3    = pr_thresholds[best_idx]
        rec_3, prec_3, f2_3,sp_3 = calc_all_metrics(thresh_3)

        thresh_4 = np.percentile(normal_losses, 90)
        rec_4, prec_4, f2_4,sp_4 = calc_all_metrics(thresh_4)


        threshold_rows = [
            {"exp_id": exp_id, "method": "target_recall",
            "threshold": thresh_1, "recall": rec_1, "precision": prec_1,
            "f2": f2_1,"specificity": sp_1},
            {"exp_id": exp_id, "method": "youden_index",
            "threshold": thresh_2, "recall": rec_2, "precision": prec_2,
            "f2": f2_2,"specificity": sp_2},
            {"exp_id": exp_id, "method": "max_f2",
            "threshold": thresh_3, "recall": rec_3, "precision": prec_3,
            "f2": f2_3,"specificity": sp_3},
            {"exp_id": exp_id, "method": "normal_percentile_90",
            "threshold": thresh_4, "recall": rec_4, "precision": prec_4,
            "f2": f2_4,"specificity": sp_4},

        ]
        return threshold_rows


    def plot_hist(self,normal_losses, anomaly_losses, save_path,exp_id):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 5))
        
        ax.hist(normal_losses, bins=50, alpha=0.4, color='steelblue', 
                density=True, label='Норма')
        ax.hist(anomaly_losses, bins=50, alpha=0.4, color='red', 
                density=True, label='Аномалия')
        
        kde_norm = stats.gaussian_kde(normal_losses)
        kde_anom = stats.gaussian_kde(anomaly_losses)
        x = np.linspace(
            min(normal_losses.min(), anomaly_losses.min()),
            max(normal_losses.max(), anomaly_losses.max()), 
            500
        )
        ax.plot(x, kde_norm(x), color='steelblue', linewidth=2)
        ax.plot(x, kde_anom(x), color='red', linewidth=2)
        
        ax.set_xlabel("Ошибка реконструкции")
        ax.set_ylabel("Плотность")
        ax.set_title(f"Hists_{exp_id}")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"hist_{exp_id}.png"))
        plt.close()

    def plots_curve(self,labels, scores,save_path,exp_id):
        fpr, tpr, roc_thresholds   = roc_curve(labels, scores)
        precision_p, recall_p, pr_thresholds = precision_recall_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        pr_auc  = auc(recall_p, precision_p)

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            ax1.plot(fpr, tpr, lw=3, label=f'ROC AUC: {roc_auc:.3f}', color='darkorange')
            ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
            ax1.set_title('ROC Curve')
            ax1.set_xlabel('FPR')
            ax1.set_ylabel('TPR')
            ax1.legend()
            ax1.grid(alpha=0.3)
            ax2.plot(recall_p, precision_p, lw=3, label=f'PR AUC: {pr_auc:.3f}', color='dodgerblue')
            ax2.set_title('Precision-Recall Curve')
            ax2.set_xlabel('Recall')
            ax2.set_ylabel('Precision')
            ax2.legend()
            ax2.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"curves_{exp_id}.png"))
            plt.close()
        return roc_auc,pr_auc

    def overlap_coefficient(self,losses_norm, losses_anom):
        kde_norm = stats.gaussian_kde(losses_norm)
        kde_anom = stats.gaussian_kde(losses_anom)
        
        x = np.linspace(
            min(losses_norm.min(), losses_anom.min()),
            max(losses_norm.max(), losses_anom.max()),
            1000
        )

        overlap = np.trapezoid(np.minimum(kde_norm(x), kde_anom(x)), x)
        return overlap 

    def ks(self,normal_scores,anomaly_scores):
        ks_stat, p_value = ks_2samp(
            normal_scores,
            anomaly_scores
        )
        return ks_stat,p_value


    def separability_evaluation(self,
                                losses_norm,
                                losses_anom,
                                save_path_hist=None,
                                save_path_curve=None, 
                                exp_id=0):
        
        self.plot_hist(losses_norm, losses_anom,save_path_hist,exp_id)

        combined_scores = np.concatenate([losses_norm, losses_anom])
        combined_labels = np.concatenate([np.zeros_like(losses_norm), np.ones_like(losses_anom)])

        roc_auc,pr_auc = self.plots_curve(combined_labels, combined_scores,save_path_curve,exp_id)
        
        overlap = self.overlap_coefficient(losses_norm, losses_anom)
        ks = self.ks(losses_norm,losses_anom)

        final_metrics = {
            "roc_auc": roc_auc,
            "pr_auc" : pr_auc,
            "overlap" : overlap,
            "ks" : ks

        }
        return final_metrics
    
