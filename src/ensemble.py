import numpy as np
from sklearn.metrics import accuracy_score


def blend_oof(oof_preds_list, y_train):
    n_models = len(oof_preds_list)
    best_acc = 0
    best_weights = None

    for w1 in np.arange(0, 1.01, 0.05):
        for w2 in np.arange(0, 1.01 - w1 + 0.01, 0.05):
            if n_models == 2:
                weights = [w1, 1 - w1]
            else:
                w3 = 1 - w1 - w2
                if w3 < 0:
                    continue
                weights = [w1, w2, w3]

            blended = np.zeros_like(oof_preds_list[0])
            for preds, w in zip(oof_preds_list, weights):
                blended += w * preds

            acc = accuracy_score(y_train, np.argmax(blended, axis=1))
            if acc > best_acc:
                best_acc = acc
                best_weights = weights

    print(f"  Best blend weights: {[f'{w:.2f}' for w in best_weights]}", flush=True)
    print(f"  Best blend OOF Accuracy: {best_acc:.5f}", flush=True)
    return best_weights


def blend_ensemble(preds_list, weights):
    blended = np.zeros_like(preds_list[0])
    for preds, w in zip(preds_list, weights):
        blended += w * preds
    return blended
