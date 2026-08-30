from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder


# A linear probe here is just saying, can we linearly seperate the factors
def probe(M, y, name):
    """5-fold CV accuracy of a linear probe against chance"""
    # encode our labels y
    yi = LabelEncoder().fit_transform(y)
    k = len(set(yi))
    # fit a linear probe f(x) = Ax + b with 5 folds (5 chunks of train and tests) composed with softmax
    # "logistic" because it models log-odds as linear: log(p/(1-p)) = Ax + b
    # then use log-loss to train, capped at 3000 iters.
    # then average the accuracy
    acc = cross_val_score(LogisticRegression(max_iter=3000), M, yi,
                            cv=5).mean()
    # silhouette measures clustering by pairwise distances
    # for each point i:
    #   a(i) = mean distance to the other points in its OWN group
    #   b(i) = mean distance to the nearest OTHER group
    #   s(i) = (b - a) / max(a, b),  averaged over all points.
    # near 1  means own group far closer than any other
    #  0  means it's in no man's land
    # near -1 means it's closer to another group than to its own
    # Note this is a different question from the probe's, that is,  classes can be
    # perfectly separable by a hyperplane and still score near 0 here
    # (think about long parallel stripes, not round blobs)
    sil = silhouette_score(M, yi)
    print(f"  {name:<10} k={k}  acc={acc:.3f}  chance={1/k:.3f}  "
            f"lift={acc - 1/k:+.3f}  silhouette={sil:+.3f}")