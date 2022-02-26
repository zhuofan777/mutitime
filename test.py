from sktime.datasets import load_unit_test
from sktime.transformations.panel.shapelet_transform import RandomShapeletTransform

X_train, y_train = load_unit_test(split="train", return_X_y=True)
t = RandomShapeletTransform(
    n_shapelet_samples=500,
    max_shapelets=10,
    batch_size=100, )
t.fit(X_train, y_train)
RandomShapeletTransform(...)
X_t = t.transform(X_train)
print(type(X_t))
