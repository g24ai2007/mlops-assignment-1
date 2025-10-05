from sklearn.tree import DecisionTreeRegressor
from misc import load_data, split_data, train_and_evaluate

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
model = DecisionTreeRegressor(random_state=42)
mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
print(f"Decision Tree MSE: {mse:.2f}")