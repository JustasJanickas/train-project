import json

from tensorflow import keras

class EnsembleModel:
    def __init__(self, dir_path):

        self._dir_path = dir_path
        if not self._dir_path.endswith("/"):
            self._dir_path += "/"

        self._models_ensemble = []

    def get_weighted_by_kernel(self, values, kernel=[16, 4, 1]):
        weighted_values = []
        for i in range(len(values)):
            lower_bound = max(0, i - (len(kernel) - 1))
            upper_bound = min(len(values) - 1, i + (len(kernel) - 1))

            sublist = values[lower_bound:upper_bound + 1]

            weights = [(loss, kernel[abs(i - (idx + lower_bound))]) for idx, loss in enumerate(sublist)]
            weighted_values.append(
                sum([loss * weight for loss, weight in weights]) / sum([weight for _, weight in weights]))

        return weighted_values

    def form_ensemble(self):
        with open(f"{self._dir_path}loss_metrics_1500_1500.json", 'r') as f:
            loss_metrics = json.load(f)

        for i in range(5):
            validation_losses = [val_loss for _, val_loss in loss_metrics[i * 100:(i + 1) * 100]]
            weighted_losses = self.get_weighted_by_kernel(validation_losses)

            best_epoch = (weighted_losses.index(min(weighted_losses)) + 1) * 3 + i * 300
            self._models_ensemble.append(keras.models.load_model(f"{self._dir_path}ResNet_model_{best_epoch}.keras"))

    def predict_single(self, image, verbose=1):
        preds = []
        for model in self._models_ensemble:
            preds.append(model.predict(image, verbose=verbose)[0][0])

        return sum(preds) / len(preds)
