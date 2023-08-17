import lightning as pl
from sklearn.metrics import (
    balanced_accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
import pandas as pd 
import numpy as np 

def get_metrics(predictions, target, healthy_threshold):
    predictions = np.nan_to_num(predictions)
    target = np.nan_to_num(target)

    rmse = round(mean_squared_error(target, predictions, squared=False), 2)
    mae = round(mean_absolute_error(target, predictions), 2)

    threshold_target = []
    threshold_prediction = []

    for i in range(len(target)):
        threshold_target.append(int(target[i] < healthy_threshold))
        threshold_prediction.append(int(predictions[i] < healthy_threshold))

    ba = round(balanced_accuracy_score(threshold_target, threshold_prediction), 2)
    return mae, rmse, ba


def create_confusion_scatter(preds, labels, model_class, model_name, output_file):
    preds_b = [x < 0.9 for x in preds]
    labels_b = [x < 0.9 for x in labels]
    MAE, RMSE, BA = get_metrics(preds, labels, 0.9)

    print(
        classification_report(
            labels_b, 
            preds_b, 
            target_names=['PAD Negative', 'PAD Positive']
        )
    )

    # make scatter plot square
    df = pd.DataFrame({'preds': preds, 'labels': labels})
    # change trendline color to green
    fig = px.scatter(
      df, 
      x="labels", 
      y="preds", 
      template='plotly_dark', 
      title = f"""
          {model_name}
          <br><sup>{model_class}</sup>
          <br><sup>{MAE = }\t{RMSE = }\t{BA = }</sup>
      """
    )

    fig.update_layout(
        autosize=False,
        width=500,
        height=500,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )

    fig.add_hline(y=0.9, line_width=1, line_dash="dash", line_color="red")
    fig.add_vline(x=0.9, line_width=1, line_dash="dash", line_color="red")
    fig.add_trace(
        go.Scatter(
            x=[0, 1.8],
            y=[0, 1.8],
            mode='lines',
            line=dict(
                color='white', 
                width=1
            )
        ),
    )
    pio.write_image(fig, output_file, format='png', scale=2)



class ConfusionScatterCallback(pl.Callback):
    def __init__(self, healthy_threshold=0.9, output_file='output.png'):
        super().__init__()
        self.healthy_threshold = healthy_threshold
        self.output_file = output_file

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get the validation data loader
        val_dataloader = pl_module.val_dataloader()

        # Initialize the predictions and labels lists
        preds, labels = [], []

        # Perform predictions on the validation data
        with torch.no_grad():
            for batch in val_dataloader:
                x, y = batch
                y_pred = pl_module(x)
                preds.append(y_pred)
                labels.append(y)

        # Concatenate the predictions and labels
        preds = torch.cat(preds).cpu().numpy()
        labels = torch.cat(labels).cpu().numpy()

        # Call the create_confusion_scatter function to generate and save the scatter plot
        create_confusion_scatter(
            preds, 
            labels, 
            pl_module.__class__.__name__, 
            f"{pl_module.__class__.__name__} Epoch {trainer.current_epoch}", 
            self.output_file
        )
