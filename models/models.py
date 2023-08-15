import os
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchvision
import torchmetrics
from pytorch_lightning.core.lightning import LightningModule

import utils.utils as utils
from constants import Losses, IGNORE_INDEX
from models.loss import CrossEntropyLoss
from utils import metrics
from models import modules


##############################################################################################
#                                                                                            #
#  Pytorch Lightning ERFNet implementation from Jan Weyler. Our Bayesian-ERFNet              #
#  implementation builds upon Jan's ERFNet implementation.                                   #
#                                                                                            #
##############################################################################################


class NetworkWrapper(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_classes = self.cfg["model"]["num_classes"]
        self.epistemic_version = "standard"
        if "epistemic_version" in self.cfg["model"]:
            self.epistemic_version = self.cfg["model"]["epistemic_version"]

        self.in_channels = self.cfg["model"]["in_channels"]
        self.dropout_prob = self.cfg["model"]["dropout_prob"]
        self.deep_encoder = self.cfg["model"]["deep_encoder"]
        self.shared_decoder = self.cfg["model"]["shared_decoder"]
        self.aleatoric_model = self.cfg["model"]["aleatoric_model"]

        self.num_mc_aleatoric = self.cfg["train"]["num_mc_aleatoric"]
        self.num_mc_epistemic = self.cfg["train"]["num_mc_epistemic"]

        self.ignore_index = IGNORE_INDEX[self.cfg["data"]["name"]]

        self.test_evaluation_metrics = {}
        self.vis_interval = 0

        self.inv_class_frequencies = None
        if "class_frequencies" in self.cfg["model"]:
            class_frequencies = torch.Tensor(self.cfg["model"]["class_frequencies"])
            self.inv_class_frequencies = class_frequencies.sum() / class_frequencies
            self.inv_class_frequencies = self.inv_class_frequencies.to(self.device)

    @staticmethod
    def identity_output_fn(x):
        identity_fn = nn.Identity()
        return identity_fn(x)

    @property
    def output_fn(self):
        return self.identity_output_fn

    @property
    def loss_fn(self):
        loss_name = self.cfg["model"]["loss"]
        if loss_name == Losses.CROSS_ENTROPY:
            return CrossEntropyLoss(
                ignore_index=self.ignore_index
                if self.ignore_index is not None
                else -100,
                weight=self.inv_class_frequencies,
            )
        else:
            raise RuntimeError(f"Loss {loss_name} not available!")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg["train"]["lr"],
            weight_decay=self.weight_decay,
        )
        return optimizer

    def replace_output_layer(self):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        self.track_evaluation_metrics(
            conf_matrices,
            stage="Validation",
        )
        self.track_confusion_matrix(conf_matrices, stage="Validation")
        self.vis_interval = 0

    def test_epoch_end(self, outputs):
        conf_matrices = [tmp["conf_matrix"] for tmp in outputs]
        calibration_info_list = [tmp["calibration_info"] for tmp in outputs]

        self.test_evaluation_metrics = self.track_evaluation_metrics(
            conf_matrices,
            stage="Test",
            calibration_info_list=calibration_info_list,
        )

        self.track_confusion_matrix(conf_matrices, stage="Test")
        self.track_epistemic_uncertainty_stats(outputs, stage="Test")

        fig_ = metrics.compute_calibration_plots(outputs)
        self.logger.experiment.add_figure(
            "UncertaintyStats/Calibration", fig_, self.current_epoch
        )

    def track_evaluation_metrics(
        self,
        conf_matrices,
        stage="Test",
        calibration_info_list=None,
    ):
        miou = metrics.mean_iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        per_class_iou = metrics.per_class_iou_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        accuracy = metrics.accuracy_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        precision = metrics.precision_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        recall = metrics.recall_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )
        f1_score = metrics.f1_score_from_conf_matrices(
            conf_matrices, ignore_index=IGNORE_INDEX[self.cfg["data"]["name"]]
        )

        ece = -1.0
        if calibration_info_list is not None:
            ece = metrics.ece_from_calibration_info(calibration_info_list, num_bins=20)

        self.log(f"{stage}/Precision", precision)
        self.log(f"{stage}/Recall", recall)
        self.log(f"{stage}/F1-Score", f1_score)
        self.log(f"{stage}/Acc", accuracy)
        self.log(f"{stage}/mIoU", miou)
        self.log(f"{stage}/ECE", ece)

        return {
            f"{stage}/Precision": precision,
            f"{stage}/Recall": recall,
            f"{stage}/F1-Score": f1_score,
            f"{stage}/Acc": accuracy,
            f"{stage}/mIoU": miou,
            f"{stage}/Per-Class-IoU": per_class_iou.tolist(),
            f"{stage}/ECE": ece,
        }

    def track_confusion_matrix(self, conf_matrices, stage="Validation"):
        total_conf_matrix = metrics.total_conf_matrix_from_conf_matrices(conf_matrices)
        df_cm = pd.DataFrame(
            total_conf_matrix.cpu().numpy(),
            index=range(self.num_classes),
            columns=range(self.num_classes),
        )

        ax = sns.heatmap(df_cm, annot=True, cmap="Spectral")
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground Truth")

        self.logger.experiment.add_figure(
            f"ConfusionMatrix/{stage}", ax.get_figure(), self.current_epoch
        )

        plt.close()
        plt.clf()
        plt.cla()

    def track_epistemic_uncertainty_stats(self, outputs, stage="Validation"):
        per_class_ep_uncertainties = torch.stack(
            [tmp["per_class_ep_uncertainty"] for tmp in outputs]
        )
        per_class_ep_uncertainty = torch.mean(per_class_ep_uncertainties, dim=0)

        ax = sns.barplot(
            x=list(range(self.num_classes)), y=per_class_ep_uncertainty.tolist()
        )
        ax.set_xlabel("Class Index")
        ax.set_ylabel("Model Uncertainty [0,1]")

        if stage == "Test" and self.active_learning:
            plt.savefig(
                os.path.join(
                    self.al_logger_name,
                    f"per_class_ep_uncertainty_{self.al_iteration}.png",
                ),
                dpi=300,
            )

        self.logger.experiment.add_figure(
            f"UncertaintyStats/{stage}/EpistemicPerClass",
            ax.get_figure(),
            self.current_epoch,
        )

        plt.close()
        plt.clf()
        plt.cla()

    def track_gradient_norms(self):
        total_grad_norm = 0
        for params in self.model.parameters():
            if params.grad is not None:
                total_grad_norm += params.grad.data.norm(2).item()

        self.log(f"LossStats/GradientNorm", total_grad_norm)

    def track_predictions(
        self,
        images,
        hard_predictions,
        prob_predictions,
        targets,
        stage="Train",
        step=0,
        epistemic_uncertainties=None,
        dist=None,
    ):
        sample_img_out = hard_predictions[:1]
        sample_img_out = utils.toOneHot(sample_img_out, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Output image",
            torch.from_numpy(sample_img_out),
            step,
            dataformats="HWC",
        )
        sample_img_in = images[:1]
        sample_anno = targets[:1]

        self.logger.experiment.add_image(
            f"{stage}/Input image", sample_img_in.squeeze(), step, dataformats="CHW"
        )

        sample_prob_prediction = prob_predictions[:1]
        cross_entropy_fn = CrossEntropyLoss(reduction="none")
        sample_error_img = cross_entropy_fn(
            sample_prob_prediction, sample_anno
        ).squeeze()

        sizes = sample_img_out.shape
        px = 1 / plt.rcParams["figure.dpi"]
        fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
        ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
        ax.set_axis_off()
        ax.imshow(sample_error_img.cpu().numpy(), cmap="gray")
        fig.add_axes(ax)
        self.logger.experiment.add_figure(f"{stage}/Error image", fig, step)
        plt.cla()
        plt.clf()

        sample_anno = utils.toOneHot(sample_anno, self.cfg["data"]["name"])
        self.logger.experiment.add_image(
            f"{stage}/Annotation",
            torch.from_numpy(sample_anno),
            step,
            dataformats="HWC",
        )
        if epistemic_uncertainties is not None:
            sample_ep_uncertainty = epistemic_uncertainties.cpu().numpy()[0, :, :]
            sizes = sample_ep_uncertainty.shape
            fig = plt.figure(figsize=(px * sizes[1], px * sizes[0]))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_ep_uncertainty, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(
                f"{stage}/Uncertainty/Epistemic", fig, step
            )
            plt.cla()
            plt.clf()

        if dist is not None:
            sample_aleatoric_unc_out = self.compute_aleatoric_uncertainty(
                dist[0], dist[1]
            )[0, :, :]
            fig = plt.figure(figsize=(px * sizes[0], px * sizes[1]))
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            ax.imshow(sample_aleatoric_unc_out, cmap="plasma")
            fig.add_axes(ax)
            self.logger.experiment.add_figure(
                f"{stage}/Uncertainty/Aleatoric", fig, step
            )
            plt.cla()
            plt.clf()

    def compute_aleatoric_uncertainty(self, seg, std):
        pass

    def compute_per_class_epistemic_uncertainty(self, preds, uncertainty_predictions):
        per_class_ep_uncertainty = torch.zeros(self.num_classes)
        for predicted_class in torch.unique(preds):
            mask = preds == predicted_class
            per_class_ep_uncertainty[predicted_class.item()] = torch.mean(
                uncertainty_predictions[mask]
            ).item()

        return per_class_ep_uncertainty

    def common_test_step(self, batch, batch_idx, aleatoric_dist=None):
        targets = batch["anno"].to(self.device)
        (
            mean_predictions,
            uncertainty_predictions,
            hidden_representations,
        ) = utils.get_predictions(
            self,
            batch,
            num_mc_dropout=self.num_mc_epistemic,
            aleatoric_model=self.aleatoric_model,
            num_mc_aleatoric=self.num_mc_aleatoric,
            ensemble_model=False,
            device=self.device,
            task="classification",
        )
        mean_predictions, uncertainty_predictions = torch.from_numpy(
            mean_predictions
        ).to(self.device), torch.from_numpy(uncertainty_predictions).to(self.device)

        _, hard_preds = torch.max(mean_predictions, dim=1)

        loss = self.get_loss(mean_predictions, targets)
        self.log("Test/Loss", loss, prog_bar=True)

        confusion_matrix = None
        calibration_info = None
        per_class_epistemic_uncertainty = None
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            hard_preds, targets, num_classes=self.num_classes, normalize=None
        )
        calibration_info = metrics.compute_calibration_info(
            mean_predictions, targets, num_bins=20
        )
        per_class_epistemic_uncertainty = self.compute_per_class_epistemic_uncertainty(
            hard_preds, uncertainty_predictions
        )

        self.track_predictions(
            batch["data"],
            hard_preds,
            mean_predictions,
            targets,
            stage="Test",
            step=batch_idx,
            epistemic_uncertainties=uncertainty_predictions,
            dist=aleatoric_dist,
        )

        return {
            "conf_matrix": confusion_matrix,
            "loss": loss,
            "per_class_ep_uncertainty": per_class_epistemic_uncertainty,
            "calibration_info": calibration_info,
        }

    @property
    def weight_decay(self):
        return self.cfg["train"]["weight_decay"]

        # return (1 - self.cfg["model"]["dropout_prob"]) / (2 * self.num_train_data)


class ERFNet(NetworkWrapper):
    def __init__(self, cfg):
        super(ERFNet, self).__init__(cfg)
        self.save_hyperparameters()

        self.model = modules.ERFNetModel(
            self.num_classes,
            self.in_channels,
            dropout_prop=self.dropout_prob,
            deep_encoder=self.deep_encoder,
            epistemic_version=self.epistemic_version,
            output_fn=self.output_fn,
        )

    def replace_output_layer(self):
        self.model.decoder.output_conv = nn.ConvTranspose2d(
            16, self.num_classes, 2, stride=2, padding=0, output_padding=0, bias=True
        )

    def get_loss(self, x, y):
        return self.loss_fn(x, y)

    def forward(self, x):
        out, hidden_representation = self.model(x)
        return out, hidden_representation

    def training_step(self, batch, batch_idx):
        out, _ = self.forward(batch["data"])
        loss = self.get_loss(out, batch["anno"])

        self.log("train:loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        out, _ = self.forward(batch["data"])
        loss = self.get_loss(out, batch["anno"])
        self.log("Validation/Loss", loss, prog_bar=True)

        confusion_matrix = None
        _, preds = torch.max(out, dim=1)
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, batch["anno"], num_classes=self.num_classes, normalize=None
        )

        if batch_idx % 50 == 0:
            self.visualize_step(batch)
        return {"conf_matrix": confusion_matrix, "loss": loss}

    def test_step(self, batch, batch_idx):
        return self.common_test_step(batch, batch_idx, aleatoric_dist=None)

    def visualize_step(self, batch):
        self.vis_interval += 1
        targets = batch["anno"].to(self.device)
        (
            mean_predictions,
            uncertainty_predictions,
            hidden_representations,
        ) = utils.get_predictions(
            self,
            batch,
            num_mc_dropout=self.num_mc_epistemic,
            aleatoric_model=self.aleatoric_model,
            num_mc_aleatoric=self.num_mc_aleatoric,
            ensemble_model=False,
            device=self.device,
            task="classification",
        )
        mean_predictions, uncertainty_predictions = torch.from_numpy(
            mean_predictions
        ).to(self.device), torch.from_numpy(uncertainty_predictions).to(self.device)

        _, hard_preds = torch.max(mean_predictions, dim=1)

        self.track_predictions(
            batch["data"],
            hard_preds,
            mean_predictions,
            targets,
            stage="Validation",
            step=self.vis_interval,
            epistemic_uncertainties=uncertainty_predictions,
            dist=None,
        )


class AleatoricERFNet(NetworkWrapper):
    def __init__(self, cfg):
        super(AleatoricERFNet, self).__init__(cfg)

        self.save_hyperparameters()
        self.model = modules.AleatoricERFNetModel(
            self.num_classes,
            self.in_channels,
            self.dropout_prob,
            use_shared_decoder=self.shared_decoder,
            deep_encoder=self.deep_encoder,
            epistemic_version=self.epistemic_version,
            output_fn=self.output_fn,
        )

    def replace_output_layer(self):
        if self.model.use_shared_decoder:
            self.model.shared_decoder.output_conv = nn.ConvTranspose2d(
                16,
                self.num_classes + 1,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True,
            )
        else:
            self.model.segmentation_decoder.output_conv = nn.ConvTranspose2d(
                16,
                self.num_classes,
                2,
                stride=2,
                padding=0,
                output_padding=0,
                bias=True,
            )

    def get_loss(self, seg, std, true_seg):
        sampled_predictions = torch.zeros(
            (self.num_mc_aleatoric, *seg.size()), device=self.device
        )

        for i in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(seg.size(), device=self.device)
            noise_std = torch.ones(seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = seg + torch.mul(std, epsilon)
            sampled_predictions[i] = sampled_seg

        mean_prediction = torch.mean(sampled_predictions, dim=0)
        return self.loss_fn(mean_prediction, true_seg)

    def forward(self, x):
        output_seg, output_std, hidden_representation = self.model(x)
        return output_seg, output_std, hidden_representation

    def compute_aleatoric_uncertainty(self, seg, std):
        predictions = []
        softmax = nn.Softmax(dim=1)
        for i in range(self.num_mc_aleatoric):
            noise_mean = torch.zeros(seg.size(), device=self.device)
            noise_std = torch.ones(seg.size(), device=self.device)
            epsilon = torch.distributions.normal.Normal(noise_mean, noise_std).sample()
            sampled_seg = seg + torch.mul(std, epsilon)
            predictions.append(softmax(sampled_seg).cpu().numpy())

        mean_predictions = np.mean(predictions, axis=0)
        return -np.sum(
            mean_predictions * np.log(mean_predictions + sys.float_info.min), axis=1
        )

    def track_aleatoric_stats(self, std):
        self.log("Variance/TrainMin", torch.min(std))
        self.log("Variance/TrainMax", torch.max(std))
        self.log("Variance/TrainMean", torch.mean(std))

    def training_step(self, batch, batch_idx):
        est_seg, est_std, _ = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])

        self.track_aleatoric_stats(est_std)
        self.track_gradient_norms()
        self.log("train:loss", loss)

        return loss

    def validation_step(self, batch, batch_idx):
        est_seg, est_std, _ = self.forward(batch["data"])
        loss = self.get_loss(est_seg, est_std, batch["anno"])
        _, preds = torch.max(est_seg, dim=1)

        self.log("Validation/Loss", loss, prog_bar=True)
        self.track_aleatoric_stats(est_std)

        confusion_matrix = None
        _, preds = torch.max(est_seg, dim=1)
        confusion_matrix = torchmetrics.functional.confusion_matrix(
            preds, batch["anno"], num_classes=self.num_classes, normalize=None
        )

        return {"conf_matrix": confusion_matrix, "loss": loss}

    def test_step(self, batch, batch_idx):
        est_seg, est_std, _ = self.forward(batch["data"])

        return self.common_test_step(batch, batch_idx, (est_seg, est_std))
